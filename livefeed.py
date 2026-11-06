"""Live transaction feed for the control room: Kafka (local Docker broker) with an in-process fallback.

  producer.py --(topic txn-live: one JSON message per transaction + an end-of-day marker)--> KafkaFeed --> LiveEngine
  KafkaFeed --(topic txn-control: ack / attack / stop)--> producer.py
  LocalFeed = the same DayGen running inside the app, used when Kafka is not reachable.

Transactions are new: real March rows (the 'normal' world) resampled onto each new simulated day with fresh IDs;
label timing (whether / when the label arrives) is taken from the Apr-Jun regime. "Launch attack" makes a share of
the day's thieves copy an honest customer's velocity profile (same idea as monitor.make_evasion).
Kafka broker:  docker run -d --name kafka -p 9092:9092 apache/kafka:3.9.1
"""
import collections
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E

M = E.M
HERE = Path(__file__).resolve().parent
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TXN, CTL = "txn-live", "txn-control"
AHEAD = 3   # producer stays at most this many days ahead of what the app has consumed
COLS = list(dict.fromkeys(["transaction_id", "transaction_datetime", "is_fraud", "label_available", "label_delay_days"]
                          + M.feats))
DTYPES = M.base[COLS].dtypes.to_dict()


def from_records(recs):
    """JSON rows -> DataFrame with exactly the training dtypes (category columns keep M.base's categories)."""
    df = pd.DataFrame(recs, columns=COLS)
    ts = pd.to_datetime(df["transaction_datetime"])
    df["transaction_datetime"] = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
    return df.astype(DTYPES)


def to_records(df):
    return json.loads(df[COLS].to_json(orient="records", date_format="iso"))


class DayGen:
    """Creates one simulated day of new transactions at a time."""

    def __init__(self, seed=None, start=M.STREAM_START):
        self.rng = np.random.default_rng(seed)
        b = M.base
        self.march = b[(b["day"] >= M.REF_START) & (b["day"] < M.REF_END)]
        self.honest = self.march[self.march["is_fraud"] == 0]
        self.per_day = len(self.march) / self.march["day"].nunique()
        # March labels arrive the same day; use the Apr-Jun label regime (median delay ~21 days, ~80% arrive).
        self.lab = b[b["day"] >= M.STREAM_START][["is_fraud", "label_available", "label_delay_days"]]
        self.day = pd.Timestamp(start)
        self.attack, self.share = False, M.EVADE_SHARE
        self.run = uuid.uuid4().hex[:6]
        self.n = 0

    def _rs(self):
        return int(self.rng.integers(1_000_000_000))

    def next_day(self):
        day, attack = self.day, self.attack
        n = int(self.rng.poisson(self.per_day))
        # without replacement: no identical twin transactions on the same day
        d = self.march.sample(min(n, len(self.march)), replace=False, random_state=self._rs())[COLS].copy()
        n = len(d)
        d["transaction_datetime"] = day + (d["transaction_datetime"] - d["transaction_datetime"].dt.floor("D"))
        d["transaction_id"] = [f"LIVE-{self.run}-{self.n + k}" for k in range(n)]
        self.n += n
        for f in (0, 1):
            msk = d["is_fraud"].values == f
            src = self.lab[self.lab["is_fraud"] == f].sample(int(msk.sum()), replace=True, random_state=self._rs())
            d.loc[msk, ["label_available", "label_delay_days"]] = src[["label_available", "label_delay_days"]].values
        if attack:
            fr = np.flatnonzero(d["is_fraud"].values == 1)
            pick = self.rng.choice(fr, int(len(fr) * self.share), replace=False)
            donors = self.honest.sample(len(pick), replace=True, random_state=self._rs())
            d.iloc[pick, [d.columns.get_loc(c) for c in M.VELOCITY]] = donors[M.VELOCITY].values
        self.day += pd.Timedelta(days=1)
        return day, d.reset_index(drop=True), attack


class LocalFeed:
    mode = "local"

    def __init__(self, seed=None):
        self.gen = DayGen(seed)
        self.received = 0

    def poll_day(self, timeout=1.0):
        day, d, attack = self.gen.next_day()
        self.received += len(d)
        return day, from_records(to_records(d)), attack   # same JSON round trip as Kafka

    def ack(self, n):
        pass

    def set_attack(self, on, share):
        self.gen.attack, self.gen.share = on, share

    def status(self):
        return {"mode": "local", "received": self.received, "attack": self.gen.attack}

    def close(self):
        pass


def kafka_up(timeout=2.0):
    try:
        from confluent_kafka.admin import AdminClient
        AdminClient({"bootstrap.servers": BOOTSTRAP, "log_level": 0}).list_topics(timeout=timeout)
        return True
    except Exception:
        return False


class KafkaFeed:
    mode = "kafka"

    def __init__(self, seed=None):
        from confluent_kafka import Consumer, Producer, TopicPartition
        from confluent_kafka.admin import AdminClient, NewTopic
        admin = AdminClient({"bootstrap.servers": BOOTSTRAP, "log_level": 0})
        for f in admin.create_topics([NewTopic(TXN, 1, 1, config={"retention.ms": "900000"}), NewTopic(CTL, 1, 1)]).values():
            try:
                f.result(5)
            except Exception:
                pass   # topic already exists
        self.run = uuid.uuid4().hex[:8]
        self.tp = TopicPartition(TXN, 0)
        self.c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "cr-" + self.run, "enable.auto.commit": False,
                           "log_level": 0})
        lo, hi = self.c.get_watermark_offsets(self.tp, timeout=3)
        self.c.assign([TopicPartition(TXN, 0, hi)])   # start at the end, before the producer starts: no missed messages
        self.p = Producer({"bootstrap.servers": BOOTSTRAP, "log_level": 0})
        self.pending, self.ready = [], collections.deque()
        self.received, self.attack, self.share = 0, False, M.EVADE_SHARE
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "producer.py"), "--run", self.run, "--seed", str(seed if seed is not None else -1)],
            cwd=str(HERE), stdout=open(HERE / "producer.log", "w"), stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def poll_day(self, timeout=1.0):
        end = time.time() + timeout
        while not self.ready and time.time() < end:
            for m in self.c.consume(2000, 0.1):
                if m.error():
                    continue
                msg = json.loads(m.value())
                if msg.get("run") != self.run:
                    continue
                self.received += 1
                if msg["t"] == "txn":
                    self.pending.append(msg["row"])
                else:
                    self.ready.append((pd.Timestamp(msg["day"]), from_records(self.pending), msg["attack"]))
                    self.pending = []
        return self.ready.popleft() if self.ready else None

    def _send(self, payload):
        self.p.produce(CTL, json.dumps({"run": self.run, **payload}).encode())
        self.p.poll(0)

    def ack(self, n):
        # The attack setting rides along with every ack, so a click made before the producer was listening is not lost.
        self._send({"cmd": "ack", "n": n, "attack": self.attack, "share": self.share})

    def set_attack(self, on, share):
        self.attack, self.share = on, share
        self._send({"cmd": "attack", "on": on, "share": share})
        self.p.flush(1)

    def status(self):
        try:
            _, hi = self.c.get_watermark_offsets(self.tp, timeout=0.5, cached=True)
            pos = self.c.position([self.tp])[0].offset
            lag = max(hi - pos, 0) if pos >= 0 else 0
        except Exception:
            lag = 0
        return {"mode": "kafka", "topic": TXN, "received": self.received, "buffered": len(self.ready), "lag": lag,
                "producer": self.proc.poll() is None, "attack": self.attack}

    def close(self):
        for fn in (lambda: self._send({"cmd": "stop"}), lambda: self.p.flush(1), self.proc.terminate, self.c.close):
            try:
                fn()
            except Exception:
                pass


def make_feed(seed=None):
    """Kafka if the broker answers within 2 s, otherwise the local feed with a notice."""
    if kafka_up():
        try:
            return KafkaFeed(seed), None
        except Exception as e:
            return LocalFeed(seed), f"Kafka not reachable ({type(e).__name__}), using local feed"
    return LocalFeed(seed), "Kafka not reachable, using local feed"


class LiveEngine(E.Engine):
    """The same monitor as Engine, fed by a never-ending live feed instead of the CSV stream."""

    def __init__(self, feed):
        super().__init__("real")
        keep = self.df["day"] < pd.Timestamp(M.STREAM_START)   # history before the stream (v1-scored) stays
        extra = ["day", "label_day", "p", "p_v1", "alert", "fn", "fp", "fn_v1", "fp_v1", "ver"]
        self.df = self.df.loc[keep, COLS + extra].copy()
        self.by_day, self.days = {}, []
        self.scenario, self.feed, self.attack_day = "kafka", feed, None

    @property
    def done(self):
        return False   # the stream never ends

    def _ingest(self, day, rows, attack):
        rows = rows.copy()
        rows["day"] = rows["transaction_datetime"].dt.floor("D")
        start = int(self.df.index.max()) + 1
        rows.index = pd.RangeIndex(start, start + len(rows))
        rows["label_day"] = (rows["day"] + pd.to_timedelta(rows["label_delay_days"], unit="D")).where(rows["label_available"] == 1)
        for c, v in (("p", np.nan), ("p_v1", np.nan), ("alert", False), ("fn", False), ("fp", False),
                     ("fn_v1", False), ("fp_v1", False), ("ver", 0)):
            rows[c] = v
        self.df = pd.concat([self.df, rows])
        self.by_day[day] = rows.index
        self.days.append(day)
        if attack and self.attack_day is None:
            self.attack_day = day

    def step(self, timeout=1.0):
        if self.i >= len(self.days):   # Engine.step needs a buffered day
            got = self.feed.poll_day(timeout)
            if got is None:
                return []
            self._ingest(*got)
        ev = super().step()
        self.feed.ack(self.i)
        return ev

    def run_to_next_event(self, max_days=60, max_sec=30):
        t = time.time()
        for _ in range(max_days):
            n = self.i
            ev = self.step()
            if ev:
                return ev
            if self.i == n or time.time() - t > max_sec:
                break
        return []
