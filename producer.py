"""Kafka producer for the live control room.

Publishes new transactions, one simulated day at a time, to topic txn-live (one JSON message per transaction, then an
end-of-day marker). Listens on txn-control for: ack (days the app has consumed), attack on/off, stop.
Started automatically by the app (livefeed.KafkaFeed); can also be run by hand:
    .venv\\Scripts\\python producer.py --run <id>
"""
import argparse
import json
import time

from confluent_kafka import Consumer, Producer, TopicPartition

import livefeed as LF

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=-1)
ap.add_argument("--bootstrap", default=LF.BOOTSTRAP)
args = ap.parse_args()

gen = LF.DayGen(args.seed if args.seed >= 0 else None)
prod = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 20, "log_level": 0})
ctl = Consumer({"bootstrap.servers": args.bootstrap, "group.id": "prod-" + args.run, "enable.auto.commit": False,
                "log_level": 0})
tp = TopicPartition(LF.CTL, 0)
_, hi = ctl.get_watermark_offsets(tp, timeout=5)
ctl.assign([TopicPartition(LF.CTL, 0, hi)])

produced, acked, last_ack = 0, 0, time.time()
print(f"producer {args.run} started, bootstrap {args.bootstrap}", flush=True)
while True:
    for m in ctl.consume(100, 0.2):
        if m.error():
            continue
        msg = json.loads(m.value())
        if msg.get("run") != args.run:
            continue
        if msg["cmd"] == "ack":
            acked, last_ack = msg["n"], time.time()
            if "attack" in msg and (msg["attack"], msg["share"]) != (gen.attack, gen.share):
                gen.attack, gen.share = msg["attack"], msg["share"]
                print(f"attack {'ON' if gen.attack else 'OFF'} share {gen.share} (from ack)", flush=True)
        elif msg["cmd"] == "attack":
            gen.attack, gen.share = msg["on"], msg["share"]
            print(f"attack {'ON' if gen.attack else 'OFF'} share {gen.share}", flush=True)
        elif msg["cmd"] == "stop":
            print("stop received", flush=True)
            raise SystemExit(0)
    if time.time() - last_ack > 300:
        print("no ack for 300 s, exiting", flush=True)
        raise SystemExit(0)
    if produced - acked < LF.AHEAD:
        day, d, attack = gen.next_day()
        for rec in LF.to_records(d):
            prod.produce(LF.TXN, json.dumps({"t": "txn", "run": args.run, "row": rec}).encode(),
                         key=rec["transaction_id"].encode())
            prod.poll(0)
        prod.produce(LF.TXN, json.dumps({"t": "eod", "run": args.run, "day": str(day.date()), "rows": len(d),
                                         "attack": attack}).encode())
        prod.flush(10)
        produced += 1
        print(f"day {day.date()} rows {len(d)} attack {attack}", flush=True)
