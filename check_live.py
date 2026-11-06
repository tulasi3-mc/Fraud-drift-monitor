"""Checks for the live feed (local fallback always; Kafka if a broker answers on localhost:9092).

Run: .venv\\Scripts\\python check_live.py   -> prints PASS or the failing check.
"""
import time

import pandas as pd

import livefeed as LF

M = LF.M
fails = []


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {info}")
    if not ok:
        fails.append(name)


# Local feed: quiet before the attack, alarm (thieves) soon after it, then labels confirm and v2 is deployed.
t = time.time()
eng = LF.LiveEngine(LF.LocalFeed(1))
quiet = [e for _ in range(20) for e in eng.step()]
check("local: no events in 20 normal days", not quiet, [e["type"] for e in quiet])
eng.feed.set_attack(True, 0.7)
events = [e for _ in range(45) for e in eng.step()]
alarms = [e for e in events if e["type"] == "feature_alarm"]
first = alarms[0] if alarms else None
check("local: label-free alarm after attack", first is not None)
if first:
    delay = (first["day"] - eng.attack_day).days
    check("local: alarm within 6 days of the attack", delay <= 6, f"(delay {delay} d, {first['feature']})")
    check("local: verdict thieves", first["diag"]["verdict"] == "thieves", first["diag"]["verdict"])
types = [e["type"] for e in events]
check("local: labels confirm", "label_confirm" in types)
check("local: model v2 deployed", "model_swap" in types)
if "label_confirm" in types and first:
    conf = next(e for e in events if e["type"] == "label_confirm")
    check("local: alarm before the answers", first["day"] < conf["day"], f"({first['day'].date()} < {conf['day'].date()})")
check("local: dtypes match training", (eng.df[M.feats].dtypes == M.base[M.feats].dtypes).all())
print(f"     local run: {time.time() - t:.1f}s, {eng.i} days")

# Kafka (only if a broker is up)
if LF.kafka_up():
    t = time.time()
    feed = LF.KafkaFeed(2)
    eng = LF.LiveEngine(feed)
    for _ in range(3):
        for _ in range(20):        # first day waits for the producer to start (model load ~3-4 s)
            n = eng.i
            eng.step(timeout=1.0)
            if eng.i > n:
                break
    st = feed.status()
    check("kafka: 3 days consumed", eng.i == 3, f"(i={eng.i})")
    check("kafka: messages received", st["received"] > 1900, f"({st['received']})")
    check("kafka: producer running", st["producer"])
    feed.close()
    print(f"     kafka run: {time.time() - t:.1f}s")
else:
    print("skip kafka: broker not reachable on", LF.BOOTSTRAP)

print("\nPASS" if not fails else f"\nFAILED: {fails}")
