"""Consistency test: the live day-by-day engine must reproduce the batch results in monitor.json.

Run: .venv\\Scripts\\python check_engine.py   -> prints PASS or the failing check.
"""
import json
import time

import pandas as pd

import engine as E

ref = json.load(open("monitor.json", encoding="utf-8"))


def replay(scenario):
    t = time.time()
    eng = E.Engine(scenario)
    while not eng.done:
        eng.step()
    return eng, time.time() - t


def alarms(eng):
    return {f: str(d.date()) for f, d in eng.alarmed.items()}


fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got} want {want}")
    if not ok:
        fails.append(name)


# Real stream
eng, secs = replay("real")
rs = ref["real"]["summary"]
check("real: label limit", round(eng.lab_hi, 2), rs["label_fn_limit_7d"])
check("real: feature alarms", alarms(eng), rs["alarms_by_feature"])
check("real: labels confirm", eng.confirmed, None)
check("real: model swap", eng.swap_day, None)
check("real: stream cost", eng.counters()["cost_true"], rs["stream_cost"])
print(f"     real replay: {secs:.1f}s")

# Thieves adapt (70% from 2026-05-01)
eng, secs = replay("evasion")
es, rt = ref["evasion"]["summary"], ref["evasion"]["retrain"]
check("evasion: feature alarms", alarms(eng), es["alarms_by_feature"])
check("evasion: labels confirm", str(eng.confirmed.date()), es["first_label_confirmation"])
check("evasion: retrain rows", eng.registry[-1]["rows"], rt["train_rows"])
c = eng.counters()
check("evasion: cost after swap, v2", c.get("after_swap_new"), rt["retrained"]["cost"])
check("evasion: cost after swap, v1 shadow", c.get("after_swap_old"), rt["old_model"]["cost"])
print(f"     evasion replay: {secs:.1f}s")

# Live training reproduces the deployed model
tr = E.train_live()
import monitor as M
d = M.base[M.base["transaction_datetime"] >= M.STREAM_START]
diff = float(abs(tr["model"].predict(d[M.feats], num_iteration=tr["best_iter"]) - E.score(M.model, d[M.feats])).max())
check("train: best iteration", tr["best_iter"], int(M.model.best_iteration))
check("train: threshold", tr["threshold"], M.thr)
check("train: max |dp| vs deployed", round(diff, 9), 0.0)

# Explanation works on a single real row
row = M.base.loc[[d.index[0]]]
top, bias = E.explain(M.model, row)
check("explain: rows", len(top), 8)

print("\nPASS" if not fails else f"\nFAILED: {fails}")
