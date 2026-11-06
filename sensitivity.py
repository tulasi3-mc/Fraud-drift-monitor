"""Sensitivity + placebo checks for the monitor.

1. Sensitivity: evasion share x start date x seed -> label-free detection day, delay, label confirmation, lead, costs.
2. Placebo: calm windows with no change -> how many label-free (daily-mean 3-sigma, 3 days in a row) alarms fire.
Run: .venv\\Scripts\\python sensitivity.py
Writes: sensitivity.json
"""
import json

import numpy as np
import pandas as pd

import monitor as M

SHARES = [0.1, 0.3, 0.5, 0.7, 0.9]
STARTS = ["2026-04-15", "2026-05-01", "2026-05-15"]
SEEDS = [42, 43, 44]

runs = []
for share in SHARES:
    for start in STARTS:
        for seed in SEEDS:
            s, _, rt = M.run_evasion(share, start, seed, retrain=True)
            runs.append({
                "share": share, "start": start, "seed": seed,
                "label_free_alarm": s["lead_measured_from"],
                "detection_delay_days": s["detection_delay_days"],
                "label_confirmation": s["first_label_confirmation"],
                "lead_days": s["lead_days_before_answers"],
                "missed_after_change": s["missed_after_change"],
                "frauds_after_change": s["frauds_after_change"],
                "cost_old": rt["old_model"]["cost"] if rt else None,
                "cost_retrained": rt["retrained"]["cost"] if rt else None,
            })
            print(f"share={share} start={start} seed={seed} -> alarm={s['lead_measured_from']} "
                  f"confirm={s['first_label_confirmation']} lead={s['lead_days_before_answers']}")

# Summary per (share, start): detection rate across seeds and median lead.
table = []
df = pd.DataFrame(runs)
for (share, start), g in df.groupby(["share", "start"]):
    caught = g["lead_days"].notna()
    table.append({
        "share": share, "start": start,
        "caught_label_free": f"{int(caught.sum())}/{len(g)}",
        "median_lead_days": float(g.loc[caught, "lead_days"].median()) if caught.any() else None,
        "median_delay_days": float(g.loc[caught, "detection_delay_days"].median()) if caught.any() else None,
        "median_cost_old": float(g["cost_old"].median()) if g["cost_old"].notna().any() else None,
        "median_cost_retrained": float(g["cost_retrained"].median()) if g["cost_retrained"].notna().any() else None,
    })


def placebo(ref_start, ref_end, test_start, test_end):
    """Same daily-mean 3-sigma, 3-days-in-a-row rule on a window where nothing was changed."""
    d = M.base[(M.base["transaction_datetime"] >= ref_start) & (M.base["transaction_datetime"] < test_end)]
    means = d.groupby("day")[M.num].mean()
    ref = means.loc[ref_start:pd.Timestamp(ref_end) - pd.Timedelta(days=1)]
    sd = ref.std().replace(0, np.nan)
    z = (means - ref.mean()) / sd
    persistent = ((z.abs() > 3).astype(float).rolling(3).sum() >= 3).loc[test_start:]
    fired = [c for c in persistent.columns if persistent[c].any()]
    return {"reference": f"{ref_start}..{ref_end}", "test": f"{test_start}..{test_end}",
            "days_tested": int(len(persistent)), "features": len(M.num), "features_alarmed": fired}


placebos = [
    placebo("2026-02-01", "2026-03-01", "2026-03-01", "2026-04-01"),
    placebo("2026-03-01", "2026-03-16", "2026-03-16", "2026-04-01"),
    placebo("2026-01-15", "2026-02-15", "2026-02-15", "2026-04-01"),
]

out = {"grid": {"shares": SHARES, "starts": STARTS, "seeds": SEEDS}, "table": table, "runs": runs, "placebo": placebos}
with open("sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)

print("\nshare start caught median_lead median_delay")
for t in table:
    print(t["share"], t["start"], t["caught_label_free"], t["median_lead_days"], t["median_delay_days"])
print("\nplacebo:")
for p in placebos:
    print(p["reference"], "->", p["test"], "alarms on:", p["features_alarmed"] or "none")
