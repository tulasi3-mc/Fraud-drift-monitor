"""Step 2: baseline fraud model (LightGBM) + cost-based threshold.

Train on Jan-Feb, early-stop on Mar, pick threshold on Mar by cost (FN=50, FP=1),
report on the Apr-Jun stream. Seed fixed for reproducibility.
Run: .venv\\Scripts\\python train_drift.py
Writes: model_drift.joblib, metrics_drift.json
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
FN_COST = 50.0  # missing one thief
FP_COST = 1.0   # bothering one honest customer

d = joblib.load("data/clean.joblib")
df, feats, tgt = d["df"], d["features"], d["target"]
tr, va, st = df.loc[d["splits"]["train"]], df.loc[d["splits"]["val"]], df.loc[d["splits"]["stream"]]

params = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    seed=SEED, deterministic=True, verbosity=-1,
)
dtr = lgb.Dataset(tr[feats], tr[tgt], categorical_feature=d["cat_cols"] or "auto")
dva = lgb.Dataset(va[feats], va[tgt], reference=dtr)
model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])


def cost_at(y, p, thr):
    pred = p >= thr
    fn = int(((~pred) & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return FN_COST * fn + FP_COST * fp, fn, fp


def best_threshold(y, p):
    grid = np.linspace(0.01, 0.99, 99)
    costs = [cost_at(y, p, t)[0] for t in grid]
    return float(grid[int(np.argmin(costs))])


p_va = model.predict(va[feats], num_iteration=model.best_iteration)
thr = best_threshold(va[tgt].values, p_va)
p_st = model.predict(st[feats], num_iteration=model.best_iteration)

c_st, fn_st, fp_st = cost_at(st[tgt].values, p_st, thr)
metrics = {
    "kind": "drift_fraud",
    "seed": SEED,
    "best_iteration": int(model.best_iteration),
    "threshold": thr,
    "val": {"auc": roc_auc_score(va[tgt], p_va), "pr_auc": average_precision_score(va[tgt], p_va),
            "cost": cost_at(va[tgt].values, p_va, thr)[0]},
    "stream": {"auc": roc_auc_score(st[tgt], p_st), "pr_auc": average_precision_score(st[tgt], p_st),
               "cost": c_st, "fn": fn_st, "fp": fp_st, "rows": int(len(st))},
}
joblib.dump({"model": model, "features": feats, "cat_cols": d["cat_cols"], "threshold": thr}, "model_drift.joblib")
with open("metrics_drift.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
