"""Step 1: load the drift dataset, drop leakage, time-sort, split, write cleaning log.

Run: .venv\\Scripts\\python prep_drift.py
Writes: data\\clean.joblib (dict with X/y/time/phase per split), data\\cleaning_log.txt
"""
import joblib
import pandas as pd

SRC = "data/credit_card_fraud_drift_2026.csv"
TARGET = "is_fraud"
TIME = "transaction_datetime"

# Never model these: they reveal the drift, the outcome, or the ID of the row.
LEAKAGE = [
    "drift_phase", "drift_phase_name", "days_into_phase", "drift_magnitude",
    "concept_drift_score", "population_shift_index", "covariate_shift_flag",
    "synthetic_id_camouflage_score", "fraud_type", "fraud_amount_lost",
    "post_transaction_outcome", "transaction_time_target", "transaction_time_eligible",
    "dispute_adjudicated_friendly_fraud", "dispute_delay_days", "dispute_event_datetime",
    "adjudication_datetime", "dispute_regime", "pseudo_label", "label_available",
    "label_delay_days", "transaction_id", "customer_id", "card_id", "merchant_id",
    "month", "week_of_year",
    # recorded after the transaction (disputes/chargebacks), so not known at decision time
    "dispute_filed", "num_chargebacks_ever",
]
DROP_TEXT_ONLY = ["transaction_datetime"]  # used for splits, not as a feature

log = []
df = pd.read_csv(SRC, parse_dates=[TIME])
log.append(f"loaded: {df.shape[0]} rows x {df.shape[1]} cols")

n0 = len(df)
df = df.drop_duplicates()
log.append(f"duplicate rows removed: {n0 - len(df)}")

df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
log.append(f"time range: {df[TIME].min()} -> {df[TIME].max()}")

missing = df.isna().sum()
log.append(f"columns with NaN: {int((missing > 0).sum())}")

leak = [c for c in set(LEAKAGE) if c in df.columns]
features = [c for c in df.columns if c not in leak + [TARGET, TIME]]
cat_cols = [c for c in features if df[c].dtype == object or str(df[c].dtype) in ("str", "string")]
for c in cat_cols:
    df[c] = df[c].astype("category")
log.append(f"features used: {len(features)} (categorical: {len(cat_cols)})")
log.append(f"leakage columns dropped: {len(leak)}")

train = df[df[TIME] < "2026-03-01"]
val = df[(df[TIME] >= "2026-03-01") & (df[TIME] < "2026-04-01")]
stream = df[df[TIME] >= "2026-04-01"]
for name, part in [("train Jan-Feb", train), ("val Mar", val), ("stream Apr-Jun", stream)]:
    log.append(f"{name}: rows={len(part)} fraud_rate={part[TARGET].mean():.4f}")

out = {
    "features": features,
    "cat_cols": cat_cols,
    "target": TARGET,
    "time": TIME,
    "df": df,
    "splits": {"train": train.index, "val": val.index, "stream": stream.index},
}
joblib.dump(out, "data/clean.joblib")
with open("data/cleaning_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log) + "\n")
print("\n".join(log))
