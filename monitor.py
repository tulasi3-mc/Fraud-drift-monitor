"""Daily fraud-stream monitor for 'The Moving Target'.

Two scenarios on the real Apr-Jun stream:
  real    - data as it is.
  evasion - simulation of "they read your alerts": from EVADE_FROM, a share of fraud rows copy the velocity
            profile of honest customers (they spread their spending so the speed signals look normal).
Two detector layers:
  label-free  (no answers needed): 7-day rolling PSI per feature vs March, and a 3-sigma control chart on the
              daily alert rate.  -> "know when the world changed, before the answers arrive"
  label-based (answers arrive late): missed frauds counted on the day their label ARRIVES
              (transaction time + label_delay_days from the dataset), 7-day rolling sum vs March 3-sigma.
Then: retrain once labels confirm the change, and compare cost of old vs retrained model afterwards.

Run: .venv\\Scripts\\python monitor.py
Writes: monitor.json
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42
FN_COST, FP_COST = 50.0, 1.0
REF_START, REF_END = "2026-03-01", "2026-04-01"   # stable reference (Jan-Feb has a warm-up artifact)
STREAM_START = "2026-04-01"
EVADE_FROM = "2026-05-01"
EVADE_SHARE = 0.7    # share of thieves who adapt
PSI_ALARM = 0.25
VELOCITY = ["amount_last_1h", "transactions_last_1h", "transactions_last_24h", "transactions_last_7d",
            "amount_last_24h", "merchant_velocity_24h"]
SHOW = ["device_change_7d", "failed_auth_attempts_24h", "amount_last_1h", "transactions_last_24h",
        "merchant_velocity_24h", "account_age_days"]   # daily means exported for the page

d = joblib.load("data/clean.joblib")
m = joblib.load("model_drift.joblib")
model, feats, thr = m["model"], m["features"], m["threshold"]
cats = set(d["cat_cols"])
num = [c for c in feats if c not in cats]
base = d["df"]
base = base.assign(day=base["transaction_datetime"].dt.floor("D"))


def make_evasion(df, share=EVADE_SHARE, start=EVADE_FROM, seed=SEED):
    """Simulated tactic change: from `start`, `share` of thieves copy the velocity profile of honest customers."""
    df = df.copy()
    rng_local = np.random.default_rng(seed)
    # Donors only from before the change, so the simulation never borrows future honest behavior.
    honest = df[(df["is_fraud"] == 0) & (df["transaction_datetime"] >= REF_START) & (df["transaction_datetime"] < start)]
    target = df.index[(df["is_fraud"] == 1) & (df["transaction_datetime"] >= start)]
    adapt = rng_local.choice(target, size=int(len(target) * share), replace=False)
    donors = honest.sample(n=len(adapt), random_state=seed, replace=True)
    df.loc[adapt, VELOCITY] = donors[VELOCITY].values
    return df, len(adapt)


def bin_edges(v):
    # Zero-inflated features (velocity counts) collapse plain deciles; keep the minimum value as its own bin.
    lo = v.min()
    rest = v[v > lo]
    if len(rest) == 0:
        return np.array([lo, lo])
    return np.unique(np.concatenate([[lo], np.quantile(rest, np.linspace(0, 1, 10))]))


def psi_fn(ref_df):
    edges = {c: bin_edges(ref_df[c].values) for c in num}
    ref_hist = {}
    for c, e in edges.items():
        if len(e) >= 3:
            ref_hist[c] = np.clip(np.histogram(ref_df[c].values, e)[0] / len(ref_df), 1e-4, None)

    def psi(window):
        out = {}
        for c, ex in ref_hist.items():
            e = edges[c]
            ey = np.histogram(np.clip(window[c].values, e[0], e[-1]), e)[0] / max(len(window), 1)
            ey = np.clip(ey, 1e-4, None)
            out[c] = float(np.sum((ey - ex) * np.log(ey / ex)))
        return out
    return psi


def first_true(dates, flags):
    for dt, f in zip(dates, flags):
        if f:
            return str(pd.Timestamp(dt).date())
    return None


def diagnose(df, feature, day):
    """Label-free: where did the shift happen? Split rows by the model's own flag (alerted vs not alerted)."""
    day = pd.Timestamp(day)
    ref = df[(df["transaction_datetime"] >= REF_START) & (df["transaction_datetime"] < REF_END)]
    win = df[(df["day"] > day - pd.Timedelta(days=7)) & (df["day"] <= day)]
    out = {}
    for name, flag in [("not_alerted", False), ("alerted", True)]:
        r, w = ref.loc[ref["alert"] == flag, feature], win.loc[win["alert"] == flag, feature]
        # Contribution = group share x mean change: how much of the overall shift this group explains.
        # (Percentages mislead on near-zero bases, e.g. 0.0014 -> 0.0002 is -83% but tiny in absolute terms.)
        contrib = (len(w) / max(len(win), 1)) * (w.mean() - r.mean())
        out[name] = {"march_mean": round(float(r.mean()), 4), "window_mean": round(float(w.mean()), 4),
                     "change_pct": round(100 * (w.mean() - r.mean()) / r.mean(), 1) if r.mean() else None,
                     "contribution": round(float(contrib), 4)}
    out["verdict"] = ("thieves" if abs(out["alerted"]["contribution"]) > abs(out["not_alerted"]["contribution"])
                      else "honest customers")
    return out


def monitor(df, scenario, change_from=STREAM_START, scored_features=None):
    """change_from / scored_features are used ONLY to score lead time (which alarms count as detecting this
    change, e.g. the features a simulation planted), never to detect."""
    df = df.copy()
    df["p"] = model.predict(df[feats], num_iteration=model.best_iteration)
    df["alert"] = df["p"] >= thr
    df["fn"] = (~df["alert"]) & (df["is_fraud"] == 1)
    df["fp"] = df["alert"] & (df["is_fraud"] == 0)
    # Labels that never arrive (label_available == 0) get no arrival day.
    df["label_day"] = (df["day"] + pd.to_timedelta(df["label_delay_days"], unit="D")).where(df["label_available"] == 1)

    ref = df[(df["transaction_datetime"] >= REF_START) & (df["transaction_datetime"] < REF_END)]
    psi = psi_fn(ref)

    # Label-free layer 1: alert-rate control chart (March mean +- 3 sigma of daily rate).
    daily = df.groupby("day").agg(rows=("alert", "size"), alerts=("alert", "sum"),
                                  fn=("fn", "sum"), fp=("fp", "sum"), frauds=("is_fraud", "sum"))
    daily["alert_rate"] = daily["alerts"] / daily["rows"]
    ref_rate = daily.loc[REF_START:"2026-03-31", "alert_rate"]
    lo, hi = ref_rate.mean() - 3 * ref_rate.std(), ref_rate.mean() + 3 * ref_rate.std()
    # 3 days in a row outside the band, to avoid single-day noise
    out_band = (daily["alert_rate"] < lo) | (daily["alert_rate"] > hi)
    daily["rate_alarm"] = out_band.rolling(3).sum() >= 3

    # Label-free layer 2: 7-day rolling PSI per feature, evaluated daily.
    days = daily.loc[STREAM_START:].index
    psi_max, psi_top = [], []
    for dday in days:
        w = df[(df["day"] > dday - pd.Timedelta(days=7)) & (df["day"] <= dday)]
        s = psi(w)
        top = max(s, key=s.get)
        psi_max.append(s[top])
        psi_top.append(top)
    daily.loc[days, "psi_max"] = psi_max
    daily.loc[days, "psi_top"] = psi_top
    daily["psi_alarm"] = daily["psi_max"] > PSI_ALARM

    # Label-free layer 3: control chart on each feature's daily mean (z vs March daily means), 3 days in a row.
    means = df.groupby("day")[num].mean()
    ref_means = means.loc[REF_START:"2026-03-31"]
    sd = ref_means.std().replace(0, np.nan)
    z = (means - ref_means.mean()) / sd
    persistent = (z.abs() > 3).astype(float).rolling(3).sum() >= 3
    daily["chart_alarm"] = persistent.any(axis=1)
    daily["chart_top"] = z.abs().where(persistent).fillna(-1).idxmax(axis=1).where(daily["chart_alarm"])
    daily["chart_z"] = z.abs().max(axis=1)
    for c in SHOW:
        daily[f"mean_{c}"] = means[c]
    bands = {c: [round(float(ref_means[c].mean()), 5), round(float(ref_means[c].std()), 5)] for c in SHOW}
    stream_persist = persistent.loc[STREAM_START:]
    alarms_by_feature = {c: str(stream_persist.index[stream_persist[c].values.argmax()].date())
                         for c in stream_persist.columns if stream_persist[c].any()}
    alarms_by_feature = dict(sorted(alarms_by_feature.items(), key=lambda kv: kv[1]))

    # Label-based layer: missed frauds counted on the day their label arrives.
    arrived = df[df["fn"]].groupby("label_day").size().reindex(daily.index, fill_value=0)
    daily["fn_arrived"] = arrived.values
    daily["fn_arrived_7d"] = daily["fn_arrived"].rolling(7, min_periods=1).sum()
    ref_arr = daily.loc[REF_START:"2026-03-31", "fn_arrived_7d"]
    lab_hi = max(ref_arr.mean() + 3 * ref_arr.std(), 1.0)
    daily["label_alarm"] = daily["fn_arrived_7d"] > lab_hi

    s = daily.loc[STREAM_START:]
    free_rate = first_true(s.index, s["rate_alarm"])
    free_psi = first_true(s.index, s["psi_alarm"])
    free_chart = first_true(s.index, s["chart_alarm"])
    label_conf = first_true(s.index, s["label_alarm"])
    free_first = min([x for x in (free_rate, free_psi, free_chart) if x], default=None)
    # Scoring only: the first label-free alarm on/after the change and on/before label confirmation detects it.
    change = str(pd.Timestamp(change_from).date())
    lead, lead_from, delay = None, None, None
    if label_conf:
        before = [dt for f, dt in alarms_by_feature.items()
                  if change <= dt <= label_conf and (scored_features is None or f in scored_features)]
        if free_rate and change <= free_rate <= label_conf:
            before.append(free_rate)
        if before:
            lead_from = min(before)
            lead = (pd.Timestamp(label_conf) - pd.Timestamp(lead_from)).days
            delay = (pd.Timestamp(lead_from) - pd.Timestamp(change)).days

    top_at_alarm = None
    if free_psi:
        top_at_alarm = s.loc[free_psi, "psi_top"]

    return df, daily, {
        "scenario": scenario,
        "alert_band": [round(lo, 4), round(hi, 4)],
        "label_fn_limit_7d": round(lab_hi, 2),
        "first_alarm_alert_rate": free_rate,
        "first_alarm_psi": free_psi,
        "psi_top_feature_at_alarm": top_at_alarm,
        "first_alarm_mean_chart": free_chart,
        "chart_feature_at_alarm": s.loc[free_chart, "chart_top"] if free_chart else None,
        "first_label_free_alarm": free_first,
        "alarms_by_feature": alarms_by_feature,
        "bands": bands,
        "diagnosis": {c: diagnose(df, c, day) for c, day in list(alarms_by_feature.items())[:4]},
        "first_label_confirmation": label_conf,
        "lead_days_before_answers": lead,
        "lead_measured_from": lead_from,
        "detection_delay_days": delay,
        "change_from": change,
        "stream_cost": float(FN_COST * s["fn"].sum() + FP_COST * s["fp"].sum()),
        "stream_fn": int(s["fn"].sum()),
        "stream_fp": int(s["fp"].sum()),
    }


def retrain_and_compare(df, retrain_day):
    """Retrain on every row whose label has arrived by retrain_day; compare costs after that day."""
    retrain_day = pd.Timestamp(retrain_day)
    train = df[df["label_day"] <= retrain_day]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  seed=SEED, deterministic=True, verbosity=-1)
    new = lgb.train(params, lgb.Dataset(train[feats], train["is_fraud"], categorical_feature=list(cats) or "auto"),
                    num_boost_round=int(model.best_iteration))
    after = df[df["day"] > retrain_day]
    out = {"retrain_day": str(retrain_day.date()), "train_rows": int(len(train)), "rows_after": int(len(after))}
    for name, p in [("old_model", after["p"].values),
                    ("retrained", new.predict(after[feats]))]:
        a = p >= thr
        y = after["is_fraud"].values
        fn, fp = int(((~a) & (y == 1)).sum()), int((a & (y == 0)).sum())
        out[name] = {"fn": fn, "fp": fp, "cost": FN_COST * fn + FP_COST * fp}
    return out


def series(daily):
    s = daily.loc[REF_START:].reset_index()  # include March so charts show the baseline
    s["day"] = s["day"].dt.strftime("%Y-%m-%d")
    cols = ["day", "rows", "alerts", "alert_rate", "frauds", "fn", "fp", "psi_max", "psi_top",
            "rate_alarm", "psi_alarm", "chart_alarm", "chart_top", "chart_z",
            "fn_arrived", "fn_arrived_7d", "label_alarm"] + [f"mean_{c}" for c in SHOW]
    s = s[cols]
    return json.loads(s.to_json(orient="records"))


def run_evasion(share=EVADE_SHARE, start=EVADE_FROM, seed=SEED, retrain=True):
    """One evasion scenario end to end (used by main, sensitivity.py and the app's what-if)."""
    evaded, n_adapt = make_evasion(base, share, start, seed)
    df_ev, daily_ev, summ = monitor(evaded, "evasion", change_from=start, scored_features=VELOCITY)
    after = df_ev[df_ev["day"] >= pd.Timestamp(start)]
    summ.update({"adapted_frauds": n_adapt, "share": share, "seed": seed,
                 "missed_after_change": int(after["fn"].sum()),
                 "frauds_after_change": int(after["is_fraud"].sum())})
    rt = retrain_and_compare(df_ev, summ["first_label_confirmation"]) if retrain and summ["first_label_confirmation"] else None
    return summ, daily_ev, rt


if __name__ == "__main__":
    results = {"settings": {"fn_cost": FN_COST, "fp_cost": FP_COST, "threshold": thr,
                            "reference": f"{REF_START}..{REF_END}", "evade_from": EVADE_FROM,
                            "evade_share": EVADE_SHARE, "psi_alarm": PSI_ALARM, "velocity_features": VELOCITY}}

    _, daily_real, sum_real = monitor(base, "real")
    results["real"] = {"summary": sum_real, "daily": series(daily_real)}

    sum_ev, daily_ev, rt = run_evasion()
    results["evasion"] = {"summary": sum_ev, "daily": series(daily_ev)}
    if rt:
        results["evasion"]["retrain"] = rt

    with open("monitor.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    for k in ("real", "evasion"):
        print(json.dumps(results[k]["summary"], indent=2))
    if rt:
        print(json.dumps(rt, indent=2))
