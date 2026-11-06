"""Live fraud-monitoring engine for 'The Moving Target'.

Replays the Apr-Jun stream one day at a time, like a production system:
  - scores each day's transactions with the model that is deployed *that day*,
  - runs the label-free alarm (daily mean of each feature vs March, |z| > 3 for 3 days in a row),
  - lets labels arrive late (transaction day + label_delay_days; some never arrive),
  - confirms a loss when missed frauds revealed in the last 7 days exceed the March limit,
  - then retrains on the labels available that day and deploys model v2 from the next day.
Every read goes through `seen()` (rows up to today), so the engine never uses the future.
The batch results in monitor.json are the reference; check_engine.py asserts the replay reproduces them.
"""
import io

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import monitor as M

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              seed=M.SEED, deterministic=True, verbosity=-1)   # same as train_drift.py / monitor.py
MARCH = (pd.Timestamp(M.REF_START), pd.Timestamp(M.REF_END) - pd.Timedelta(days=1))


def score(model, X):
    # v1 has best_iteration from early stopping; a retrained model has 0 -> use all trees.
    return model.predict(X, num_iteration=model.best_iteration or None)


def explain(model, row_df, top=8):
    """Per-feature push toward fraud for one transaction (LightGBM pred_contrib, log-odds; last column = bias)."""
    c = model.predict(row_df[M.feats], pred_contrib=True, num_iteration=model.best_iteration or None)[0]
    out = pd.DataFrame({"feature": M.feats, "value": row_df[M.feats].iloc[0].astype(str).values,
                        "contribution": c[:-1]})
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top)
    return out.reset_index(drop=True), float(c[-1])


def model_bytes(model):
    buf = io.BytesIO()
    joblib.dump({"model": model, "features": M.feats, "cat_cols": sorted(M.cats), "threshold": M.thr}, buf)
    return buf.getvalue()


def train_live(on_iter=None):
    """Same training as train_drift.py (Jan-Feb train, March early stopping + cost threshold), with a live log."""
    df = M.base
    tr = df[df["transaction_datetime"] < M.REF_START]
    va = df[(df["transaction_datetime"] >= M.REF_START) & (df["transaction_datetime"] < M.REF_END)]
    cats = sorted(M.cats) or "auto"
    dtr = lgb.Dataset(tr[M.feats], tr["is_fraud"], categorical_feature=cats)
    dva = lgb.Dataset(va[M.feats], va["is_fraud"], reference=dtr)
    evals = {}
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.record_evaluation(evals)]
    if on_iter is not None:
        callbacks.append(lambda env: on_iter(env.iteration))
    model = lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dtr, dva],
                      valid_names=["train (Jan-Feb)", "validation (Mar)"], callbacks=callbacks)
    p_val = model.predict(va[M.feats], num_iteration=model.best_iteration)
    y_val = va["is_fraud"].values
    rows = []
    for t in np.linspace(0.01, 0.99, 99):   # same grid as train_drift.best_threshold
        a = p_val >= t
        fn, fp = int(((~a) & (y_val == 1)).sum()), int((a & (y_val == 0)).sum())
        rows.append({"threshold": round(float(t), 2), "cost": M.FN_COST * fn + M.FP_COST * fp, "fn": fn, "fp": fp})
    curve = pd.DataFrame(rows)
    thr = float(curve.loc[curve["cost"].idxmin(), "threshold"])
    a = p_val >= thr
    confusion = {"tn": int(((~a) & (y_val == 0)).sum()), "fp": int((a & (y_val == 0)).sum()),
                 "fn": int(((~a) & (y_val == 1)).sum()), "tp": int((a & (y_val == 1)).sum())}
    return {"model": model, "evals": evals, "best_iter": int(model.best_iteration), "threshold": thr,
            "p_val": p_val, "y_val": y_val, "cost_curve": curve, "confusion": confusion,
            "train_rows": int(len(tr)), "val_rows": int(len(va))}


class Engine:
    def __init__(self, scenario="real", share=M.EVADE_SHARE, start=M.EVADE_FROM, seed=M.SEED, auto_retrain=True):
        self.scenario, self.share, self.start, self.auto_retrain = scenario, share, start, auto_retrain
        df = M.make_evasion(M.base, share, start, seed)[0] if scenario == "evasion" else M.base.copy()
        df["label_day"] = (df["day"] + pd.to_timedelta(df["label_delay_days"], unit="D")).where(df["label_available"] == 1)
        for c, v in (("p", np.nan), ("p_v1", np.nan), ("alert", False), ("fn", False), ("fp", False),
                     ("fn_v1", False), ("fp_v1", False), ("ver", 0)):
            df[c] = v
        self.df = df
        self.by_day = df.groupby("day").groups
        self.models = {1: M.model}
        self.active = 1
        self.registry = [{"version": "v1", "trained on": "Jan–Feb labels", "rows": int((df["day"] < M.REF_START).sum()),
                          "deployed": "2026-04-01"}]

        # History before the stream: scored by v1 (needed for March reference and for late labels arriving in April).
        hist = df["day"] < pd.Timestamp(M.STREAM_START)
        self._score_rows(df.index[hist])

        march = df[(df["day"] >= MARCH[0]) & (df["day"] <= MARCH[1])]
        ref_means = march.groupby("day")[M.num].mean()
        self.ref_mu, self.ref_sd = ref_means.mean(), ref_means.std().replace(0, np.nan)
        self.march_means = ref_means   # for charts: the reference period the system learned 'normal' from
        self.streak = pd.Series(0, index=M.num)
        for d in sorted(ref_means.index):   # seed streaks so a run can cross from March into April
            self._update_streaks(ref_means.loc[d])

        fn_days = df.loc[df["fn"], "label_day"]
        counts = [int(((fn_days > d - pd.Timedelta(days=7)) & (fn_days <= d)).sum()) for d in sorted(ref_means.index)]
        self.lab_hi = max(float(np.mean(counts) + 3 * np.std(counts, ddof=1)), 1.0)

        self.days = sorted(d for d in self.by_day if d >= pd.Timestamp(M.STREAM_START))
        self.i = 0
        self.today = pd.Timestamp(M.STREAM_START) - pd.Timedelta(days=1)
        self.alarmed, self.confirmed, self.swap_day = {}, None, None
        self.events, self.daily = [], []

    # ---- internals ----
    def _score_rows(self, idx):
        X = self.df.loc[idx, M.feats]
        y = self.df.loc[idx, "is_fraud"].values == 1
        p = score(self.models[self.active], X)
        p1 = p if self.active == 1 else score(self.models[1], X)
        a, a1 = p >= M.thr, p1 >= M.thr
        self.df.loc[idx, "p"] = p
        self.df.loc[idx, "p_v1"] = p1
        self.df.loc[idx, "alert"] = a
        self.df.loc[idx, "fn"] = (~a) & y
        self.df.loc[idx, "fp"] = a & ~y
        self.df.loc[idx, "fn_v1"] = (~a1) & y
        self.df.loc[idx, "fp_v1"] = a1 & ~y
        self.df.loc[idx, "ver"] = self.active

    def _update_streaks(self, means):
        z = (means - self.ref_mu) / self.ref_sd
        hit = z.abs() > 3
        self.streak = (self.streak + 1).where(hit, 0)
        return z

    # ---- public ----
    @property
    def done(self):
        return self.i >= len(self.days)

    def seen(self):
        return self.df[self.df["day"] <= self.today]

    def step(self):
        """Advance one day. Returns the list of events raised that day."""
        if self.done:
            return []
        d = self.days[self.i]
        self.i += 1
        self.today = d
        idx = self.by_day[d]
        self._score_rows(idx)
        day_rows = self.df.loc[idx]
        events = []

        # Label-free layer: feature daily means vs March, 3 days in a row.
        z = self._update_streaks(day_rows[M.num].mean())
        for f in self.streak.index[self.streak >= 3]:
            if f not in self.alarmed:
                self.alarmed[f] = d
                events.append({"day": d, "type": "feature_alarm", "feature": f, "z": round(float(z[f]), 1),
                               "diag": M.diagnose(self.seen(), f, d)})

        # Label-based layer: labels arriving today; missed frauds revealed in the last 7 days.
        lab = self.df["label_day"]
        arrived_today = self.df[lab == d]
        recent = self.df[(lab > d - pd.Timedelta(days=7)) & (lab <= d)]
        fn_7d = int(recent["fn"].sum())
        if self.confirmed is None and fn_7d > self.lab_hi:
            self.confirmed = d
            events.append({"day": d, "type": "label_confirm", "fn_7d": fn_7d})
            if self.auto_retrain:
                events.append(self.retrain(d))

        za = z.abs()
        self.daily.append({
            "day": d, "rows": int(len(day_rows)), "alerts": int(day_rows["alert"].sum()),
            "fn": int(day_rows["fn"].sum()), "fp": int(day_rows["fp"].sum()),
            "fn_v1": int(day_rows["fn_v1"].sum()), "fp_v1": int(day_rows["fp_v1"].sum()),
            "fn_arrived": int(arrived_today["fn"].sum()), "fn_7d": fn_7d, "ver": int(self.df.loc[idx[0], "ver"]),
            "max_z": float(za.max()), "max_z_feature": za.idxmax(),
            **{f"mean_{c}": float(day_rows[c].mean()) for c in M.SHOW},
        })
        self.events.extend(events)
        return events

    def run_to_next_event(self, max_days=200):
        for _ in range(max_days):
            if self.done:
                return []
            ev = self.step()
            if ev:
                return ev
        return []

    def retrain(self, d):
        """Retrain on every transaction whose label has arrived by day d; deploy as the next version tomorrow."""
        train = self.df[self.df["label_day"] <= d]
        new = lgb.train(PARAMS, lgb.Dataset(train[M.feats], train["is_fraud"], categorical_feature=sorted(M.cats) or "auto"),
                        num_boost_round=int(M.model.best_iteration))
        # shortcut: one auto-retrain per run; add a cooldown rule if several changes are simulated in one run.
        self.auto_retrain = False
        v = max(self.models) + 1
        self.models[v] = new
        self.active = v
        self.swap_day = d
        self.registry.append({"version": f"v{v}", "trained on": f"labels arrived by {d.date()}", "rows": int(len(train)),
                              "deployed": str((d + pd.Timedelta(days=1)).date())})
        return {"day": d, "type": "model_swap", "version": f"v{v}", "rows": int(len(train))}

    def counters(self):
        s = self.seen()
        stream = s[s["day"] >= pd.Timestamp(M.STREAM_START)]
        known = stream[stream["label_day"] <= self.today]   # labels the system has actually received
        out = {
            "today": self.today, "transactions": int(len(stream)), "alerts": int(stream["alert"].sum()),
            "frauds_known": int(known["is_fraud"].sum()), "missed_known": int(known["fn"].sum()),
            "cost_known": float(M.FN_COST * known["fn"].sum() + M.FP_COST * known["fp"].sum()),
            "cost_true": float(M.FN_COST * stream["fn"].sum() + M.FP_COST * stream["fp"].sum()),
            "active": f"v{self.active}",
        }
        if self.swap_day is not None:
            after = stream[stream["day"] > self.swap_day]
            out["after_swap_new"] = float(M.FN_COST * after["fn"].sum() + M.FP_COST * after["fp"].sum())
            out["after_swap_old"] = float(M.FN_COST * after["fn_v1"].sum() + M.FP_COST * after["fp_v1"].sum())
        return out
