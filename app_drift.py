"""Streamlit site for problem #3 'The Moving Target'.
Run: .venv\\Scripts\\streamlit run app_drift.py
Reads only monitor.json and metrics_drift.json (made by monitor.py and train_drift.py).
"""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="The Moving Target", page_icon="◆", layout="wide")

INK, PAPER, CARD, LINE, MUTED = "#1D2125", "#F6F2E9", "#FFFDF8", "#DDD5C4", "#5E6168"
ACCENT, SLATE, OCHRE, SAGE = "#B4411F", "#2F4B5C", "#C99A2E", "#6B7F5A"

st.markdown(f"""<style>
:root {{--ink:{INK}; --paper:{PAPER}; --card:{CARD}; --line:{LINE}; --muted:{MUTED}; --accent:{ACCENT};
  --serif: Georgia, "Iowan Old Style", "Palatino Linotype", Cambria, serif;
  --sans: "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, sans-serif;}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{display: none !important; visibility: hidden;}}
html, body, [data-testid="stAppViewContainer"], .stApp {{background: var(--paper); color: var(--ink); font-family: var(--sans);}}
[data-testid="stHeader"] {{background: transparent;}}
.block-container, [data-testid="stMainBlockContainer"] {{padding: 2.2rem 2.5rem 3rem; max-width: min(1600px, 96vw);}}
h1, h2, h3 {{font-family: var(--serif) !important; color: var(--ink); letter-spacing: -.01em;}}
.topbar {{display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; flex-wrap: wrap;
  padding: .2rem 0 .9rem; margin-bottom: 1rem; border-bottom: 2px solid var(--ink); animation: rise .5s ease-out both;}}
.topbar .brand {{display: flex; flex-direction: column; gap: .15rem;}}
.topbar .kicker {{font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; color: var(--accent); font-weight: 700;}}
.topbar .name {{font-family: var(--serif); font-size: 1.75rem; font-weight: 700; line-height: 1.1;}}
.topbar .tag {{color: var(--muted); font-size: .98rem; font-weight: 400; margin: 0; padding: 0;}}
.topbar .chips {{display: flex; flex-wrap: wrap; gap: .5rem;}}
.topbar .chip {{background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 4px; padding: .35rem .7rem; font-size: .85rem; font-weight: 600; color: var(--ink);}}
.hero {{padding: 0 0 1.4rem; margin-bottom: 1.4rem; border-bottom: 2px solid var(--ink); animation: rise .6s ease-out both;}}
.hero .kicker {{font-size: .75rem; letter-spacing: .18em; text-transform: uppercase; color: var(--accent); font-weight: 700;}}
.hero h1 {{margin: .3rem 0 0; padding: 0; font-size: 2.6rem; line-height: 1.08; font-weight: 700;}}
.hero p {{margin: .6rem 0 0; color: var(--muted); font-size: 1.08rem; max-width: 48rem;}}
.stats {{display: flex; flex-wrap: wrap; gap: 2.2rem; margin-top: 1.2rem;}}
.stat .k {{font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);}}
.stat .v {{font-family: var(--serif); font-size: 1.45rem; font-weight: 700; color: var(--ink);}}
[data-testid="stMetric"] {{background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--accent);
  border-radius: 6px; padding: .9rem 1.1rem;}}
[data-testid="stMetricValue"] {{font-family: var(--serif); color: var(--ink);}}
[data-testid="stMetricLabel"] p {{color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-size: .72rem;}}
.result {{padding: 1.2rem 1.6rem; margin: .8rem 0 1.2rem; background: var(--card); border: 1px solid var(--line);
  border-left: 6px solid var(--accent); border-radius: 6px; animation: reveal .5s cubic-bezier(.2,.8,.2,1) both;}}
.result .label {{color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .14em;}}
.result .value {{font-family: var(--serif); font-size: 1.7rem; font-weight: 700; color: var(--ink); line-height: 1.2;}}
.result .why {{margin-top: .4rem; color: var(--ink);}}
.feed {{background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: .9rem 1.2rem;}}
.feed .row {{padding: .25rem 0; border-bottom: 1px dashed var(--line);}}
.feed .row:last-child {{border-bottom: 0;}}
.tag {{display: inline-block; padding: .05rem .5rem; border-radius: 3px; font-size: .75rem; font-weight: 700; letter-spacing: .06em;}}
.tag.ok {{background: #E4EADF; color: {SAGE};}} .tag.alarm {{background: #F4DDD3; color: {ACCENT};}}
.tag.wait {{background: #F2E8CF; color: #8A6A12;}}
@keyframes rise {{from {{opacity: 0; transform: translateY(10px);}} to {{opacity: 1; transform: none;}}}}
@keyframes reveal {{from {{opacity: 0; transform: translateX(-14px);}} to {{opacity: 1; transform: none;}}}}
.stTabs [data-baseweb="tab-list"] {{gap: 1.6rem; border-bottom: 1px solid var(--line);}}
.stTabs [data-baseweb="tab"] {{background: transparent; padding: .5rem 0; color: var(--muted); font-weight: 600;}}
.stTabs [data-baseweb="tab"]:hover, .stTabs [aria-selected="true"] {{color: var(--ink);}}
.stTabs [data-baseweb="tab-highlight"] {{background: var(--accent); height: 3px;}}
[data-testid="stPlotlyChart"] {{background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 0;
  box-sizing: border-box; overflow: hidden;}}  /* no padding: Plotly sizes to the box, padding pushed it past the edge */
@media (prefers-reduced-motion: reduce) {{*, *::before, *::after {{animation: none !important; transition: none !important;}}}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load(stamp):  # stamp = file mtimes, so a re-run of monitor.py shows up on refresh
    with open("monitor.json", encoding="utf-8") as f:
        mon = json.load(f)
    with open("metrics_drift.json", encoding="utf-8") as f:
        met = json.load(f)
    sens = None
    if os.path.exists("sensitivity.json"):
        with open("sensitivity.json", encoding="utf-8") as f:
            sens = json.load(f)
    return mon, met, sens


@st.cache_resource
def monitor_module():
    import monitor  # loads data + model once (~2 s)
    return monitor


@st.cache_data(show_spinner=False)
def what_if(share, start):
    M = monitor_module()
    summ, daily, rt = M.run_evasion(share, start, M.SEED, retrain=True)
    d = pd.DataFrame(M.series(daily))
    d["day"] = pd.to_datetime(d["day"])
    return summ, d, rt


mon, met, sens = load(tuple(os.path.getmtime(p) for p in ("monitor.json", "metrics_drift.json", "sensitivity.json")
                            if os.path.exists(p)))
real, ev = mon["real"], mon["evasion"]
rs, es = real["summary"], ev["summary"]
rd, ed = pd.DataFrame(real["daily"]), pd.DataFrame(ev["daily"])
for frame in (rd, ed):
    frame["day"] = pd.to_datetime(frame["day"])
LABELED_ONSET = pd.Timestamp("2026-04-01")
EVADE_FROM = pd.Timestamp(mon["settings"]["evade_from"])


def style(fig, title, height=330):
    fig.update_layout(title=dict(text=title, x=0.012, xanchor="left"), height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=54, l=22, r=20, b=14), legend=dict(orientation="h", y=-0.18),
                      font=dict(family="Segoe UI, system-ui, sans-serif", color=INK, size=13),
                      title_font=dict(family="Georgia, serif", size=17, color=INK))
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#E9E3D4", linecolor=LINE)
    # No wheel-zoom / toolbar: otherwise the chart swallows the mouse wheel and the page cannot scroll.
    fig.update_layout(dragmode=False)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": False, "displayModeBar": False})


def vline(fig, day, text, color, i=0):
    fig.add_vline(x=day, line_width=1.5, line_dash="dot", line_color=color)
    fig.add_annotation(x=day, y=1.04 - 0.08 * i, yref="paper", text=text, showarrow=False, font=dict(size=11, color=color),
                       xanchor="left")


def replay_cut(daily, marks, upto):
    """Replay: keep only what had happened by `upto` (data and marks), plus a 'selected day' line."""
    if upto is None:
        return daily, marks
    upto = pd.Timestamp(upto)
    marks = [(d, t, c) for d, t, c in marks if d is not None and pd.Timestamp(d) <= upto] + [(upto, "selected day", INK)]
    return daily[daily["day"] <= upto], marks


def feature_chart(daily, summary, feature, title, marks, upto=None):
    mu, sd = summary["bands"][feature]
    daily, marks = replay_cut(daily, marks, upto)
    fig = go.Figure()
    fig.add_hrect(y0=mu - 3 * sd, y1=mu + 3 * sd, fillcolor="#E4EADF", opacity=0.6, line_width=0,
                  annotation_text="March normal range (±3σ)", annotation_position="top left",
                  annotation_font=dict(size=11, color=SAGE))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily[f"mean_{feature}"], mode="lines", name=f"daily mean of {feature}",
                             line=dict(color=SLATE, width=2)))
    for i, (day, text, color) in enumerate(marks):
        if day is not None:
            vline(fig, pd.Timestamp(day), text, color, i)
    if upto is not None:
        fig.update_xaxes(range=["2026-03-01", "2026-06-30"])   # fixed axis so the line grows as you drag
    style(fig, title)


def label_chart(daily, summary, title, marks, upto=None):
    daily, marks = replay_cut(daily, marks, upto)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fn_arrived_7d"], name="missed frauds revealed by late labels (7-day sum)",
                         marker_color=OCHRE))
    fig.add_hline(y=summary["label_fn_limit_7d"], line_dash="dash", line_color=ACCENT,
                  annotation_text="March limit", annotation_position="top left")
    for i, (day, text, color) in enumerate(marks):
        if day is not None:
            vline(fig, pd.Timestamp(day), text, color, i)
    if upto is not None:
        fig.update_xaxes(range=["2026-03-01", "2026-06-30"])
    style(fig, title, height=300)


def tag(kind, text):
    return f'<span class="tag {kind}">{text}</span>'


def feed(daily, upto, summary):
    """What the monitor knew on a given day: label-free alarms vs label-confirmed losses."""
    seen = daily[daily["day"] <= upto]
    rows = []
    free = seen[seen["chart_alarm"] & (seen["day"] >= LABELED_ONSET)]
    if len(free):
        first = free.iloc[0]
        rows.append(f'{tag("alarm", "WORLD CHANGED")} first label-free alarm on <b>{first.day.date()}</b> '
                    f'(feature: <code>{first.chart_top}</code>)')
        new = {f: d for f, d in summary["alarms_by_feature"].items() if pd.Timestamp(d) <= upto}
        rows.append("Features outside their normal range so far: " +
                    ", ".join(f"<code>{f}</code> since {d}" for f, d in new.items()))
    else:
        rows.append(f'{tag("ok", "NORMAL")} no feature outside its March range yet')
    lab = seen[seen["label_alarm"] & (seen["day"] >= LABELED_ONSET)]
    if len(lab):
        rows.append(f'{tag("alarm", "ANSWERS CONFIRM")} late labels show extra missed fraud from '
                    f'<b>{lab.iloc[0].day.date()}</b>')
    else:
        rows.append(f'{tag("wait", "WAITING FOR ANSWERS")} late labels show no rise in missed fraud yet '
                    f'({int(seen["fn_arrived"].sum())} missed frauds revealed so far)')
    st.markdown('<div class="feed">' + "".join(f'<div class="row">{r}</div>' for r in rows) + "</div>",
                unsafe_allow_html=True)


def diagnosis_table(summary, features):
    rows = []
    for f in features:
        dg = summary["diagnosis"].get(f)
        if not dg:
            continue
        na, al = dg["not_alerted"], dg["alerted"]
        rows.append({"feature": f,
                     "not alerted (mostly honest): March → alarm week": f'{na["march_mean"]} → {na["window_mean"]}',
                     "alerted (mostly thieves): March → alarm week": f'{al["march_mean"]} → {al["window_mean"]}',
                     "shift explained by honest / thieves": f'{na["contribution"]:+.4f} / {al["contribution"]:+.4f}',
                     "verdict": dg["verdict"]})
    st.table(pd.DataFrame(rows).set_index("feature"))  # static table: does not capture the mouse wheel
    st.caption("Shift explained = group's share of traffic × change in its mean. The larger one decides the verdict.")


# ---------------- Top bar ----------------
st.markdown("""<div class="topbar">
<div class="brand"><span class="kicker">Problem 3 · The Moving Target</span><span class="name">Fraud drift monitor</span><span class="tag">Spots when thieves change tactics, before the fraud labels arrive.</span></div>
<div class="chips"><span class="chip">LightGBM fraud model</span><span class="chip">Kafka live stream</span>
<span class="chip">Self-retraining</span><span class="chip">120,000 transactions</span></div>
</div>""", unsafe_allow_html=True)


# ---------------- Live control room (engine.py) ----------------
import engine as E  # noqa: E402  (loads data + model once per process)
import livefeed as LF  # noqa: E402  (Kafka live feed + local fallback)

SCENARIOS = {"Thieves adapt (simulation on the real stream)": "evasion", "Real stream as it is": "real"}
ss = st.session_state


def new_engine():
    if ss.get("feed") is not None:
        ss.feed.close()   # stops the old producer
    ss.feed, ss.kafka_notice, ss.attack_on = None, None, False
    if ss.get("lc_kafka"):
        ss.feed, ss.kafka_notice = LF.make_feed()
        ss.eng = LF.LiveEngine(ss.feed)
    else:
        ss.eng = E.Engine(SCENARIOS[ss.get("lc_scenario", next(iter(SCENARIOS)))],
                          share=ss.get("lc_share", 0.7), start=ss.get("lc_start", "2026-05-01"))
    ss.running = False


def set_attack(on):
    # Idempotent (not a toggle): a click registered twice during a live refresh cannot flip it back.
    ss.attack_on = on
    ss.feed.set_attack(on, ss.get("lc_share", 0.7))


if "eng" not in ss:
    new_engine()
ss.setdefault("running", False)
ss.setdefault("speed", 2.0)   # slow enough to read the incoming data and the faults

EVENT_TEXT = {
    "feature_alarm": lambda e: (tag("alarm", "WORLD CHANGED") +
                                f' <code>{e["feature"]}</code> left its March range (z = {e["z"]}) · verdict: '
                                f'<b>{e["diag"]["verdict"]}</b> changed'),
    "label_confirm": lambda e: (tag("alarm", "ANSWERS CONFIRM") +
                                f' {e["fn_7d"]} missed frauds revealed by late labels in 7 days (limit {ss.eng.lab_hi:.1f})'),
    "model_swap": lambda e: (tag("ok", f'MODEL {e["version"].upper()} DEPLOYED') +
                             f' retrained on {e["rows"]:,} labelled transactions; serving from tomorrow'),
}
START_OPTIONS = sorted({str(d.date()) for d in pd.date_range("2026-04-15", "2026-06-03", freq="7D")} | {"2026-05-01"})


def live_controls():
    kafka = bool(ss.get("lc_kafka"))
    c0, c1, c2, c3 = st.columns([1.1, 2.2, 1.2, 1.2])
    c0.toggle("Kafka Live Data", key="lc_kafka", on_change=new_engine,
              help="New transactions arrive through a local Kafka broker (topic txn-live) from producer.py")
    c1.radio("Scenario", list(SCENARIOS), key="lc_scenario", on_change=new_engine, horizontal=True, disabled=kafka)
    evasion = SCENARIOS[ss.lc_scenario] == "evasion"
    c2.slider("Share of thieves who adapt", 0.1, 0.9, 0.7, 0.1, key="lc_share",
              on_change=None if kafka else new_engine, disabled=not (evasion or kafka))
    c3.select_slider("Tactic change date", options=START_OPTIONS, value="2026-05-01", key="lc_start",
                     on_change=new_engine, disabled=kafka or not evasion)
    if kafka:
        b0, b1, b2, b3, b4, b5 = st.columns([1.3, 1, 1, 1.3, 1, 1.3])
        if ss.get("attack_on"):
            b0.button("Stop attack", key="lc_attack_stop", on_click=set_attack, args=(False,))
        else:
            b0.button("Launch attack", key="lc_attack_go", on_click=set_attack, args=(True,),
                      help="From now on, a share of thieves copy honest customers' spending speed")
    else:
        b1, b2, b3, b4, b5 = st.columns([1, 1, 1.3, 1, 1.3])
    if b1.button("Pause" if ss.running else "Start", key="lc_run", type="primary", disabled=ss.eng.done):
        ss.running = not ss.running
        st.rerun()
    if b2.button("Step 1 day", key="lc_step", disabled=ss.eng.done):
        ss.running = False
        ss.eng.step()
    if b3.button("Run to next event", key="lc_next", disabled=ss.eng.done):
        ss.running = False
        ss.eng.run_to_next_event()
    if b4.button("Reset", key="lc_reset"):
        new_engine()
    b5.select_slider("Speed (seconds per day)", options=[0.5, 1.0, 2.0, 3.0, 5.0], key="speed")


def live_feature_chart(eng, feature):
    mu, sd = float(eng.ref_mu[feature]), float(eng.ref_sd[feature])
    march = eng.march_means[feature]
    live = pd.DataFrame(eng.daily)
    fig = go.Figure()
    fig.add_hrect(y0=mu - 3 * sd, y1=mu + 3 * sd, fillcolor="#E4EADF", opacity=0.6, line_width=0,
                  annotation_text="normal range learned in March (±3σ)", annotation_position="top left",
                  annotation_font=dict(size=11, color=SAGE))
    fig.add_trace(go.Scatter(x=march.index, y=march.values, mode="lines", name="March (reference)",
                             line=dict(color=MUTED, width=1.5)))
    if len(live):
        fig.add_trace(go.Scatter(x=live["day"], y=live[f"mean_{feature}"], mode="lines", name="live stream",
                                 line=dict(color=SLATE, width=2.2)))
    marks = [(d, "alarm", ACCENT) for f, d in eng.alarmed.items() if f == feature]
    if eng.scenario == "evasion":
        marks.insert(0, (pd.Timestamp(eng.start), "tactic change (hidden from system)", MUTED))
    if eng.scenario == "kafka" and eng.attack_day is not None:
        marks.insert(0, (eng.attack_day, "attack launched (hidden from system)", MUTED))
    for i, (day, text, color) in enumerate(marks):
        vline(fig, day, text, color, i)
    fig.update_xaxes(range=["2026-03-01", x_end(eng)])
    style(fig, f"Daily mean of {feature}", height=300)


def live_label_chart(eng):
    live = pd.DataFrame(eng.daily)
    fig = go.Figure()
    if len(live):
        fig.add_trace(go.Bar(x=live["day"], y=live["fn_7d"], name="missed frauds revealed by late labels (7 days)",
                             marker_color=OCHRE))
    else:
        fig.add_annotation(text="No labels yet: press Start or Step 1 day", xref="paper", yref="paper", x=0.5, y=0.45,
                           showarrow=False, font=dict(size=13, color=MUTED))
    fig.add_hline(y=eng.lab_hi, line_dash="dash", line_color=ACCENT, annotation_text="March limit",
                  annotation_position="top left")
    marks = []
    if eng.confirmed is not None:
        marks.append((eng.confirmed, "answers confirm", OCHRE))
    if eng.swap_day is not None:
        marks.append((eng.swap_day + pd.Timedelta(days=1), "v2 deployed", SAGE))
    for i, (day, text, color) in enumerate(marks):
        vline(fig, day, text, color, i)
    fig.update_xaxes(range=["2026-04-01", x_end(eng)])
    style(fig, "What the late labels reveal", height=260)


def x_end(eng):
    # The CSV stream ends 2026-06-30; the Kafka stream never ends, so the axis grows with it.
    return max(pd.Timestamp("2026-06-30"), eng.today + pd.Timedelta(days=5)).strftime("%Y-%m-%d")


def live_render(eng, feature):
    c = eng.counters()
    day_txt = "not started" if eng.i == 0 else str(c["today"].date())
    status = "running" if ss.running else ("finished" if eng.done else "paused")
    if eng.scenario == "kafka":
        stt = eng.feed.status()
        if ss.get("kafka_notice"):
            st.warning(ss.kafka_notice)
        if stt["mode"] == "kafka":
            src = (f"Kafka · connected · topic {stt['topic']} · {stt['received']:,} messages · lag {stt['lag']} · "
                   f"producer {'running' if stt['producer'] else 'stopped'}")
            if stt["received"] == 0:
                src += " · waiting for the producer to start (a few seconds)"
        else:
            src = f"Local feed · {stt['received']:,} transactions generated"
        st.caption(f"{src} · attack {'ON' if stt['attack'] else 'OFF'}"
                   + (f" (launched in the stream on {eng.attack_day.date()})" if eng.attack_day is not None else ""))
        st.markdown(f"**Day {day_txt}** · {eng.i} days streamed · model **{c['active']}** serving · {status}")
    else:
        st.markdown(f"**Day {day_txt}** · {eng.i} of {len(eng.days)} days · model **{c['active']}** serving · {status}")
        st.progress(eng.i / len(eng.days))
    m = st.columns(6)
    m[0].metric("Transactions scored", f"{c['transactions']:,}")
    m[1].metric("Alerts raised", f"{c['alerts']:,}")
    m[2].metric("Frauds confirmed by labels", f"{c['frauds_known']:,}")
    m[3].metric("Missed (labels so far)", f"{c['missed_known']:,}")
    m[4].metric("Cost known so far", f"{c['cost_known']:,.0f}", help="50 × missed + 1 × false alarm, among labels received")
    m[5].metric("Model serving", c["active"])
    left, right = st.columns([3, 2])
    with left:
        live_feature_chart(eng, feature)
        live_label_chart(eng)
    with right:
        st.markdown("**Event feed**")
        if eng.events:
            rows = "".join(f'<div class="row"><b>{e["day"].date()}</b> · {EVENT_TEXT[e["type"]](e)}</div>'
                           for e in reversed(eng.events))
        else:
            rows = f'<div class="row">{tag("ok", "NORMAL")} no alarms yet</div>'
        st.markdown(f'<div class="feed">{rows}</div>', unsafe_allow_html=True)
        st.markdown("**Model registry**")
        st.table(pd.DataFrame(eng.registry).set_index("version"))
        if "after_swap_new" in c:
            st.markdown(f"**After the swap** (true cost, hidden from the system, shown for judges): "
                        f"v2 **{c['after_swap_new']:,.0f}** vs old v1 would have been **{c['after_swap_old']:,.0f}**")
        st.caption(f"True cost so far (hidden from the system): {c['cost_true']:,.0f}")
    live_data_sections(eng)




REASON_TEXT = {
    "prior_dispute_ratio_90d": lambda v: f"past disputes on {float(v):.1%} of transactions",
    "amount_last_1h": lambda v: f"spent {float(v):,.1f} in the last hour",
    "amount_last_24h": lambda v: f"spent {float(v):,.1f} in the last 24 h",
    "transactions_last_1h": lambda v: f"{int(v)} transactions in the last hour",
    "transactions_last_24h": lambda v: f"{int(v)} transactions in the last 24 h",
    "transactions_last_7d": lambda v: f"{int(v)} transactions this week",
    "merchant_velocity_24h": lambda v: f"merchant hit {int(v)} times in 24 h",
    "device_change_7d": lambda v: "new device this week" if int(v) else "same device",
    "failed_auth_attempts_24h": lambda v: f"{int(v)} failed logins in 24 h",
    "pin_retry_count_24h": lambda v: f"{int(v)} PIN retries in 24 h",
    "account_age_days": lambda v: f"account {int(v)} days old",
    "account_profile_change_7d": lambda v: "profile changed this week" if int(v) else "profile unchanged",
    "billing_shipping_mismatch": lambda v: "billing ≠ shipping address" if int(v) else "addresses match",
    "num_declined_last_30d": lambda v: f"{int(v)} declines in 30 days",
    "high_risk_country": lambda v: "high-risk country" if int(v) else "normal country",
    "international_transaction": lambda v: "international" if int(v) else "domestic",
}



@st.cache_resource
def flag_basis():
    """Typical honest vs fraud values (median) on the model's main signals, from the Jan-Feb training data."""
    b = E.M.base
    tr = b[b["transaction_datetime"] < E.M.REF_START]
    gain = pd.Series(E.M.model.feature_importance("gain"), index=E.M.model.feature_name())
    top = [f for f in gain.sort_values(ascending=False).index if f in E.M.num][:5]
    honest_med = tr[tr["is_fraud"] == 0][E.M.num].median()
    fraud_med = tr[tr["is_fraud"] == 1][E.M.num].median()
    share = (gain / gain.sum())
    return top, honest_med, fraud_med, share


def flag_rule_box():
    top, honest_med, fraud_med, share = flag_basis()
    thr = E.M.thr
    st.markdown(f"""<div class="result"><div class="label">How a transaction is flagged</div>
<div class="why"><b>1. Score.</b> The fraud model (LightGBM, trained on 39,133 labelled January–February transactions)
gives every incoming transaction a fraud score from 0 to 1, using its 42 signals (spending speed, dispute history, device,
merchant, country...).<br>
<b>2. Rule.</b> <b>ALERT if score ≥ {thr:.2f}</b>, otherwise pass. The {thr:.2f} cut-off was chosen on March data to
minimise <i>50 × missed thieves + 1 × false alarm</i>, because one missed thief costs as much as 50 false alarms.<br>
<b>3. Fault found.</b> For each alert, the two signals that pushed its score up the most (the model's own per-signal
contributions), shown next to what is normal for honest customers.</div></div>""", unsafe_allow_html=True)
    fmt = lambda v: f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    st.table(pd.DataFrame({
        "signal the model relies on": top,
        "importance in the model": [f"{share[f]:.0%}" for f in top],
        "typical honest customer": [fmt(float(honest_med[f])) for f in top],
        "typical fraud": [fmt(float(fraud_med[f])) for f in top],
    }).set_index("signal the model relies on"))

def fault_reasons(model, rows, top=2):
    """Plain-words reason for each alert: the features that pushed the score toward fraud the most (LightGBM pred_contrib)."""
    contrib = model.predict(rows[E.M.feats], pred_contrib=True, num_iteration=model.best_iteration or None)[:, :-1]
    out = []
    for r, c in zip(rows[E.M.feats].itertuples(index=False), contrib):
        idx = [i for i in c.argsort()[::-1][:top] if c[i] > 0]
        parts = []
        for i in idx:
            f, v = E.M.feats[i], r[i]
            txt = REASON_TEXT[f](v) if f in REASON_TEXT else f"{f} = {v}"
            if f in E.M.num:
                txt += f" (honest normal: {float(flag_basis()[1][f]):,.3g})"
            parts.append(txt)
        out.append(" · ".join(parts) if parts else "—")
    return out

SHOW_COLS = {"transaction_datetime": "time", "transaction_amount": "amount", "merchant_category": "merchant",
             "merchant_country": "country", "transaction_channel": "channel", "device_type": "device",
             "transactions_last_24h": "txns 24h", "amount_last_1h": "spent 1h", "device_change_7d": "new device 7d",
             "prior_dispute_ratio_90d": "dispute ratio"}


def live_data_sections(eng):
    """What came in today, and what the system concluded from it (only information the system has today)."""
    if eng.i == 0:
        st.info("Press Start or Step 1 day: the incoming transactions and what the system detects will appear here.")
        return
    today = eng.today
    rows = eng.df.loc[eng.by_day[today]]
    src = "Kafka topic txn-live" if eng.scenario == "kafka" and eng.feed.status()["mode"] == "kafka" else (
        "local live feed" if eng.scenario == "kafka" else "transaction stream (CSV replay)")

    st.markdown(f"#### Incoming data · {len(rows):,} transactions received on {today.date()} ({src})")
    latest = rows.sort_values("transaction_datetime", ascending=False)
    flag_rule_box()
    view = pd.concat([latest[latest["alert"]].head(7), latest[~latest["alert"]].head(8)])  # alerts always visible
    view = view.sort_values("transaction_datetime", ascending=False)
    t = view[list(SHOW_COLS)].rename(columns=SHOW_COLS).copy()
    t["time"] = view["transaction_datetime"].dt.strftime("%H:%M")
    t["amount"] = view["transaction_amount"].map("{:,.2f}".format)   # strings: st.table would show 4 decimals
    t["spent 1h"] = view["amount_last_1h"].map("{:,.1f}".format)
    t["dispute ratio"] = view["prior_dispute_ratio_90d"].map("{:.3f}".format)
    t.insert(0, "transaction", view["transaction_id"].astype(str).values)
    t["decision"] = [f"ALERT · {p:.3f} ≥ {E.M.thr:.2f}" if a else f"pass · {p:.3f} < {E.M.thr:.2f}"
                     for a, p in zip(view["alert"], view["p"])]
    reasons = fault_reasons(eng.models[eng.active], view)
    t["fault found"] = [r if a else f"none: score below {E.M.thr:.2f}" for r, a in zip(reasons, view["alert"])]
    arrived = (view["label_day"] <= today).fillna(False).values   # NaT (label never arrives) -> pending
    t["true label"] = [("fraud" if f else "honest") if ok else "pending" for f, ok in zip(view["is_fraud"], arrived)]
    st.table(t.set_index("transaction"))
    st.caption("Today's latest alerts and latest normal transactions. 'Fault found' = the signals that pushed the model "
               "toward fraud the most. 'True label' stays pending until the label arrives (the system never sees it earlier).")

    st.markdown(f"#### What the system detects on {today.date()}")
    c1, c2 = st.columns([2, 3])
    with c1:
        n_alert = int(rows["alert"].sum())
        st.markdown(f"**Fraud model ({'v' + str(eng.active)}):** flagged **{n_alert} of {len(rows):,}** transactions "
                    f"(score ≥ {E.M.thr:.2f}).")
        top = rows.sort_values("p", ascending=False).head(5)
        st.table(pd.DataFrame({"transaction": top["transaction_id"].astype(str).values,
                               "score": top["p"].map("{:.3f}".format).values,
                               "amount": top["transaction_amount"].map("{:,.2f}".format).values,
                               "fault found": fault_reasons(eng.models[eng.active], top, top=1)}).set_index("transaction"))
        lab = eng.df[eng.df["label_day"] == today]
        st.markdown(f"**Late labels arriving today:** {len(lab):,} (fraud {int(lab['is_fraud'].sum())}, "
                    f"missed by the model {int(lab['fn'].sum())}).")
    with c2:
        means = rows[E.M.num].mean()
        z = ((means - eng.ref_mu) / eng.ref_sd).dropna()
        order = z.abs().sort_values(ascending=False).head(8).index
        status = ["ALARM" if f in eng.alarmed else ("out of range" if abs(z[f]) > 3 else "normal") for f in order]
        st.markdown("**Monitor (label-free):** today's average of each feature vs its March normal range. "
                    "3 days in a row out of range = alarm.")
        st.table(pd.DataFrame({
            "feature": list(order),
            "today": [f"{means[f]:,.3f}" for f in order],
            "March normal range": [f"{eng.ref_mu[f] - 3 * eng.ref_sd[f]:,.3f} – {eng.ref_mu[f] + 3 * eng.ref_sd[f]:,.3f}"
                                   for f in order],
            "z": [f"{z[f]:+.1f}" for f in order],
            "days out in a row": [int(eng.streak[f]) for f in order],
            "status": status,
            "fault": [("below normal range" if z[f] < -3 else "above normal range") if abs(z[f]) > 3 else "—"
                      for f in order],
        }).set_index("feature"))

def live_tab():
    st.markdown("""<div class="result"><div class="label">Live prototype · engine.py</div>
<div class="value">Watch the system run, day by day.</div>
<div class="why">Each simulated day the deployed model scores that day's transactions. The monitor sees only the
transactions so far and the labels that have arrived so far. It raises a label-free alarm when a feature leaves its
March range for 3 days, confirms a loss when late labels reveal missed fraud, then <b>retrains and deploys a new model
automatically</b>.</div></div>""", unsafe_allow_html=True)
    live_controls()
    default = "amount_last_1h" if ss.eng.scenario in ("evasion", "kafka") else "device_change_7d"
    feature = st.selectbox("Feature to watch", M_SHOW, index=M_SHOW.index(default), key="lc_feature")

    @st.fragment(run_every=ss.speed if ss.running else None)
    def ticker():
        eng = ss.eng
        if ss.running:
            ev = eng.step()
            if eng.done or any(e["type"] == "model_swap" for e in ev):
                ss.running = False   # pause on the swap / at the end so the presenter can talk
                st.rerun()
        live_render(eng, feature)

    ticker()


M_SHOW = list(E.M.SHOW)


# ---------------- Train the model ----------------
def train_tab():
    st.markdown("""<div class="result"><div class="label">Model training · live</div>
<div class="value">Train the fraud model now.</div>
<div class="why">LightGBM on January–February transactions, early stopping on March, then the alert threshold is chosen
on March to minimise 50 × missed thieves + 1 × false alarm. Time order is kept: the model never sees April–June.</div></div>""",
                unsafe_allow_html=True)
    if st.button("Train the model now", key="tr_run", type="primary"):
        bar = st.progress(0.0, text="training...")

        def tick(it):
            if it % 10 == 0:
                bar.progress(min(it / 350, 1.0), text=f"boosting round {it}")

        res = E.train_live(on_iter=tick)
        b = E.M.base
        d = b[b["transaction_datetime"] >= E.M.STREAM_START]
        res["max_diff"] = float(abs(res["model"].predict(d[E.M.feats], num_iteration=res["best_iter"])
                                    - E.score(E.M.model, d[E.M.feats])).max())
        bar.progress(1.0, text=f"done: stopped at round {res['best_iter']} (no improvement on March for 50 rounds)")
        ss.tr = res
    res = ss.get("tr")
    if res is None:
        st.caption("Press the button: training takes about a second on 39,133 transactions.")
        return

    c = st.columns(4)
    c[0].metric("Best boosting round", res["best_iter"])
    c[1].metric("Alert threshold (cost-chosen)", f"{res['threshold']:.2f}")
    c[2].metric("Training rows (Jan–Feb)", f"{res['train_rows']:,}")
    c[3].metric("Validation rows (Mar)", f"{res['val_rows']:,}")
    same = res["max_diff"] < 1e-9 and res["best_iter"] == int(E.M.model.best_iteration)
    st.markdown(f"{tag('ok' if same else 'alarm', 'REPRODUCIBLE' if same else 'DIFFERENT')} Retrained from scratch: best "
                f"round {res['best_iter']}, largest difference from the deployed model's predictions on Apr–Jun = "
                f"{res['max_diff']:.1g}.", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for (name, ev), color in zip(res["evals"].items(), [MUTED, ACCENT]):
            fig.add_trace(go.Scatter(y=ev["binary_logloss"], mode="lines", name=name, line=dict(color=color, width=2)))
        vline(fig, res["best_iter"], f"best round {res['best_iter']}", SLATE)
        fig.update_xaxes(title="boosting round")
        fig.update_yaxes(title="log loss", type="log")
        style(fig, "Loss curve: training vs validation", height=320)
    with right:
        fn_cost = st.slider("What-if: cost of one missed thief (false alarm = 1)", 10, 200, 50, 10, key="tr_fncost")
        curve = res["cost_curve"].copy()
        curve["cost"] = fn_cost * curve["fn"] + curve["fp"]
        best = curve.loc[curve["cost"].idxmin()]
        fig = go.Figure(go.Scatter(x=curve["threshold"], y=curve["cost"], mode="lines", line=dict(color=SLATE, width=2),
                                   name="cost on March"))
        fig.add_trace(go.Scatter(x=[best["threshold"]], y=[best["cost"]], mode="markers+text", marker=dict(color=ACCENT, size=11),
                                 text=[f"best {best['threshold']:.2f}"], textposition="top center", name="chosen"))
        fig.update_xaxes(title="alert threshold")
        fig.update_yaxes(title=f"{fn_cost} × missed + false alarms", type="log")
        style(fig, "Cost vs alert threshold (March)", height=320)
        if fn_cost != 50:
            st.caption("What-if only: the deployed threshold stays at the problem's 50-to-1 rule.")

    left, right = st.columns([1, 2])
    with left:
        cm = res["confusion"]
        st.markdown(f"**Confusion matrix on March** (threshold {res['threshold']:.2f})")
        st.table(pd.DataFrame({"predicted honest": [cm["tn"], cm["fn"]], "predicted fraud": [cm["fp"], cm["tp"]]},
                              index=["actually honest", "actually fraud"]))
    with right:
        imp = pd.Series(res["model"].feature_importance("gain"), index=res["model"].feature_name())
        imp = (imp / imp.sum()).sort_values(ascending=False).head(12)[::-1]
        fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h", marker_color=SLATE))
        fig.update_xaxes(title="share of total gain (%)")
        style(fig, "What the model relies on (top 12 features)", height=360)


# ---------------- Check a transaction ----------------
EDITABLE = ["amount_last_1h", "transactions_last_24h", "merchant_velocity_24h", "device_change_7d", "prior_dispute_ratio_90d"]


def stream_rows():
    b = E.M.base
    return b[b["transaction_datetime"] >= E.M.STREAM_START]


def pick(kind):
    s = stream_rows()
    pool = s[s["is_fraud"] == (1 if kind == "fraud" else 0)]
    ss.ck_idx = int(pool.sample(1, random_state=int(pd.Timestamp.now().value % 1_000_000)).index[0])
    ss.ck_row = E.M.base.loc[[ss.ck_idx]].copy()   # slice keeps category dtypes (a dict-built row would not)
    ss.ck_note = None
    ss.ck_ver = ss.get("ck_ver", 0) + 1   # resets the edit boxes


def copy_honest_speed():
    b = E.M.base
    honest = b[(b["is_fraud"] == 0) & (b["transaction_datetime"] >= E.M.REF_START) & (b["transaction_datetime"] < E.M.REF_END)]
    donor = honest.sample(1, random_state=int(pd.Timestamp.now().value % 1_000_000))
    for c in E.M.VELOCITY:
        ss.ck_row.loc[ss.ck_idx, c] = donor[c].values[0]
    ss.ck_note = "Velocity features replaced with a random honest March customer's values (the 'thief adapts' attack)."
    ss.ck_ver = ss.get("ck_ver", 0) + 1


def check_tab():
    st.markdown("""<div class="result"><div class="label">Model in action</div>
<div class="value">Score one transaction and see why.</div>
<div class="why">Pick a real transaction from the Apr–Jun stream. The model gives a fraud probability, the alert decision
at the cost-based threshold, and the features that pushed the score up or down (LightGBM's own per-feature contributions).
Then let the thief copy an honest customer's spending speed and score it again.</div></div>""", unsafe_allow_html=True)
    if "ck_row" not in ss:
        pick("fraud")
    b1, b2, b3, b4 = st.columns([1, 1, 1.6, 1.4])
    if b1.button("Random fraud", key="ck_fraud"):
        pick("fraud")
    if b2.button("Random honest", key="ck_honest"):
        pick("honest")
    if b3.button("Thief copies an honest customer's speed", key="ck_copy"):
        copy_honest_speed()
    models = {"v1 (deployed)": E.M.model}
    if len(ss.eng.models) > 1:
        models["v2 (retrained in the control room)"] = ss.eng.models[max(ss.eng.models)]
    mname = b4.radio("Model", list(models), key="ck_model", horizontal=False)
    model = models[mname]

    row = ss.ck_row
    with st.expander("Edit key fields", expanded=False):
        cols = st.columns(len(EDITABLE))
        for c, f in zip(cols, EDITABLE):
            v = c.number_input(f, value=float(row[f].iloc[0]), key=f"ck_{f}_{ss.ck_ver}")
            row.loc[ss.ck_idx, f] = type(row[f].iloc[0])(v) if not isinstance(row[f].iloc[0], float) else v

    p = float(E.score(model, row[E.M.feats])[0])
    alert = p >= E.M.thr
    truth = "fraud" if int(row["is_fraud"].iloc[0]) == 1 else "honest"
    st.markdown(f"""<div class="result"><div class="label">Transaction {row['transaction_id'].iloc[0]} ·
{row['transaction_datetime'].iloc[0]:%Y-%m-%d %H:%M} · true label (hidden from the model): {truth}</div>
<div class="value">{'ALERT' if alert else 'No alert'} · fraud probability {p:.3f}</div>
<div class="why">Threshold {E.M.thr:.2f} (one missed thief costs as much as 50 false alarms).
{('<br><i>' + ss.ck_note + '</i>') if ss.get('ck_note') else ''}</div></div>""", unsafe_allow_html=True)

    top, bias = E.explain(model, row)
    fig = go.Figure(go.Bar(x=top["contribution"][::-1], y=[f"{f} = {v}" for f, v in zip(top["feature"], top["value"])][::-1],
                           orientation="h", marker_color=[ACCENT if c > 0 else SAGE for c in top["contribution"][::-1]]))
    fig.update_xaxes(title="push toward fraud (log-odds)  ·  red = toward fraud, green = toward honest")
    style(fig, "Why: top 8 feature contributions", height=360)

t0, tc, tt, t1, t2, t3, t4 = st.tabs(["Live control room", "Check a transaction", "Train the model", "1 · Real stream",
                                      "2 · Thieves read the alerts", "3 · Model & cost", "4 · Method & limits"])

with t0:
    live_tab()

with tc:
    check_tab()

with tt:
    train_tab()


# ---------------- Tab 1: real data ----------------
with t1:
    dg = rs["diagnosis"]["device_change_7d"]
    st.markdown(f"""<div class="result"><div class="label">Real data · label-free alarm</div>
<div class="value">The world changed on {rs['first_label_free_alarm']}: device changes doubled.</div>
<div class="why">Honest-looking traffic changed <b>{dg['not_alerted']['change_pct']:+}%</b>, alerted traffic
<b>{dg['alerted']['change_pct']:+}%</b> → verdict: <b>{dg['verdict']}</b> changed.
The late labels never show a rise in missed fraud, so the model is kept as it is.</div></div>""",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label-free alarm", rs["first_label_free_alarm"], help="Daily mean outside March ±3σ, 3 days in a row")
    c2.metric("Dataset's own drift start", "2026-04-01", help="Used only to check the detector, never as input")
    c3.metric("Missed thieves (Apr–Jun)", rs["stream_fn"])
    c4.metric("Stream cost", f"{rs['stream_cost']:,.0f}", help="50 × missed + 1 × false alarms")

    st.markdown("**Replay: drag the day to see what the monitor knew at that point** (the chart and the feed follow)")
    days = rd.loc[rd["day"] >= LABELED_ONSET, "day"].dt.date.tolist()
    pick = st.select_slider("Day", options=days, value=days[-1], key="real_day")
    feature_chart(rd, rs, "device_change_7d", "Share of transactions after a device change in the last 7 days",
                  [(LABELED_ONSET, "dataset's drift start", MUTED), (rs["first_label_free_alarm"], "our alarm", ACCENT)],
                  upto=pick)
    feed(rd, pd.Timestamp(pick), rs)

    st.markdown("**Who changed? (label-free diagnosis: split by the model's own alert)**")
    diagnosis_table(rs, list(rs["alarms_by_feature"]))
    with st.expander("Other signal: account age"):
        feature_chart(rd, rs, "account_age_days", "Average account age (grows over time; slow drift)",
                      [(rs["first_alarm_psi"], "PSI alarm", ACCENT)])

# ---------------- Tab 2: evasion simulation ----------------
with t2:
    rt = ev.get("retrain", {})
    st.markdown(f"""<div class="result"><div class="label">Simulation on the real stream · clearly a test</div>
<div class="value">Thieves changed tactics on {mon['settings']['evade_from']}. We knew on {es['lead_measured_from']},
{es['lead_days_before_answers']} days before the answers.</div>
<div class="why">From May 1, {int(mon['settings']['evade_share'] * 100)}% of thieves ({es['adapted_frauds']:,} transactions)
copy the spending speed of honest customers. Speed signals leave their normal range {es['detection_delay_days']} days later.
Most are still caught, because thieves can copy speed but not their dispute history:
{es['missed_after_change']} of {es['frauds_after_change']:,} frauds after the change slip through.
The late labels confirm the losses on <b>{es['first_label_confirmation']}</b>. Retraining then cuts the cost after that day
from <b>{rt.get('old_model', {}).get('cost', 0):,.0f}</b> to <b>{rt.get('retrained', {}).get('cost', 0):,.0f}</b>.
Baseline = wait for the labels; the warning is measured against that.</div></div>""",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tactic change (planted)", mon["settings"]["evade_from"])
    c2.metric("Label-free alarm", es["lead_measured_from"])
    c3.metric("Answers confirm", es["first_label_confirmation"])
    c4.metric("Warning before answers", f"{es['lead_days_before_answers']} days")

    st.markdown("**Replay: drag the day to see what the monitor knew at that point** (both charts and the feed follow)")
    days2 = ed.loc[ed["day"] >= LABELED_ONSET, "day"].dt.date.tolist()
    pick2 = st.select_slider("Day", options=days2, value=days2[-1], key="ev_day")
    feature_chart(ed, es, "amount_last_1h", "Average amount spent in the previous hour (speed signal)",
                  [(EVADE_FROM, "tactic change", MUTED), (es["lead_measured_from"], "our alarm", ACCENT)], upto=pick2)
    label_chart(ed, es, "Missed frauds revealed by late labels",
                [(EVADE_FROM, "tactic change", MUTED), (es["lead_measured_from"], "our alarm", ACCENT),
                 (es["first_label_confirmation"], "answers confirm", OCHRE)], upto=pick2)
    feed(ed, pd.Timestamp(pick2), es)

    st.markdown("**Who changed?** Here the alerted traffic moves, so the diagnosis points at the thieves.")
    diagnosis_table(es, [f for f in es["alarms_by_feature"] if f in mon["settings"]["velocity_features"]][:3])

    if rt:
        st.markdown(f"**Retraining after the answers confirm** (retrained on {rt['train_rows']:,} labelled rows "
                    f"available by {rt['retrain_day']}; scored on the {rt['rows_after']:,} transactions after it)")
        fig = go.Figure(go.Bar(x=["Old model", "Retrained"], y=[rt["old_model"]["cost"], rt["retrained"]["cost"]],
                               marker_color=[MUTED, ACCENT],
                               text=[f"cost {rt['old_model']['cost']:,.0f}<br>missed {rt['old_model']['fn']} · "
                                     f"false alarms {rt['old_model']['fp']}",
                                     f"cost {rt['retrained']['cost']:,.0f}<br>missed {rt['retrained']['fn']} · "
                                     f"false alarms {rt['retrained']['fp']}"],
                               textposition="outside"))
        fig.update_yaxes(range=[0, rt["old_model"]["cost"] * 1.3])
        style(fig, "Cost after the retrain day", height=320)

    # ---- How small a change can we catch? ----
    if sens:
        st.markdown("**How small a change can we catch?** "
                    f"({len(sens['grid']['shares'])} shares × {len(sens['grid']['starts'])} start dates × "
                    f"{len(sens['grid']['seeds'])} random seeds)")
        tab = pd.DataFrame(sens["table"])

        def cell(r):
            if pd.isna(r["median_lead_days"]):
                return f"missed ({r['caught_label_free']})"
            return f"{r['caught_label_free']} · alarm +{r['median_delay_days']:.0f} d · warns {r['median_lead_days']:.0f} d early"
        tab["result"] = tab.apply(cell, axis=1)
        grid = tab.pivot(index="share", columns="start", values="result")
        grid.index = [f"{int(s * 100)}% of thieves adapt" for s in grid.index]
        st.table(grid)
        st.caption("Cell = runs caught by the label-free layer · days from change to alarm · days of warning before "
                   "the labels confirm. 'missed' = only the late labels catch it.")

    # ---- Live what-if ----
    with st.expander("Try it: change the attack and re-run the monitor live", expanded=False):
        cc1, cc2 = st.columns(2)
        share = cc1.slider("Share of thieves who adapt", 0.1, 0.9, 0.7, 0.1)
        start = cc2.select_slider("Tactic change date", options=[str(d.date()) for d in
                                  pd.date_range("2026-04-15", "2026-06-01", freq="7D")], value="2026-05-06")
        if st.button("Run monitor"):
            with st.spinner("Re-running the monitor and the retrain..."):
                ws, wd, wrt = what_if(share, start)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Label-free alarm", ws["lead_measured_from"] or "missed")
            m2.metric("Answers confirm", ws["first_label_confirmation"] or "not yet")
            m3.metric("Warning before answers",
                      f"{ws['lead_days_before_answers']} days" if ws["lead_days_before_answers"] is not None else "none")
            if wrt:  # short value so it fits a laptop-width card; change vs old model goes in the delta
                old, new = wrt["old_model"]["cost"], wrt["retrained"]["cost"]
                m4.metric("Cost after retrain", f"{new:,.0f}", delta=f"{new - old:,.0f} vs old model {old:,.0f}",
                          delta_color="inverse")
            else:
                m4.metric("Cost after retrain", "no retrain")
            st.caption(f"{ws['adapted_frauds']:,} adapted fraud transactions · {ws['missed_after_change']} of "
                       f"{ws['frauds_after_change']:,} frauds after the change missed by the model.")
            feature_chart(wd, ws, "amount_last_1h", "Average amount spent in the previous hour",
                          [(start, "tactic change", MUTED), (ws["lead_measured_from"], "our alarm", ACCENT),
                           (ws["first_label_confirmation"], "answers confirm", OCHRE)])

# ---------------- Tab 3: model ----------------
with t3:
    v, s = met["val"], met["stream"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "LightGBM")
    c2.metric("Alert threshold", f"{met['threshold']:.2f}", help="Chosen on March to minimise 50×missed + 1×false alarm")
    c3.metric("ROC-AUC (Apr–Jun)", f"{s['auc']:.5f}")
    c4.metric("PR-AUC (Apr–Jun)", f"{s['pr_auc']:.5f}")
    st.markdown(f"""
- **Train** Jan–Feb · **tune threshold** March · **test** Apr–Jun (time order kept, no shuffling).
- **Why the threshold is low ({met['threshold']:.2f}):** one missed thief costs as much as 50 false alarms, so the model
  should alert early. On Apr–Jun it missed **{s['fn']}** thief and raised **{s['fp']}** false alarms out of {s['rows']:,}.
- **29 columns removed as leakage:** drift labels, fraud type, dispute and chargeback outcomes, IDs, label timing.
  These are only known after the transaction, or they give away the answer.
- **Caution:** the score is near-perfect because spending-speed features almost separate fraud in this dataset.
  Real fraud is harder. The monitor is the point of this project, not the score.
""")

# ---------------- Tab 4: method ----------------
with t4:
    if sens:
        fired = sum(len(p["features_alarmed"]) for p in sens["placebo"])
        placebo_line = (f"{fired} false alarms across {len(sens['placebo'])} windows × "
                        f"{sens['placebo'][0]['features']} features")
    else:
        placebo_line = "not run (run sensitivity.py)"
    st.markdown(f"""
**Data.** Public Kaggle dataset *Credit Card Fraud Detection with Drift 2026* (120,000 transactions, 73 columns, Jan–Jun 2026,
fraud rate 9.5%, label delay 0–45 days). The organizers provided no data. The dataset's drift is engineered.

**Reference period.** March. January–February has a warm-up artifact: `days_since_last_transaction` starts at a 90-day cap
for most rows and decays, so a Jan–Feb reference raises false alarms.

**Label-free layer (before the answers).**
1. Daily mean of each feature vs its March mean ± 3σ, **3 days in a row** to ignore one-day noise. This is the layer that fired.
2. 7-day PSI per feature (> {mon['settings']['psi_alarm']} = significant). Slow; caught only the account-age trend.
3. Daily alert rate vs March ± 3σ. Never fired: the evasion only changes the alert rate by under one point.

**False alarms (placebo test).** The same rule run on 3 calm windows where nothing changed
(Feb→Mar, early→late March, mid-Jan→mid-Feb reference): **{placebo_line}**.

**Who changed?** Split the window by the model's own alert flag. Each group's *shift explained* = its share of traffic ×
the change in its mean. If non-alerted traffic explains more, honest customers changed; if alerted traffic explains more,
thieves changed. No labels needed.

**Label-based layer (when the answers arrive).** Each missed fraud is counted on the day its label arrives
(transaction day + its real delay; labels that never arrive are not counted). 7-day sum vs March limit
(March has almost no missed fraud, so in practice 2 missed frauds in 7 days confirms). This confirms or rejects the
early alarm. Baseline to beat: a team that waits for these labels.

**Action.** Retrain only after the answers confirm a loss, on all labels available by then.

**Limits.**
- The evasion scenario is a simulation planted on the real stream, not observed thief behavior.
- The detector watches feature averages. A tactic change that keeps every average the same would not be seen early.
- Daily windows: the alarm needs 3 days of evidence.
- Small changes: when only 10–30% of thieves adapt, the label-free layer misses it and only the late labels catch it.
- Part of the warning comes from slow labels: in this dataset Apr–May fraud labels arrive after ~30 days (median).
- Thieves who evade completely look like normal traffic, so a pure shift in non-alerted traffic could also be hidden
  thieves. The late labels are the tie-breaker (on the real data they show the fraud share flat → honest customers).
- The fraud model scores near-perfectly because of this dataset's speed features; real fraud is harder.
""")
