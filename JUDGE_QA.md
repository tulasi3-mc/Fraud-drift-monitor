# Judge Q&A cheat sheet (The Moving Target)

## 30-second pitch

Fraud labels arrive days or weeks late, so a fraud team learns about new tactics only after the money is gone.
Our monitor watches the transactions themselves. When a signal leaves its normal range for 3 days, it raises an alarm,
names the signal, and says whether honest customers or thieves changed. When the late labels arrive, it checks itself and
retrains only if there is a real loss. In our simulation it warned 18 days before the labels did, and retraining cut the
cost from 7,217 to 2,357.

## Likely questions

**Where is your data from? The organizers gave none.**
A public Kaggle dataset (CC0 license): 120,000 card transactions, January to June 2026, with real label delays of 0–45 days.
Its drift is engineered by the uploader, and we say so.

**How do you know the world changed without labels?**
For each feature we learn its normal daily average in March (mean ± 3 standard deviations). If a feature stays outside
that range for 3 days in a row, we raise an alarm. Three days avoids reacting to one noisy day.

**What did you find on the real data?**
On April 3, the share of transactions after a device change doubled (about 7.5% to 14%). The dataset's own drift label
says the change started April 1, so we caught it in 2 days, without using that label.

**How do you know who changed?**
We split the traffic by our own model's alert. Rows the model doesn't flag are mostly honest; rows it flags are mostly
thieves. We measure how much of the shift each group explains (its share of traffic × change in its mean). On the real
data the non-flagged rows explain it (+65% vs −8.5%), so honest customers changed.
The labels later confirmed it: honest customers 5.3% → 13.1%, thieves flat at about 30%.

**So why not retrain on April 3?**
Because the late labels never showed more missed fraud. Retraining on a change that doesn't hurt wastes effort and can
make the model worse. We retrain only when labels confirm a loss.

**"They read your alerts": how did you test that?**
We simulated it on the real stream: from May 1, 70% of thieves copy the spending speed of honest customers so the model
misses them. Most are still caught, because thieves can copy speed but not their dispute history: 243 of 3,895 frauds
slip through. The speed signals left their normal range on May 3. The late labels confirmed the losses on May 21. That is
18 days of warning. Retraining on the labels available on May 21 cut the cost afterwards from 7,217 to 2,357.

**Isn't the simulation fake?**
It is a test, and we label it as one. It runs on the real transactions; only the thieves' speed values are swapped for
honest ones. It proves the detector reacts to the kind of change the problem describes.

**Why is the alert threshold so low (0.11)?**
Missing one thief costs as much as 50 false alarms. We picked the threshold on March data to minimise
50 × missed + 1 × false alarms.

**Your AUC is 0.99999. Isn't that suspicious?**
Yes, and we checked. We removed 29 leakage columns (drift labels, fraud type, dispute and chargeback outcomes, IDs).
The high score remains because spending-speed features nearly separate fraud in this dataset. Real fraud is harder.
Our contribution is the monitor, not the score.

**Why March as the reference, not January?**
In January–February one feature (`days_since_last_transaction`) starts at a 90-day cap and decays. Using that period
raises false alarms, so we use March, the stable month before the drift.

**Why LightGBM?**
Tabular data, fast to train, handles categories, and easy to retrain when labels confirm a change.

**What doesn't it catch?**
A change that keeps every feature average the same. Also, the alarm needs 3 days of evidence.

**How would this work in real life?**
The monitor needs only feature averages per day and the model's alerts, so it is cheap to run on any stream.
A bank would plug in its own reference month and label feed.

**What if only a few thieves adapt?**
We tested it: 5 shares × 3 start dates × 3 random seeds. When 70–90% adapt we catch it every time, 2 days after the change,
13–16 days before the labels. At 50% we catch 7 of 9. At 10–30% the label-free layer misses it and only the late labels
catch it. We say that openly.

**How many false alarms?**
We ran the same rule on 3 calm windows where nothing changed: 0 alarms across 33 numeric features.

**Couldn't hidden thieves look like "honest customers changed"?**
Yes. Thieves who evade completely look like normal traffic. That is why the late labels are the tie-breaker: on the real
data they show the fraud share flat, so it really was honest customers.

**Isn't the 18-day warning just slow labels?**
Partly. In this dataset Apr–May fraud labels arrive after about 30 days. That is exactly the problem's point: the answers
arrive late, and our alarm doesn't wait for them.

**Can you show it live?**
Yes: Tab 2 → "Try it". Pick the share of thieves and the date, press Run, and the monitor recomputes in a few seconds.

**Is this live, or just precomputed charts?**
Live. The "Live control room" runs the system day by day: the deployed model scores each day's transactions, the monitor
raises alarms, late labels arrive, and on confirmation it retrains and deploys model v2 by itself. "Train the model"
trains LightGBM in about a second, in front of you.

**How do we know the live run is right?**
`check_engine.py` replays both scenarios day by day and checks every result against the batch analysis: same alarm dates,
same confirmation date, same 75,256 retraining rows, same costs (2,357 vs 7,217). It prints PASS.

**Can the model explain one decision?**
Yes: "Check a transaction" shows the fraud probability and the features that pushed it up or down. When the thief copies
an honest customer's speed, the score drops, but `prior_dispute_ratio_90d` still pushes it over the threshold.

**Is the Kafka data real?**
The transactions are new: real March transactions resampled onto each new day with fresh IDs, sent one by one through a
local Kafka broker by `producer.py`. Kafka is the pipe a bank would use; in production the producer would be the card
system. "Launch attack" makes thieves copy honest spending speed from that day on, and the alarm fires about 2 days later.

**What if Kafka goes down during the demo?**
The app checks the broker in 2 seconds. If it is not reachable, it shows "Kafka not reachable, using local feed" and runs
the same generator inside the app, so the demo keeps working.
