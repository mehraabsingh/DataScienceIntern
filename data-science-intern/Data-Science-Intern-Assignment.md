# Data Science Intern Assignment
**Company:** Millipixels Interactive (Cerebrent Group) · **Duration: 2 hours**

---

## Objective

Given one year of shipment data, predict **which deliveries will arrive late**,
evaluate the model honestly, explain *why* deliveries are late, and turn that
into one recommendation the operations team could act on tomorrow.

We care more about how you **evaluate and explain** the model than about the
last decimal of accuracy.

---

## Scenario

**BharatBasket** is a fictional online grocery and personal-care brand. Every
order gets a promised delivery time — "3 days" — shown to the customer at
checkout. Roughly **one in four shipments misses that promise**, and each miss
costs a support call, a refund request, or a lost customer.

Operations wants to know two things:

1. **At dispatch time**, which shipments are at risk — so they can upgrade the
   courier or warn the customer before they complain.
2. **In general**, what actually drives lateness — so they can fix the cause.

---

## Dataset

In the `data/` folder next to this file:

| File | Rows | What it is |
|---|---:|---|
| `shipments_train.csv` | 9,060 | Historical shipments **with** the `is_late` label |
| `shipments_holdout.csv` | 3,000 | Shipments to predict — **no label** |
| `sample_submission.csv` | 5 | The exact output format we expect |

### Columns

| Column | Meaning |
|---|---|
| `shipment_id` | Unique ID |
| `dispatch_date` | Date the parcel left the warehouse |
| `warehouse_id` | Origin warehouse (`WH-CHD`, `WH-DEL`, `WH-BLR`, `WH-PUN`) |
| `destination_city` | Delivery city |
| `courier_partner` | Courier used |
| `distance_km` | Warehouse → destination distance |
| `weight_kg` | Parcel weight |
| `items_count` | Number of items in the parcel |
| `order_value_inr` | Order value |
| `promised_days` | Days promised to the customer at checkout |
| `dispatch_hour` | Hour of day the parcel was dispatched (8–21) |
| `is_weekend_dispatch` | 1 if dispatched on Sat/Sun |
| `is_festival_week` | 1 if dispatched during a festival week |
| `rainfall_mm` | Rainfall on the dispatch date |
| `customer_prior_late_deliveries` | How many late deliveries this customer had before |
| `customer_segment` | `Retail` / `Business` / `Wholesale` |
| `payment_method` | `UPI` / `Card` / `COD` / `Netbanking` / `Wallet` |
| `actual_delivery_days` | Days the delivery actually took |
| `delivery_note` | Free-text note left by the delivery agent |
| **`is_late`** | **Target.** 1 = missed the promise, 0 = on time |

> ⚠️ **Read this carefully.** `shipments_holdout.csv` has **three fewer columns**
> than the training file. Compare the two headers before you build features and
> ask yourself *why* those columns are missing — the answer matters, and it is
> worth marks.

The data is realistically imperfect: missing values, a few duplicate rows,
inconsistent city capitalisation, and some impossible values. Handle them.

*(BharatBasket is fictional and the data is synthetic.)*

---

## What you need to run this

Everything runs on a **normal student laptop.** No GPU, no cloud account, no
internet connection during the assignment, and nothing to pay for.

| | |
|---|---|
| **Disk** | 1.3 MB of data |
| **RAM** | Our reference run peaked at **241 MB** |
| **Runtime** | The whole notebook — load, clean, 4 charts, train 3 models, score the holdout — takes about **13 seconds**, and 10 of those are the `import` statements |
| **Model fit times** | LogisticRegression **0.07 s** · RandomForest, 300 trees **0.61 s** · GradientBoosting **1.59 s** |
| **Python** | 3.9 or newer |

Install once:

```bash
pip install pandas scikit-learn matplotlib
```

9,000 rows and roughly 40 features is a **small** dataset by machine-learning
standards. You do not need a GPU, deep learning, XGBoost, or any heavy library —
scikit-learn on CPU is the right tool and it is effectively instant here.

If a model takes more than a minute to fit, you have almost certainly created far
too many one-hot columns. Check `X.shape` before you blame your laptop: around
40 columns is expected, and a few thousand means you one-hot encoded something
you should not have.

**Harmless warning:** on macOS with numpy 2.x you may see
`RuntimeWarning: divide by zero encountered in matmul`. We verified it fires on
clean random data too — it is a known numpy/Accelerate quirk, not a problem with
your code or this dataset, and the results are still correct.

---

## Tasks

### Task 1 — Explore the data (25 minutes)

- Load the training data. Report shape, dtypes, missing values per column,
  duplicates, and the **class balance** of `is_late`.
- Produce **at least four charts** that tell you something real, for example:
  - late rate by `courier_partner`
  - late rate by `promised_days`
  - distribution of `distance_km` for late vs on-time shipments
  - late rate by month of `dispatch_date`
- Write **3–5 bullet points**: what did you learn, and what surprised you?

### Task 2 — Clean and build features (25 minutes)

- Handle missing values, duplicate rows and impossible values (there are
  negative distances and at least one absurd weight). Say what you did and why.
- Standardise the messy categorical text.
- Encode categoricals sensibly (one-hot or ordinal — your call, justify it).
- Create **at least three new features** of your own, for example:
  - day of week / month from `dispatch_date`
  - `distance_km / promised_days` — distance the courier must cover per promised day
  - a "late dispatch" flag for parcels leaving after a cut-off hour
- **Decide which columns must not be used as features, and say why.**

### Task 3 — Baseline and model (30 minutes)

1. **Baseline first.** Predict "never late" for everything and report its
   accuracy. Note what that number tells you about using accuracy as a metric.
2. Split the training data properly. Explain your split — and whether a random
   split is the right choice for data that has dates in it.
3. Train **two** models — e.g. Logistic Regression and a tree-based model
   (RandomForest / GradientBoosting). Both are in scikit-learn.
4. Report **accuracy, precision, recall, F1 and ROC-AUC** for each, plus a
   confusion matrix for your better model.

### Task 4 — Evaluate and analyse the errors (25 minutes)

- **Pick the metric that matters here and defend it in two lines.** Operations
  would rather investigate 10 shipments that turn out fine than miss a real
  late one. What does that imply about precision vs recall?
- Tune the decision threshold — it does not have to be 0.5. Show the metric at
  two or three thresholds and pick one.
- Show **feature importances or coefficients** for your chosen model.
- **Error analysis:** look at the shipments the model got wrong. Is there a
  group it consistently fails on — a courier, a city, a distance band, a season?
  Two or three sentences.
- Score the holdout set and write `predictions.csv` in exactly this format:

  ```csv
  shipment_id,is_late_probability,is_late_prediction
  HLD000001,0.81,1
  HLD000002,0.12,0
  ```

  We score it against the true labels, which we hold back.

### Task 5 — Turn it into a business recommendation (15 minutes)

In `REPORT.md` (**one page maximum**):

- Your headline result: chosen model, chosen metric, chosen threshold, and the
  number it scores.
- The **top three drivers** of lateness, in plain English a non-technical
  operations manager would understand.
- **One specific, concrete recommendation.** Not "improve logistics" — something
  like *"route festival-week shipments over 800 km away from ValueShip; on this
  data that would have prevented roughly N late deliveries."*
- Where your model is weakest, and what you would do next with one more week.

*(~5 minutes are left as a buffer for setup and debugging.)*

---

## Deliverables

| File | What it is |
|---|---|
| `analysis.ipynb` **or** `analysis.py` | EDA, features, models, evaluation — with your charts |
| `predictions.csv` | Holdout predictions in the format above |
| `REPORT.md` | The one-page write-up from Task 5 |
| `README.md` | 5 lines: how to run it, and any assumption you made |

Zip the folder, or share a Git repo link.

---

## Bonus (only if you finish early — worth nothing if the basics are incomplete)

- Cross-validation instead of a single split.
- Any hyperparameter tuning, with the search shown.
- SHAP values or a partial dependence plot for one important feature.
- Convert the probability into a rupee figure: what is a late delivery worth?
- A small dashboard of the late-risk breakdown. Streamlit works on any OS;
  Power BI Desktop is **Windows-only** — skip it on Mac or Linux, it is not
  required and carries no marks of its own.

---

## Constraints

- **Python only.** pandas, numpy, scikit-learn, matplotlib/seaborn. No GPU, no
  internet, no deep learning needed.
- Works identically on Windows, macOS and Linux.
- **Set `random_state=42`** everywhere randomness affects a result, so your
  numbers reproduce.
- Charts must be readable: axis labels and a title on every one.
- Do not use any column that would not exist at dispatch time.

---

## How we score this

| Weight | Area |
|---:|---|
| 25% | Evaluation setup — right metric, honest validation, **no data leakage** |
| 20% | Modelling — sensible features, working models, reasonable performance |
| 20% | Error analysis — did you find where and why the model fails? |
| 20% | Business communication — is `REPORT.md` useful to a non-technical reader? |
| 15% | Code clarity and reproducibility |

**A model with 0.79 ROC-AUC and an honest account of what it gets wrong scores
higher than a 0.99 that leaked the answer.** If a result looks too good, that is
a finding to investigate, not a number to submit.

---

## Practical advice

- Watch the clock. **Finishing all five tasks roughly** beats perfecting Task 1.
- Where the assignment is ambiguous, **make a reasonable call, write one line in
  `README.md` explaining it, and keep going.** That is what the job looks like.
- You may use AI tools. Add one line to `README.md` saying where you used them.
  We will ask you to explain your own code in the interview.
