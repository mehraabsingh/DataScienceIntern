# Shipment data — BharatBasket

| File | Rows | What it is |
|---|---:|---|
| `shipments_train.csv` | 9,060 | Historical shipments **with** the `is_late` label |
| `shipments_holdout.csv` | 3,000 | Shipments to predict — **no label** |
| `sample_submission.csv` | 5 | The exact output format expected |

`shipments_holdout.csv` has **three fewer columns** than the training file.
Compare the two headers before you build features — the reason matters.

Your `predictions.csv` must cover all 3,000 holdout shipments:

```csv
shipment_id,is_late_probability,is_late_prediction
HLD000001,0.81,1
HLD000002,0.12,0
```

Read `../Data-Science-Intern-Assignment.md` for the task.

*BharatBasket is fictional; this data is synthetic.*
