import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

SUB_DIR.mkdir(exist_ok=True)

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

variants = {
    "v1_more_bbig": {
        "A_small": 0.200000,
        "A_big": 0.235000,
        "B_small": 0.185000,
        "B_big": 0.380000
    },

    "v2_less_bbig": {
        "A_small": 0.215000,
        "A_big": 0.240000,
        "B_small": 0.195000,
        "B_big": 0.350000
    },

    "v3_more_abig": {
        "A_small": 0.205000,
        "A_big": 0.250000,
        "B_small": 0.185000,
        "B_big": 0.360000
    },

    "v4_more_bsmall": {
        "A_small": 0.205000,
        "A_big": 0.235000,
        "B_small": 0.205000,
        "B_big": 0.355000
    }
}

for name, probs in variants.items():

    sub = sample.copy()

    for col in class_cols:
        sub[col] = probs[col]

    # Normalize rows
    sub[class_cols] = sub[class_cols].div(
        sub[class_cols].sum(axis=1),
        axis=0
    )

    # Safety checks
    assert sub.shape[0] == 53, "Submission must have 53 rows"
    assert sub[class_cols].isnull().sum().sum() == 0, "NaN found"
    assert ((sub[class_cols] >= 0) & (sub[class_cols] <= 1)).all().all(), "Invalid probability"
    assert ((sub[class_cols].sum(axis=1) - 1.0).abs() < 0.001).all(), "Rows do not sum to 1"

    out_path = SUB_DIR / f"submission_{name}.csv"
    sub.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(sub[class_cols].head(1))
    