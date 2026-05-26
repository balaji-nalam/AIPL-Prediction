import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

variants = {
    "v6_bbig_375": {
        "A_small": 0.202,
        "A_big": 0.236,
        "B_small": 0.187,
        "B_big": 0.375
    },

    "v7_bbig_378": {
        "A_small": 0.201,
        "A_big": 0.236,
        "B_small": 0.185,
        "B_big": 0.378
    },

    "v8_bbig_382": {
        "A_small": 0.199,
        "A_big": 0.236,
        "B_small": 0.183,
        "B_big": 0.382
    }
}

for name, probs in variants.items():

    sub = sample.copy()

    for col in class_cols:
        sub[col] = probs[col]

    sub[class_cols] = sub[class_cols].div(
        sub[class_cols].sum(axis=1),
        axis=0
    )

    assert sub.shape[0] == 53
    assert sub[class_cols].isnull().sum().sum() == 0
    assert ((sub[class_cols] >= 0) & (sub[class_cols] <= 1)).all().all()
    assert ((sub[class_cols].sum(axis=1) - 1.0).abs() < 0.001).all()

    out_path = SUB_DIR / f"submission_{name}.csv"
    sub.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(sub[class_cols].head(1))