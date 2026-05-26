import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

variants = {
    # Current best style, but slightly more A_big
    "v9_more_abig_378": {
        "A_small": 0.198,
        "A_big": 0.240,
        "B_small": 0.184,
        "B_big": 0.378
    },

    # Slightly more A_small
    "v10_more_asmall_378": {
        "A_small": 0.205,
        "A_big": 0.233,
        "B_small": 0.184,
        "B_big": 0.378
    },

    # Slightly more B_small
    "v11_more_bsmall_378": {
        "A_small": 0.199,
        "A_big": 0.235,
        "B_small": 0.188,
        "B_big": 0.378
    },

    # Slightly lower B_big but more A_big
    "v12_abig_bbig376": {
        "A_small": 0.200,
        "A_big": 0.241,
        "B_small": 0.183,
        "B_big": 0.376
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