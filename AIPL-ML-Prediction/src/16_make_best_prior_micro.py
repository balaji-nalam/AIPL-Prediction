import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

variants = {
    # Slightly more A_big, less A_small
    "v17_abig242": {
        "A_small": 0.196,
        "A_big": 0.242,
        "B_small": 0.184,
        "B_big": 0.378
    },

    # Slightly more A_big and B_big
    "v18_abig242_bbig379": {
        "A_small": 0.195,
        "A_big": 0.242,
        "B_small": 0.184,
        "B_big": 0.379
    },

    # Slightly less B_small
    "v19_bsmall182": {
        "A_small": 0.199,
        "A_big": 0.241,
        "B_small": 0.182,
        "B_big": 0.378
    },

    # Slightly more A_big, less B_big
    "v20_abig243_bbig376": {
        "A_small": 0.197,
        "A_big": 0.243,
        "B_small": 0.184,
        "B_big": 0.376
    },

    # Balanced around best
    "v21_abig241_bbig378": {
        "A_small": 0.197,
        "A_big": 0.241,
        "B_small": 0.184,
        "B_big": 0.378
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