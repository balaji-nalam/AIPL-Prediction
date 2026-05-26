import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

variants = {
    "v22_abig2415": {
        "A_small": 0.1965,
        "A_big": 0.2415,
        "B_small": 0.1840,
        "B_big": 0.3780
    },

    "v23_abig2420": {
        "A_small": 0.1960,
        "A_big": 0.2420,
        "B_small": 0.1840,
        "B_big": 0.3780
    },

    "v24_bbig379": {
        "A_small": 0.1960,
        "A_big": 0.2410,
        "B_small": 0.1840,
        "B_big": 0.3790
    },

    "v25_bsmall183": {
        "A_small": 0.1980,
        "A_big": 0.2410,
        "B_small": 0.1830,
        "B_big": 0.3780
    },

    "v26_abig2415_bbig3785": {
        "A_small": 0.1960,
        "A_big": 0.2415,
        "B_small": 0.1840,
        "B_big": 0.3785
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