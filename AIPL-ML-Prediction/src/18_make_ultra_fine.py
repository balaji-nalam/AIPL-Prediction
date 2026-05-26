import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

variants = {
    "v27_abig2425": {
        "A_small": 0.1955,
        "A_big": 0.2425,
        "B_small": 0.1840,
        "B_big": 0.3780
    },

    "v28_abig2430": {
        "A_small": 0.1950,
        "A_big": 0.2430,
        "B_small": 0.1840,
        "B_big": 0.3780
    },

    "v29_abig2425_bbig3785": {
        "A_small": 0.1950,
        "A_big": 0.2425,
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

    out_path = SUB_DIR / f"submission_{name}.csv"
    sub.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(sub[class_cols].head(1))