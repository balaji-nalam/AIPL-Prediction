import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

probs = {
    "A_small": 0.195,
    "A_big": 0.235,
    "B_small": 0.180,
    "B_big": 0.390
}

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

out_path = SUB_DIR / "submission_v5_bbig_390.csv"
sub.to_csv(out_path, index=False)

print("Saved:", out_path)
print(sub.head())
print("Average probabilities:")
print(sub[class_cols].mean())