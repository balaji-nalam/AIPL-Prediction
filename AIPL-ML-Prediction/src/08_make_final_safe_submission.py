import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

SUB_DIR.mkdir(exist_ok=True)

sample_path = DATA_DIR / "sample_submission.csv"

sample = pd.read_csv(sample_path)

print("Sample shape:", sample.shape)

class_cols = ["A_small", "A_big", "B_small", "B_big"]

# Safer validation-based class prior
# Order must match submission columns, not label encoder order
priors = {
    "A_small": 0.20689655172413793,
    "A_big": 0.23692992213570635,
    "B_small": 0.19021134593993325,
    "B_big": 0.3659621802002225
}

submission = sample.copy()

for col in class_cols:
    submission[col] = priors[col]

# Normalize exactly
submission[class_cols] = submission[class_cols].div(
    submission[class_cols].sum(axis=1),
    axis=0
)

# Safety checks
assert submission.shape[0] == 53, "Submission must have 53 rows"
assert submission[class_cols].isnull().sum().sum() == 0, "NaN found"
assert ((submission[class_cols] >= 0) & (submission[class_cols] <= 1)).all().all(), "Invalid probability"

row_sums = submission[class_cols].sum(axis=1)
assert ((row_sums - 1.0).abs() < 0.001).all(), "Rows do not sum to 1"

out_path = SUB_DIR / "submission_final_safe.csv"
submission.to_csv(out_path, index=False)

print("Saved:", out_path)
print(submission.head())
print("\nRow sums:")
print(submission[class_cols].sum(axis=1).head())