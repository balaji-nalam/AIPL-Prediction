import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

SUB_DIR.mkdir(exist_ok=True)

features_path = DATA_DIR / "features_train.csv"
sample_path = DATA_DIR / "sample_submission.csv"

train = pd.read_csv(features_path)
sample = pd.read_csv(sample_path)

print("Training rows:", train.shape)
print("Sample submission rows:", sample.shape)

# Required competition columns
class_cols = ["A_small", "A_big", "B_small", "B_big"]

# Calculate class probabilities from training labels
label_counts = train["label"].value_counts(normalize=True)

print("\nHistorical class probabilities:")
print(label_counts)

# Fill sample submission with historical priors
submission = sample.copy()

for col in class_cols:
    submission[col] = label_counts.get(col, 0.25)

# Normalize rows to sum exactly 1
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

out_path = SUB_DIR / "submission_class_prior.csv"
submission.to_csv(out_path, index=False)

print("\nSaved submission:", out_path)
print(submission.head())
print("\nRow sums check:")
print(submission[class_cols].sum(axis=1).head())