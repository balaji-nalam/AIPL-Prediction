import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

match_path = DATA_DIR / "match_level_dataset.csv"

df = pd.read_csv(match_path)

print("Loaded match-level data:", df.shape)

# Convert date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract year/month
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

# Basic pre-match features only
feature_cols = [
    "Match ID",
    "Date",
    "year",
    "month",
    "season",
    "Venue",
    "city",
    "Bat First",
    "Bat Second",
    "toss_winner",
    "toss_decision",
    "label"
]

features = df[feature_cols].copy()

# Rename for clarity
features = features.rename(columns={
    "Match ID": "match_id",
    "Bat First": "team_a",
    "Bat Second": "team_b"
})

# Fill missing categorical values
cat_cols = [
    "season",
    "Venue",
    "city",
    "team_a",
    "team_b",
    "toss_winner",
    "toss_decision"
]

for col in cat_cols:
    features[col] = features[col].fillna("Unknown").astype(str)

# Drop rows with missing date/year/month
features = features.dropna(subset=["Date", "year", "month"])

# Final missing check
missing = features.isnull().sum()
print("\nMissing values:")
print(missing[missing > 0])

# Save
out_path = DATA_DIR / "features_train.csv"
features.to_csv(out_path, index=False)

print("\nFeature file saved:", out_path)
print("Final shape:", features.shape)
print("\nColumns:")
print(features.columns.tolist())