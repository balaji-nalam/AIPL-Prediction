import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "train_IPL.csv"

train = pd.read_csv(train_path)

print("Original shape:", train.shape)

# Remove exact duplicate rows
train = train.drop_duplicates()

# Convert date
train["Date"] = pd.to_datetime(train["Date"], errors="coerce")

# Normalize team names
team_map = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants"
}

team_cols = [
    "Bat First",
    "Bat Second",
    "toss_winner",
    "match_won_by"
]

for col in team_cols:
    if col in train.columns:
        train[col] = train[col].replace(team_map)

# Fill only safe text columns
safe_text_fill = {
    "Dismissal Method": "No Wicket",
    "Player Out": "None",
    "Extra Type": "None"
}

for col, value in safe_text_fill.items():
    if col in train.columns:
        train[col] = train[col].fillna(value)

# Sort chronologically
train = train.sort_values(
    ["Date", "Match ID", "Innings", "Over", "Ball"]
)

# Save clean data
out_path = DATA_DIR / "clean_train.csv"
train.to_csv(out_path, index=False)

print("Clean shape:", train.shape)
print("Saved:", out_path)

print("\nRemaining missing values:")
missing = train.isnull().sum()
print(missing[missing > 0])