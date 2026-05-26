import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "clean_train.csv"

train = pd.read_csv(train_path, low_memory=False)

print("Loaded clean data:", train.shape)

# Remove tie / no-result / abandoned matches
bad_result_mask = train["result_type"].isin(["tie", "no result"])

bad_match_ids = train.loc[bad_result_mask, "Match ID"].unique()

train = train[~train["Match ID"].isin(bad_match_ids)]

# Also remove matches where winner or batting teams are missing
valid_match_ids = (
    train.groupby("Match ID")
    .filter(
        lambda x:
        x["match_won_by"].notna().all()
        and x["Bat First"].notna().all()
        and x["Bat Second"].notna().all()
    )["Match ID"]
    .unique()
)

train = train[train["Match ID"].isin(valid_match_ids)]

print("After removing invalid matches:", train["Match ID"].nunique())

# Match-level table
matches = (
    train.groupby("Match ID")
    .first()
    .reset_index()
)

# Final innings scores
innings_totals = (
    train.groupby(["Match ID", "Innings"])["Innings Runs"]
    .max()
    .unstack()
    .reset_index()
)

innings_totals.columns = [
    "Match ID",
    "innings1_runs",
    "innings2_runs"
]

# Wickets lost by chasing side
innings2_wickets = (
    train[train["Innings"] == 2]
    .groupby("Match ID")["Innings Wickets"]
    .max()
    .reset_index()
)

innings2_wickets.columns = [
    "Match ID",
    "innings2_wickets"
]

# Merge
matches = matches.merge(
    innings_totals,
    on="Match ID",
    how="inner"
)

matches = matches.merge(
    innings2_wickets,
    on="Match ID",
    how="inner"
)

# Remove rows with missing required values
required_cols = [
    "match_won_by",
    "Bat First",
    "Bat Second",
    "innings1_runs",
    "innings2_runs",
    "innings2_wickets"
]

matches = matches.dropna(subset=required_cols)

def get_label(row):
    winner = row["match_won_by"]
    team_a = row["Bat First"]

    innings1 = row["innings1_runs"]
    innings2 = row["innings2_runs"]
    wickets_lost = row["innings2_wickets"]

    # Team A = Bat First for historical training matches
    if winner == team_a:
        run_margin = innings1 - innings2

        if run_margin <= 20:
            return "A_small"
        else:
            return "A_big"

    else:
        wickets_remaining = 10 - wickets_lost

        if wickets_remaining <= 5:
            return "B_small"
        else:
            return "B_big"

matches["label"] = matches.apply(get_label, axis=1)

# Keep useful match-level columns
keep_cols = [
    "Match ID",
    "Date",
    "season",
    "Venue",
    "city",
    "Bat First",
    "Bat Second",
    "toss_winner",
    "toss_decision",
    "match_won_by",
    "innings1_runs",
    "innings2_runs",
    "innings2_wickets",
    "label"
]

matches = matches[keep_cols]

out_path = DATA_DIR / "match_level_dataset.csv"

matches.to_csv(out_path, index=False)

print("\nLabel distribution:")
print(matches["label"].value_counts())

print("\nTotal labelable matches:", len(matches))
print("Saved:", out_path)