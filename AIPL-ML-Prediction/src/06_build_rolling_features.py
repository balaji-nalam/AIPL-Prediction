import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

match_path = DATA_DIR / "match_level_dataset.csv"

df = pd.read_csv(match_path)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values(["Date", "Match ID"]).reset_index(drop=True)

# Helper functions
def is_team_a_win(label):
    return label in ["A_small", "A_big"]

def is_team_b_win(label):
    return label in ["B_small", "B_big"]

def is_big_win(label):
    return label in ["A_big", "B_big"]

def team_won(label, side):
    if side == "A":
        return is_team_a_win(label)
    else:
        return is_team_b_win(label)

def team_big_win(label, side):
    if side == "A":
        return label == "A_big"
    else:
        return label == "B_big"

# History stores
team_history = defaultdict(lambda: deque(maxlen=10))
venue_history = defaultdict(lambda: deque(maxlen=20))
h2h_history = defaultdict(lambda: deque(maxlen=10))

rows = []

for _, row in df.iterrows():

    team_a = row["Bat First"]
    team_b = row["Bat Second"]
    venue = row["Venue"]
    label = row["label"]

    h2h_key = tuple(sorted([team_a, team_b]))

    # Team histories before current match
    hist_a = list(team_history[team_a])
    hist_b = list(team_history[team_b])

    venue_hist = list(venue_history[venue])
    h2h_hist = list(h2h_history[h2h_key])

    def safe_mean(values, default):
        return sum(values) / len(values) if len(values) > 0 else default

    # Historical class prior fallback
    global_a_win_default = 0.45
    global_big_default = 0.55

    team_a_recent_win_rate = safe_mean(
        [x["won"] for x in hist_a],
        global_a_win_default
    )

    team_b_recent_win_rate = safe_mean(
        [x["won"] for x in hist_b],
        global_a_win_default
    )

    team_a_recent_big_rate = safe_mean(
        [x["big_win"] for x in hist_a],
        global_big_default
    )

    team_b_recent_big_rate = safe_mean(
        [x["big_win"] for x in hist_b],
        global_big_default
    )

    venue_a_win_rate = safe_mean(
        [x["a_win"] for x in venue_hist],
        0.45
    )

    venue_big_rate = safe_mean(
        [x["big"] for x in venue_hist],
        0.55
    )

    h2h_team_a_win_rate = safe_mean(
        [
            1 if x["winner"] == team_a else 0
            for x in h2h_hist
        ],
        0.50
    )

    feature_row = {
        "match_id": row["Match ID"],
        "Date": row["Date"],
        "year": row["Date"].year,
        "month": row["Date"].month,
        "season": row["season"],
        "Venue": venue,
        "city": row["city"],
        "team_a": team_a,
        "team_b": team_b,
        "toss_winner": row["toss_winner"],
        "toss_decision": row["toss_decision"],

        "team_a_recent_win_rate": team_a_recent_win_rate,
        "team_b_recent_win_rate": team_b_recent_win_rate,
        "team_a_recent_big_rate": team_a_recent_big_rate,
        "team_b_recent_big_rate": team_b_recent_big_rate,
        "venue_a_win_rate": venue_a_win_rate,
        "venue_big_rate": venue_big_rate,
        "h2h_team_a_win_rate": h2h_team_a_win_rate,

        "label": label
    }

    rows.append(feature_row)

    # Update histories AFTER feature creation to avoid leakage

    a_won = is_team_a_win(label)
    b_won = is_team_b_win(label)
    big = is_big_win(label)

    team_history[team_a].append({
        "won": 1 if a_won else 0,
        "big_win": 1 if label == "A_big" else 0
    })

    team_history[team_b].append({
        "won": 1 if b_won else 0,
        "big_win": 1 if label == "B_big" else 0
    })

    venue_history[venue].append({
        "a_win": 1 if a_won else 0,
        "big": 1 if big else 0
    })

    winner_team = team_a if a_won else team_b

    h2h_history[h2h_key].append({
        "winner": winner_team
    })

features = pd.DataFrame(rows)

# Fill categorical missing values
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

# Fill numeric missing values
num_cols = [
    "team_a_recent_win_rate",
    "team_b_recent_win_rate",
    "team_a_recent_big_rate",
    "team_b_recent_big_rate",
    "venue_a_win_rate",
    "venue_big_rate",
    "h2h_team_a_win_rate"
]

for col in num_cols:
    features[col] = features[col].fillna(features[col].mean())

print("Missing values:")
print(features.isnull().sum()[features.isnull().sum() > 0])

out_path = DATA_DIR / "features_rolling_train.csv"
features.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print("Shape:", features.shape)
print(features.head())