import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
team_venue = pd.read_csv(SUB_DIR / "submission_team_venue_v1.csv")

class_cols = ["A_small", "A_big", "B_small", "B_big"]

# Best public-tuned prior so far
best_prior = {
    "A_small": 0.198,
    "A_big": 0.240,
    "B_small": 0.184,
    "B_big": 0.378
}

# Blend match-specific team venue with best prior
# High prior weight because public leaderboard liked this prior
variants = {
    "team_venue_v2_prior70": 0.70,
    "team_venue_v3_prior80": 0.80,
    "team_venue_v4_prior90": 0.90
}

for name, prior_weight in variants.items():

    sub = sample.copy()

    for col in class_cols:
        sub[col] = (
            prior_weight * best_prior[col]
            + (1 - prior_weight) * team_venue[col]
        )

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
    print("Average probabilities:")
    print(sub[class_cols].mean())
    print("Unique rows:", sub[class_cols].drop_duplicates().shape[0])
    print()