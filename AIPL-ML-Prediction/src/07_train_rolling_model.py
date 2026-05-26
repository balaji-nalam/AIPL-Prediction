import pandas as pd
from pathlib import Path
import joblib

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

data_path = DATA_DIR / "features_rolling_train.csv"

df = pd.read_csv(data_path)

print("Loaded rolling features:", df.shape)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

target_col = "label"

feature_cols = [
    "year",
    "month",
    "season",
    "Venue",
    "city",
    "team_a",
    "team_b",
    "toss_winner",
    "toss_decision",
    "team_a_recent_win_rate",
    "team_b_recent_win_rate",
    "team_a_recent_big_rate",
    "team_b_recent_big_rate",
    "venue_a_win_rate",
    "venue_big_rate",
    "h2h_team_a_win_rate"
]

X = df[feature_cols]
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", list(label_encoder.classes_))

# Time-based split: latest 20% validation
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_valid = X.iloc[split_index:]

y_train = y_encoded[:split_index]
y_valid = y_encoded[split_index:]

print("Train rows:", len(X_train))
print("Valid rows:", len(X_valid))

categorical_features = [
    "season",
    "Venue",
    "city",
    "team_a",
    "team_b",
    "toss_winner",
    "toss_decision"
]

numeric_features = [
    "year",
    "month",
    "team_a_recent_win_rate",
    "team_b_recent_win_rate",
    "team_a_recent_big_rate",
    "team_b_recent_big_rate",
    "venue_a_win_rate",
    "venue_big_rate",
    "h2h_team_a_win_rate"
]

# Make OneHotEncoder dense for HistGradientBoosting
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            encoder,
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

gb = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.02,
                max_leaf_nodes=10,
                l2_regularization=2.0,
                random_state=42
            )
        )
    ]
)

print("\nTraining Gradient Boosting...")
gb.fit(X_train, y_train)

gb_probs = gb.predict_proba(X_valid)
gb_preds = gb.predict(X_valid)

gb_loss = log_loss(y_valid, gb_probs)
gb_acc = accuracy_score(y_valid, gb_preds)

print("GB Log Loss:", gb_loss)
print("GB Accuracy:", gb_acc)

# Class prior baseline on validation
train_labels = pd.Series(y_train)
prior_probs = train_labels.value_counts(normalize=True).sort_index()

prior_matrix = []
for _ in range(len(y_valid)):
    prior_matrix.append([
        prior_probs.get(i, 0.25)
        for i in range(len(label_encoder.classes_))
    ])

prior_loss = log_loss(y_valid, prior_matrix)

print("Class Prior Validation Log Loss:", prior_loss)

# Blend model with class prior
# This protects against overconfident bad model probabilities
prior_df = pd.DataFrame(
    prior_matrix,
    columns=range(len(label_encoder.classes_))
)

blend_probs = 0.25 * gb_probs + 0.75 * prior_df.values
blend_loss = log_loss(y_valid, blend_probs)

print("Blended GB + Prior Log Loss:", blend_loss)

# Save model objects
joblib.dump(gb, MODEL_DIR / "rolling_gb_model.pkl")
joblib.dump(label_encoder, MODEL_DIR / "rolling_label_encoder.pkl")

# Save prior probabilities also
prior_save = {
    label_encoder.classes_[i]: prior_probs.get(i, 0.25)
    for i in range(len(label_encoder.classes_))
}

joblib.dump(prior_save, MODEL_DIR / "class_prior.pkl")

print("\nSaved models to:", MODEL_DIR)
print("Saved class prior:", prior_save)