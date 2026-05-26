import pandas as pd
from pathlib import Path
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

data_path = DATA_DIR / "features_train.csv"

df = pd.read_csv(data_path)

print("Loaded features:", df.shape)

# Convert date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Sort by date for time-based split
df = df.sort_values("Date")

# Target
target_col = "label"

# Features - no leakage columns
feature_cols = [
    "year",
    "month",
    "season",
    "Venue",
    "city",
    "team_a",
    "team_b",
    "toss_winner",
    "toss_decision"
]

X = df[feature_cols]
y = df[target_col]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", list(label_encoder.classes_))

# Time-based validation split
# Train on first 80%, validate on latest 20%
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
    "month"
]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

valid_probs = pipeline.predict_proba(X_valid)
valid_preds = pipeline.predict(X_valid)

loss = log_loss(y_valid, valid_probs)
acc = accuracy_score(y_valid, valid_preds)

print("\nValidation Log Loss:", loss)
print("Validation Accuracy:", acc)

# Save model and label encoder
model_path = MODEL_DIR / "baseline_model.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

joblib.dump(pipeline, model_path)
joblib.dump(label_encoder, encoder_path)

print("\nSaved model:", model_path)
print("Saved label encoder:", encoder_path)