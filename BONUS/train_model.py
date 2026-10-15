"""
BONUS CHALLENGE — Step 3
Robotic Arm Joint Health Monitoring System
Model Training Script

Trains a Random Forest classifier on the synthetic joint health dataset.
Features : torque, temperature, vibration
Target   : label  (SAFE / MONITOR / SCHEDULE_MAINTENANCE / EMERGENCY_STOP)

Saves the trained model pipeline to: BONUS/joint_model.pkl

Run:
  python BONUS/train_model.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_FILE   = BASE_DIR / "joint_health_dataset.csv"
MODEL_FILE  = BASE_DIR / "joint_model.pkl"

# ── Load dataset ──────────────────────────────────────────────────────────────
print("=" * 58)
print("  MODEL TRAINING — Robotic Arm Joint Health")
print("=" * 58)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print(f"\n  Dataset loaded : {len(df)} records from '{DATA_FILE.name}'")
print(f"\n  Label counts:")
for lbl, cnt in df["label"].value_counts().items():
    print(f"    {lbl:<25} {cnt}")

# ── Features & target ─────────────────────────────────────────────────────────
FEATURES = ["torque", "temperature", "vibration"]
X = df[FEATURES]
y = df["label"]

# ── Train / test split (stratified) ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")

# ── Model pipeline ────────────────────────────────────────────────────────────
#    StandardScaler normalises features before the forest sees them —
#    not strictly needed for Random Forest but keeps the pipeline consistent
#    and compatible with future swaps to SVM/KNN.
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        class_weight="balanced",   # handles class imbalance fairly
        random_state=42,
        n_jobs=-1
    )
)

# ── Train ──────────────────────────────────────────────────────────────────────
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Test Accuracy : {accuracy * 100:.1f}%")
print()
print("  Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ── Cross-validation (5-fold) ─────────────────────────────────────────────────
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"  5-Fold CV Accuracy : {cv_scores.mean() * 100:.1f}% "
      f"(+/- {cv_scores.std() * 100:.1f}%)")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_FILE)
print(f"\n  Model saved to : {MODEL_FILE.resolve()}")
print("=" * 58)

# ── Quick sanity predict ──────────────────────────────────────────────────────
print("\n  Sanity Checks (manual predictions):")
samples = [
    {"label": "Expected SAFE",                 "values": [30.0, 55.0, 1.5]},
    {"label": "Expected MONITOR",              "values": [65.0, 75.0, 5.0]},
    {"label": "Expected SCHEDULE_MAINTENANCE", "values": [80.0, 87.0, 7.5]},
    {"label": "Expected EMERGENCY_STOP",       "values": [110.0, 105.0, 12.0]},
]

for s in samples:
    pred = model.predict([s["values"]])[0]
    match = "OK" if s["label"].split()[-1] == pred else "MISMATCH"
    print(f"    [{match}] {s['label']:<35} -> Predicted: {pred}")

print()
