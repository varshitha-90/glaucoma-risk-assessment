
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "papila_clinical_clean.csv"
OUTPUT_DIR = BASE_DIR / "ml_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clinical features only. Exclude patient ID, eye identifier and diagnosis.
FEATURES = [
    "Age",
    "Gender",
    "Dioptre_1",
    "Dioptre_2",
    "Astigmatism",
    "Phakic_Pseudophakic",
    "IOP_Pneumatic",
    "IOP_Perkins",
    "Pachymetry",
    "Axial_Length",
    "VF_MD",
]

TARGET = "Diagnosis_Code"

required = FEATURES + [TARGET, "Patient_ID"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert clinical measurements to numeric values
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET, "Patient_ID"]).copy()
df = df[df[TARGET].isin([0, 1, 2])].copy()
df[TARGET] = df[TARGET].astype(int)

X = df[FEATURES]
y = df[TARGET]
groups = df["Patient_ID"].astype(str)

if y.nunique() < 3:
    raise ValueError("The dataset must contain all three diagnosis classes.")

print("=" * 55)
print("PAPILA GLAUCOMA CLASSIFICATION")
print("=" * 55)
print(f"Records: {len(df)}")
print(f"Unique patients: {groups.nunique()}")
print("\nClass distribution:")
print(y.map({
    0: "Healthy",
    1: "Glaucoma",
    2: "Glaucoma Suspect"
}).value_counts())

# Split by patient, not by individual eye
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# Models share the same missing-value handling and evaluation process
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ),
    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    ),
}

# Keep preprocessing inside each pipeline to prevent data leakage
pipelines = {}

for name, classifier in models.items():
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if name in ["Logistic Regression", "SVM"]:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", classifier))
    pipelines[name] = Pipeline(steps)

# Compare models using grouped cross-validation on training patients
number_of_splits = min(4, groups_train.nunique())
cv = GroupKFold(n_splits=number_of_splits)

results = {}

print("\nGrouped cross-validation (training data only):")

for name, pipeline in pipelines.items():
    fold_scores = []

    for fit_idx, val_idx in cv.split(X_train, y_train, groups_train):
        model = clone(pipeline)
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])

        predictions = model.predict(X_train.iloc[val_idx])

        score = f1_score(
            y_train.iloc[val_idx],
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
        fold_scores.append(score)

    mean_score = float(np.mean(fold_scores))
    results[name] = mean_score
    print(f"{name}: Macro F1 = {mean_score:.4f}")

# Select using training cross-validation, not test performance
best_name = max(results, key=results.get)
best_model = clone(pipelines[best_name])
best_model.fit(X_train, y_train)

# Evaluate once on the held-out patients
predictions = best_model.predict(X_test)

test_metrics = {
    "selected_model": best_name,
    "cv_macro_f1": results[best_name],
    "test_accuracy": float(accuracy_score(y_test, predictions)),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, predictions)
    ),
    "test_macro_f1": float(
        f1_score(
            y_test,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    ),
    "classes": {
        "0": "Healthy",
        "1": "Glaucoma",
        "2": "Glaucoma Suspect"
    },
    "features": FEATURES,
    "warning": (
        "Educational research prototype only; not validated for "
        "clinical diagnosis. Results are based on a small dataset."
    )
}

print("\n" + "=" * 55)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 55)
print(f"Selected model: {best_name}")
print(f"Accuracy: {test_metrics['test_accuracy']:.4f}")
print(f"Balanced accuracy: {test_metrics['test_balanced_accuracy']:.4f}")
print(f"Macro F1: {test_metrics['test_macro_f1']:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=["Healthy", "Glaucoma", "Glaucoma Suspect"],
    zero_division=0
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# Save model, feature names and evaluation results
joblib.dump(best_model, OUTPUT_DIR / "glaucoma_model.joblib")

with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
    json.dump(test_metrics, file, indent=4)

with open(OUTPUT_DIR / "model_features.json", "w", encoding="utf-8") as file:
    json.dump(FEATURES, file, indent=4)

print("\nModel saved successfully!")
print(f"Model: {OUTPUT_DIR / 'glaucoma_model.joblib'}")
print(f"Metrics: {OUTPUT_DIR / 'model_metrics.json'}")
print("\nNext: integrate this model with Django.")
