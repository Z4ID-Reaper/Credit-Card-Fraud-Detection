"""
Training pipeline for the Credit Card Fraud Detection project.

Stages: load data -> clean -> scale (Time, Amount) -> handle class imbalance
        (undersampling of the majority class) -> train/test split ->
        train RandomForestClassifier -> evaluate -> save model + scaler + metrics.

Usage:
    python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Run `python data/generate_synthetic_data.py` first, "
            f"or place the real Kaggle creditcard.csv in the data/ folder."
        )
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def handle_imbalance(X, y, random_state=42):
    """Simple random undersampling of the majority (genuine) class."""
    df = X.copy()
    df["Class"] = y.values
    fraud = df[df["Class"] == 1]
    genuine = df[df["Class"] == 0]

    # Keep genuine transactions at most ~4x the number of fraud transactions
    n_genuine_keep = min(len(genuine), len(fraud) * 4 if len(fraud) > 0 else len(genuine))
    genuine_sampled = genuine.sample(n=n_genuine_keep, random_state=random_state)

    balanced = pd.concat([fraud, genuine_sampled]).sample(frac=1, random_state=random_state)
    y_balanced = balanced["Class"]
    X_balanced = balanced.drop(columns=["Class"])
    return X_balanced, y_balanced


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    print(f"Rows after cleaning: {len(df)}")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    feature_order = list(X.columns)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_order, f)

    # Scale Time and Amount (V1-V28 are already PCA-scaled in the real dataset)
    scaler = StandardScaler()
    scale_cols = [c for c in ["Time", "Amount"] if c in X.columns]
    X_scaled = X.copy()
    X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])

    print("Handling class imbalance (undersampling majority class)...")
    X_bal, y_bal = handle_imbalance(X_scaled, y)
    print(f"Balanced dataset size: {len(X_bal)} | Fraud ratio: {y_bal.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, zero_division=0))

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(scale_cols, os.path.join(MODEL_DIR, "scale_cols.pkl"))

    print(f"\nModel, scaler and metrics saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
