"""
Generates a synthetic dataset that mimics the structure of the Kaggle
"Credit Card Fraud Detection" dataset (Time, V1-V28, Amount, Class).

Why this exists
----------------
The real Kaggle dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud) requires
a Kaggle account and manual download - it can't be fetched automatically here.
This script creates a synthetic dataset with the SAME columns and a similar
(highly imbalanced) fraud ratio, so you can run the full pipeline immediately.

For your real project / report, replace data/creditcard.csv with the actual
Kaggle CSV (same column names), then just re-run train_model.py -- everything
else works unchanged.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SAMPLES = 20000
FRAUD_RATIO = 0.0025  # ~0.25%, similar order of magnitude to the real dataset

n_fraud = int(N_SAMPLES * FRAUD_RATIO)
n_genuine = N_SAMPLES - n_fraud

# Genuine transactions: PCA-like features centered around 0
genuine = pd.DataFrame(
    np.random.normal(loc=0.0, scale=1.0, size=(n_genuine, 28)),
    columns=[f"V{i}" for i in range(1, 29)],
)
genuine["Time"] = np.sort(np.random.uniform(0, 172800, n_genuine))  # 2 days, in seconds
genuine["Amount"] = np.round(np.random.exponential(scale=60, size=n_genuine), 2)
genuine["Class"] = 0

# Fraudulent transactions: shifted / more spread out distribution
fraud = pd.DataFrame(
    np.random.normal(loc=2.5, scale=2.5, size=(n_fraud, 28)),
    columns=[f"V{i}" for i in range(1, 29)],
)
fraud["Time"] = np.sort(np.random.uniform(0, 172800, n_fraud))
fraud["Amount"] = np.round(np.random.exponential(scale=250, size=n_fraud), 2)
fraud["Class"] = 1

df = pd.concat([genuine, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Reorder columns to match the real Kaggle dataset: Time, V1..V28, Amount, Class
cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
df = df[cols]

out_path = os.path.join(os.path.dirname(__file__), "creditcard.csv")
df.to_csv(out_path, index=False)

print(f"Synthetic dataset written to: {out_path}")
print(f"Total rows: {len(df)} | Fraud rows: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
