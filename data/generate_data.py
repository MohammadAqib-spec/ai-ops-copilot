"""
Generates a synthetic transaction stream dataset with injected anomalies.

Why synthetic data?
- Runs instantly, no download, no API key needed, fully reproducible.
- Swap in a real dataset later (e.g. Kaggle "Credit Card Fraud Detection")
  by replacing the output of this script with a CSV that has the same
  column names: amount, tx_hour, tx_count_last_hour, avg_amount_last_7d,
  location_risk_score, is_anomaly.

Usage:
    python data/generate_data.py --rows 20000 --anomaly_rate 0.02
"""

import argparse
import numpy as np
import pandas as pd


def generate(rows: int, anomaly_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anomalies = int(rows * anomaly_rate)
    n_normal = rows - n_anomalies

    # --- normal transactions ---
    normal = pd.DataFrame({
        "amount": rng.gamma(shape=2.0, scale=40, size=n_normal),
        "tx_hour": rng.integers(0, 24, n_normal),
        "tx_count_last_hour": rng.poisson(2, n_normal),
        "avg_amount_last_7d": rng.gamma(shape=2.0, scale=45, size=n_normal),
        "location_risk_score": rng.beta(2, 8, n_normal),  # skewed low (low risk)
        "is_anomaly": 0,
    })

    # --- anomalous transactions (fraud-like patterns) ---
    anomalies = pd.DataFrame({
        "amount": rng.gamma(shape=5.0, scale=250, size=n_anomalies),       # much bigger
        "tx_hour": rng.choice([0, 1, 2, 3, 4], n_anomalies),                # odd hours
        "tx_count_last_hour": rng.poisson(9, n_anomalies),                  # rapid-fire
        "avg_amount_last_7d": rng.gamma(shape=2.0, scale=45, size=n_anomalies),
        "location_risk_score": rng.beta(8, 2, n_anomalies),                 # skewed high risk
        "is_anomaly": 1,
    })

    df = pd.concat([normal, anomalies], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    df["transaction_id"] = [f"TXN{100000+i}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--anomaly_rate", type=float, default=0.02)
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.anomaly_rate)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({df['is_anomaly'].sum()} anomalies) -> {args.out}")
