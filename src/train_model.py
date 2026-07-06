"""
Trains a supervised anomaly detector (RandomForest) on transaction data
and logs the run (params, metrics, model artifact) with MLflow.

Why RandomForest instead of only IsolationForest?
- We have labels here (is_anomaly), so a supervised model gives us a
  calibrated probability score the agents can reason about later.
- RandomForest also converts cleanly to ONNX for the optimization step.

Usage:
    python src/train_model.py --data data/transactions.csv
"""

import argparse
import json
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURES = [
    "amount",
    "tx_hour",
    "tx_count_last_hour",
    "avg_amount_last_7d",
    "location_risk_score",
]
TARGET = "is_anomaly"


def main(data_path: str, n_estimators: int, max_depth: int, out_path: str):
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("ai-ops-copilot-anomaly-detection")
    with mlflow.start_run():
        mlflow.log_param("hidden_layer_sizes", "(64, 32)")
        mlflow.log_param("max_iter", 300)
        mlflow.log_param("train_rows", len(X_train))

        # MLP (small neural net) instead of a tree ensemble: trees don't have
        # weight matrices, so ONNX dynamic quantization has nothing to act on.
        # An MLP gives us real MatMul weights -> a genuine before/after
        # quantization benchmark in optimize_model.py / benchmark.py.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
            )),
        ])

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "train_time_sec": train_time,
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        joblib.dump(model, out_path)
        mlflow.log_artifact(out_path, artifact_path="model")

        print(json.dumps(metrics, indent=2))
        print(f"Model saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/transactions.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--out", type=str, default="models/model.joblib")
    args = parser.parse_args()
    main(args.data, args.n_estimators, args.max_depth, args.out)
