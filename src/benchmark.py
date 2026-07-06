"""
Benchmarks three versions of the model on the same test data:
  1. Raw sklearn pipeline  (models/model.joblib)
  2. ONNX fp32             (models/model.onnx)
  3. ONNX int8 quantized   (models/model_quant.onnx)

Reports: p50/p95 latency per single-row prediction, throughput
(rows/sec at batch=256), accuracy, and on-disk size. This is the
before/after table you put in the README.

Usage:
    python src/benchmark.py
"""

import json
import time

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score

FEATURES = [
    "amount",
    "tx_hour",
    "tx_count_last_hour",
    "avg_amount_last_7d",
    "location_risk_score",
]


def load_test_data(path="data/transactions.csv", n=2000, seed=42):
    df = pd.read_csv(path).sample(n=n, random_state=seed).reset_index(drop=True)
    X = df[FEATURES].values.astype(np.float32)
    y = df["is_anomaly"].values
    return X, y


def bench_sklearn(model_path, X, y, n_single=500):
    model = joblib.load(model_path)

    # single-row latency
    times = []
    for i in range(n_single):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)

    # throughput (batch)
    t0 = time.perf_counter()
    preds = model.predict(X)
    batch_time = time.perf_counter() - t0

    import os

    size_kb = os.path.getsize(model_path) / 1024
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput_rows_per_sec": len(X) / batch_time,
        "accuracy": float(accuracy_score(y, preds)),
        "size_kb": round(size_kb, 1),
    }


def bench_onnx(model_path, X, y, n_single=500):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    times = []
    for i in range(n_single):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        sess.run(None, {input_name: row})
        times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    outputs = sess.run(None, {input_name: X})
    batch_time = time.perf_counter() - t0
    preds = outputs[0]

    import os

    size_kb = os.path.getsize(model_path) / 1024
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput_rows_per_sec": len(X) / batch_time,
        "accuracy": float(accuracy_score(y, preds)),
        "size_kb": round(size_kb, 1),
    }


if __name__ == "__main__":
    X, y = load_test_data()

    results = {
        "sklearn_fp32": bench_sklearn("models/model.joblib", X, y),
        "onnx_fp32": bench_onnx("models/model.onnx", X, y),
        "onnx_int8_quantized": bench_onnx("models/model_quant.onnx", X, y),
    }

    print(json.dumps(results, indent=2))

    with open("models/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print a README-ready markdown table
    print("\n| Version | p50 latency (ms) | p95 latency (ms) | Throughput (rows/sec) | Accuracy | Size (KB) |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        print(
            f"| {name} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | "
            f"{r['throughput_rows_per_sec']:.0f} | {r['accuracy']:.4f} | {r['size_kb']} |"
        )
