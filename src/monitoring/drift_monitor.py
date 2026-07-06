"""
Data drift monitor: compares a reference dataset (what the model was
trained on) against current production traffic, and reports which
features have drifted. In a real deployment this runs on a schedule
(e.g. daily) and alerts if drift crosses a threshold.

This uses simple statistical tests (no heavy dependency required),
so it runs anywhere. If you have `evidently` installed, swap in its
DataDriftPreset for a richer HTML report -- see the commented block.

Usage:
    python src/monitoring/drift_monitor.py --reference data/transactions.csv --current data/transactions.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = [
    "amount",
    "tx_hour",
    "tx_count_last_hour",
    "avg_amount_last_7d",
    "location_risk_score",
]


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test per feature: p < alpha means the current
    distribution has likely drifted from the reference distribution."""
    report = {}
    for feature in FEATURES:
        stat, p_value = stats.ks_2samp(reference[feature], current[feature])
        report[feature] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": bool(p_value < alpha),
        }
    n_drifted = sum(1 for r in report.values() if r["drifted"])
    report["_summary"] = {
        "n_features_drifted": n_drifted,
        "total_features": len(FEATURES),
        "dataset_drift_detected": n_drifted / len(FEATURES) > 0.5,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="data/transactions.csv")
    parser.add_argument("--current", type=str, default="data/transactions.csv")
    parser.add_argument("--out", type=str, default="models/drift_report.json")
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)
    report = compute_drift(ref, cur)

    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # --- Optional: richer HTML report with Evidently ---
    # from evidently.report import Report
    # from evidently.metric_preset import DataDriftPreset
    # evidently_report = Report(metrics=[DataDriftPreset()])
    # evidently_report.run(reference_data=ref[FEATURES], current_data=cur[FEATURES])
    # evidently_report.save_html("models/drift_report.html")
