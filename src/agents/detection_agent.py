"""
DetectionAgent: loads the optimized (quantized ONNX) model and scores
an incoming transaction. This is the "fast path" -- it must be cheap
enough to run on every single transaction in real time, which is why
we hand it the quantized model rather than the raw sklearn one.
"""

import numpy as np
import onnxruntime as ort

FEATURES = [
    "amount",
    "tx_hour",
    "tx_count_last_hour",
    "avg_amount_last_7d",
    "location_risk_score",
]


class DetectionAgent:
    def __init__(self, model_path: str = "models/model_quant.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def score(self, transaction: dict) -> dict:
        row = np.array([[transaction[f] for f in FEATURES]], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: row})
        label = int(outputs[0][0])
        # second output is per-class probabilities in most skl2onnx exports
        probs = outputs[1][0] if len(outputs) > 1 else None
        anomaly_prob = float(probs[1]) if probs is not None else float(label)

        return {
            "transaction_id": transaction.get("transaction_id", "unknown"),
            "is_anomaly": bool(label),
            "anomaly_probability": round(anomaly_prob, 4),
            "severity": self._severity(anomaly_prob),
        }

    @staticmethod
    def _severity(prob: float) -> str:
        if prob >= 0.85:
            return "high"
        if prob >= 0.5:
            return "medium"
        return "low"
