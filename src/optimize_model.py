"""
Converts the trained sklearn model to ONNX, then applies dynamic
int8 quantization, so we have three versions to compare in benchmark.py:

    1. models/model.joblib        (raw sklearn)
    2. models/model.onnx          (ONNX, fp32)
    3. models/model_quant.onnx    (ONNX, dynamically quantized int8)

This mirrors the real production step of taking a trained model and
preparing it for low-latency serving -- the same workflow used before
deploying to edge devices or high-throughput inference servers.

Usage:
    python src/optimize_model.py
"""

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

FEATURES = [
    "amount",
    "tx_hour",
    "tx_count_last_hour",
    "avg_amount_last_7d",
    "location_risk_score",
]


def export_onnx(model_path: str, onnx_path: str):
    model = joblib.load(model_path)
    initial_type = [("input", FloatTensorType([None, len(FEATURES)]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model -> {onnx_path}")


def quantize(onnx_path: str, preprocessed_path: str, quant_path: str):
    # Required pre-processing step (shape inference + optimization) before
    # dynamic quantization -- skipping it silently degrades accuracy.
    # skip_symbolic_shape: symbolic shape inference targets transformer-style
    # graphs and chokes on skl2onnx's ZipMap/Cast ops; plain ONNX shape
    # inference is sufficient for this small MLP.
    quant_pre_process(onnx_path, preprocessed_path, skip_symbolic_shape=True)
    quantize_dynamic(preprocessed_path, quant_path, weight_type=QuantType.QInt8)
    print(f"Exported quantized ONNX model -> {quant_path}")


if __name__ == "__main__":
    export_onnx("models/model.joblib", "models/model.onnx")
    quantize(
        "models/model.onnx",
        "models/model_preprocessed.onnx",
        "models/model_quant.onnx",
    )
