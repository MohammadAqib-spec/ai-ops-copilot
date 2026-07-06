# AI Ops Copilot

Real-time transaction anomaly detection with a multi-agent explanation & decision
pipeline — optimized for low-latency inference and served as a production-style API.

A transaction comes in → a quantized neural net scores it in under a millisecond →
if (and only if) it's flagged, an LLM agent explains *why* in plain English → a
deterministic decision agent turns that into an action (allow / review / block) →
everything is logged, tested, containerized, and monitored for data drift.

## Why this project exists

Most "AI portfolio projects" are a single notebook that calls an LLM once. This one
is built the way these systems actually get built in industry: a model that's been
optimized for inference cost, an agent pipeline with explicit control flow (not just
a prompt chain), a real API with tests and CI, and monitoring for when it degrades.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              FastAPI  (/analyze)             │
                    └───────────────────────┬───────────────────────┘
                                            │
                                  LangGraph orchestrator
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                                               │
          ┌───────────────────┐                                     │
          │  DetectionAgent    │  quantized ONNX model (int8)        │
          │  (always runs)     │  — the "fast path", <1ms/row        │
          └─────────┬─────────┘                                     │
                    │  is_anomaly?                                  │
              ┌─────┴─────┐                                         │
              │ no        │ yes                                     │
              ▼           ▼                                         │
        ┌──────────┐  ┌────────────────────┐                        │
        │  skip     │  │ ExplanationAgent    │  LLM call (Groq /     │
        │           │  │ (only if flagged)   │  Gemini free tier,    │
        │           │  │                     │  offline fallback)    │
        └────┬─────┘  └─────────┬───────────┘                        │
             │                  │                                    │
             └─────────┬────────┘                                    │
                       ▼                                              │
              ┌────────────────────┐                                  │
              │  DecisionAgent      │  rule-based, deterministic       │
              │  (always runs)      │  allow / review / block          │
              └─────────┬──────────┘                                  │
                       │                                              │
                       └──────────────────────────────────────────────┘
```

**Design decisions worth defending in an interview:**

- **Why skip the LLM call for normal transactions?** Cost and latency. If 98% of
  traffic is normal, calling an LLM on every row multiplies your inference bill for
  no benefit — the conditional edge in the graph only invokes the LLM on flagged
  transactions.
- **Why a quantized model for detection but a full LLM only downstream?** The
  detection path runs on *every* transaction and must be cheap; the explanation path
  runs on a tiny fraction and can afford a heavier call.
- **Why is the decision step rule-based, not another LLM call?** Auditability. An
  automated block/escalate action needs a deterministic, explainable trigger — you
  don't want an LLM's sampling variance deciding whether to freeze someone's account.

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Model | scikit-learn MLP | Small neural net — real weight matrices to optimize |
| Experiment tracking | MLflow | Versioned params/metrics per training run |
| Inference optimization | ONNX Runtime + dynamic int8 quantization | Production-style latency/size reduction |
| Agent orchestration | LangGraph | Explicit state graph with conditional branching |
| LLM | Groq / Gemini (free tier) via OpenAI-compatible API | Swappable, zero-cost to run |
| API | FastAPI | Async, typed, auto-documented |
| Testing | pytest | API-level integration tests |
| Containerization | Docker | Self-contained, reproducible image |
| CI/CD | GitHub Actions | Lint, train, benchmark, test, build on every push |
| Monitoring | KS-test drift detector (Evidently-ready) | Catches silent model degradation |

## Benchmark: why the optimization step matters

Run `python src/benchmark.py` yourself — this table is generated, not hand-typed:

| Version | p50 latency (ms) | p95 latency (ms) | Throughput (rows/sec) | Size (KB) |
|---|---|---|---|---|
| sklearn (raw) | 0.216 | 0.320 | ~1.0M | 87.3 |
| ONNX (fp32) | 0.011 | 0.018 | ~1.7M | 11.2 |
| **ONNX int8 (quantized)** | 0.014 | 0.024 | **~2.5M** | **6.6** |

> Numbers from a synthetic 5-feature MLP on CPU — the point isn't the absolute
> numbers, it's the *methodology*: always benchmark before/after an optimization,
> on the same data, with the same harness. Swap in a bigger model or real GPU and
> the relative story (ONNX ≫ raw sklearn, quantized ≫ fp32 on size) holds.

## Quickstart (everything free, no API key required)

```bash
git clone <your-repo-url>
cd ai-ops-copilot
pip install -r requirements.txt

# 1. Generate data (swap for a real dataset later, see data/generate_data.py)
python data/generate_data.py --rows 20000 --anomaly_rate 0.02

# 2. Train the model (tracked with MLflow)
python src/train_model.py
mlflow ui   # optional: view experiment runs at http://localhost:5000

# 3. Optimize: export to ONNX + quantize
python src/optimize_model.py

# 4. Benchmark the three versions
python src/benchmark.py

# 5. Run tests
pytest tests/ -v

# 6. Serve the API
uvicorn src.api.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive Swagger UI
```

### Try it

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 2500, "tx_hour": 3, "tx_count_last_hour": 10,
    "avg_amount_last_7d": 40, "location_risk_score": 0.95
  }'
```

### Optional: real LLM explanations

Copy `.env.example` to `.env`, set `LLM_PROVIDER=groq` and `LLM_API_KEY=<your free
key from console.groq.com>`. Without this, the pipeline still runs end-to-end using
a deterministic offline explanation — nothing breaks, nothing costs money.

### Monitoring for drift

```bash
python src/monitoring/drift_monitor.py --reference data/transactions.csv --current data/transactions.csv
```

Runs a Kolmogorov-Smirnov test per feature and flags dataset-level drift — the same
signal that would tell you in production that your model needs retraining.

## Free deployment options

- **API**: [Render](https://render.com) or [Railway](https://railway.app) free tier
  — connect the GitHub repo, it builds the Dockerfile automatically.
- **Demo UI**: wrap `/analyze` in a small [Streamlit](https://streamlit.io) app and
  deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (free) or
  [HuggingFace Spaces](https://huggingface.co/spaces) (free).
- **CI**: GitHub Actions free tier (already configured in `.github/workflows/ci.yml`).

## Swapping in a real dataset

Replace `data/transactions.csv` with the
[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(or any labeled tabular anomaly dataset) with the same column names, or edit
`FEATURES` in `src/train_model.py`, `src/optimize_model.py`, and
`src/agents/detection_agent.py` to match. Real data will give more realistic
(noisier, non-perfect) metrics than the clean synthetic data used here — expected,
and worth mentioning in an interview as evidence you understand the difference
between a clean-data demo and production data.

## Project structure

```
ai-ops-copilot/
├── data/generate_data.py          # synthetic data generator
├── src/
│   ├── train_model.py             # MLP training + MLflow tracking
│   ├── optimize_model.py          # ONNX export + int8 quantization
│   ├── benchmark.py                # latency/throughput/accuracy comparison
│   ├── agents/
│   │   ├── llm_client.py          # pluggable free-tier LLM client
│   │   ├── detection_agent.py     # fast-path anomaly scoring
│   │   ├── explanation_agent.py   # LLM-based explanation
│   │   ├── decision_agent.py      # rule-based action decision
│   │   └── orchestrator.py        # LangGraph state machine
│   ├── api/main.py                 # FastAPI service
│   └── monitoring/drift_monitor.py # KS-test drift detection
├── tests/test_api.py
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## What I'd do next with more time

- Swap the synthetic dataset for the real Kaggle fraud dataset and re-benchmark
- Add a feature store (e.g. Feast) so features are computed identically in
  training and serving
- Load-test the API with Locust and document p95 latency under concurrent load
- Add TensorRT conversion for GPU inference and compare against ONNX Runtime CPU
- Replace the KS-test drift monitor with Evidently's full drift report + a
  scheduled job (GitHub Actions cron or Airflow) that runs it daily
