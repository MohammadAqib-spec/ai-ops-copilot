"""
FastAPI service exposing the multi-agent pipeline.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-ops-copilot")

app = FastAPI(
    title="AI Ops Copilot",
    description="Real-time transaction anomaly detection with a multi-agent explanation & decision pipeline.",
    version="1.0.0",
)


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float
    tx_hour: int = Field(ge=0, le=23)
    tx_count_last_hour: int = Field(ge=0)
    avg_amount_last_7d: float
    location_risk_score: float = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(transaction: Transaction):
    start = time.perf_counter()
    try:
        result = run_pipeline(transaction.model_dump())
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "tx=%s anomaly=%s action=%s latency_ms=%.2f",
        result["transaction_id"],
        result["detection"]["is_anomaly"],
        result["decision"]["action"],
        latency_ms,
    )
    result["latency_ms"] = round(latency_ms, 2)
    return result
