from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_normal_transaction():
    payload = {
        "transaction_id": "TEST_NORMAL",
        "amount": 20.0,
        "tx_hour": 14,
        "tx_count_last_hour": 1,
        "avg_amount_last_7d": 35.0,
        "location_risk_score": 0.1,
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"]["action"] == "allow"
    assert body["detection"]["is_anomaly"] is False


def test_analyze_anomalous_transaction():
    payload = {
        "transaction_id": "TEST_ANOMALY",
        "amount": 3000.0,
        "tx_hour": 3,
        "tx_count_last_hour": 12,
        "avg_amount_last_7d": 40.0,
        "location_risk_score": 0.95,
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection"]["is_anomaly"] is True
    assert body["decision"]["action"] in {"auto_block_and_escalate", "flag_for_manual_review"}


def test_analyze_missing_field_returns_422():
    resp = client.post("/analyze", json={"amount": 10.0})
    assert resp.status_code == 422
