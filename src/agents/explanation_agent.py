"""
ExplanationAgent: takes the detection result + raw transaction features
and produces a human-readable explanation an ops/fraud analyst can act
on immediately, instead of a bare probability score.
"""

from src.agents.llm_client import call_llm

SYSTEM_PROMPT = (
    "You are a fraud/anomaly analyst assistant. Given transaction features "
    "and a model's anomaly score, explain in 2-3 concise sentences why the "
    "transaction looks suspicious (or normal), referencing the specific "
    "feature values. Be precise, not generic."
)


class ExplanationAgent:
    def explain(self, transaction: dict, detection_result: dict) -> str:
        if not detection_result["is_anomaly"]:
            return "No explanation needed: transaction was not flagged as anomalous."

        prompt = (
            f"Transaction {transaction.get('transaction_id')}:\n"
            f"- amount: {transaction['amount']:.2f}\n"
            f"- hour of day: {transaction['tx_hour']}\n"
            f"- transactions in last hour: {transaction['tx_count_last_hour']}\n"
            f"- average amount last 7 days: {transaction['avg_amount_last_7d']:.2f}\n"
            f"- location risk score (0-1): {transaction['location_risk_score']:.2f}\n"
            f"- model anomaly probability: {detection_result['anomaly_probability']}\n"
            f"- severity: {detection_result['severity']}\n\n"
            "Explain why this was flagged."
        )
        return call_llm(prompt, system=SYSTEM_PROMPT)
