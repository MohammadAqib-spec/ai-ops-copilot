"""
DecisionAgent: the final step in the pipeline. Converts (detection +
explanation) into a concrete action. This is deliberately rule-based
rather than another LLM call -- in a real ops system you want the
action-taking step to be deterministic and auditable, not subject to
LLM variance. The LLM informs humans; the rules take the action.
"""


class DecisionAgent:
    ACTIONS = {
        "high": "auto_block_and_escalate",
        "medium": "flag_for_manual_review",
        "low": "log_only",
    }

    def decide(self, detection_result: dict) -> dict:
        if not detection_result["is_anomaly"]:
            action = "allow"
        else:
            action = self.ACTIONS[detection_result["severity"]]

        return {
            "transaction_id": detection_result["transaction_id"],
            "action": action,
            "requires_human_review": action == "flag_for_manual_review",
        }
