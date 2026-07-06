"""
Orchestrator: wires DetectionAgent -> ExplanationAgent -> DecisionAgent
into a LangGraph StateGraph.

Why a graph instead of just calling three functions in a row?
  - Conditional branching: normal transactions skip the (slower, paid)
    explanation step entirely -- only anomalies pay the LLM cost.
  - Each node is independently testable and swappable (e.g. replace
    DetectionAgent's model without touching the other two agents).
  - This is the same pattern used for production agent pipelines: an
    explicit state machine, not an implicit chain of prompts.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.agents.detection_agent import DetectionAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.decision_agent import DecisionAgent


class PipelineState(TypedDict):
    transaction: dict
    detection_result: dict
    explanation: str
    decision: dict


detection_agent = DetectionAgent()
explanation_agent = ExplanationAgent()
decision_agent = DecisionAgent()


def detect_node(state: PipelineState) -> PipelineState:
    state["detection_result"] = detection_agent.score(state["transaction"])
    return state


def explain_node(state: PipelineState) -> PipelineState:
    state["explanation"] = explanation_agent.explain(state["transaction"], state["detection_result"])
    return state


def decide_node(state: PipelineState) -> PipelineState:
    state["decision"] = decision_agent.decide(state["detection_result"])
    return state


def route_after_detection(state: PipelineState) -> str:
    # Conditional edge: only pay for an LLM explanation if something is
    # actually wrong. This is a meaningful cost/latency saving at scale.
    return "explain" if state["detection_result"]["is_anomaly"] else "decide"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("detect", detect_node)
    graph.add_node("explain", explain_node)
    graph.add_node("decide", decide_node)

    graph.set_entry_point("detect")
    graph.add_conditional_edges("detect", route_after_detection, {"explain": "explain", "decide": "decide"})
    graph.add_edge("explain", "decide")
    graph.add_edge("decide", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(transaction: dict) -> dict:
    graph = get_graph()
    result = graph.invoke({
        "transaction": transaction,
        "detection_result": {},
        "explanation": "",
        "decision": {},
    })
    return {
        "transaction_id": transaction.get("transaction_id"),
        "detection": result["detection_result"],
        "explanation": result.get("explanation") or "Not applicable: transaction was not flagged as anomalous.",
        "decision": result["decision"],
    }
