"""
Stage V - Agentic AI / Multi-Agent System
Four cooperating agents with separated responsibilities. Each agent returns
a structured dict; the Planning Agent consumes the OTHER agents' structured
outputs (not raw data) to build its recommendation -- this is the
"agents pass structured information" requirement, orchestrated here as a
simple deterministic pipeline (swap for LangGraph if you want a graph
orchestrator; this keeps the demo dependency-free and auditable).
"""

from dataclasses import dataclass, asdict
from typing import Optional

import xai
import vision
import rag
import digital_twin as dt
from data_pipeline import FEATURE_COLS


@dataclass
class AgentMessage:
    agent: str
    payload: dict


class VisionAgent:
    name = "Vision Agent"

    def run(self, image=None) -> AgentMessage:
        if image is None:
            return AgentMessage(self.name, {"available": False, "reason": "no image supplied"})
        result = vision.classical_defect_score(image)
        payload = {
            "available": True,
            "severity": result["severity"],
            "defect_fraction": round(result["defect_fraction"], 4),
            "confidence": result["confidence"],
            "is_defective": result["is_defective"],
        }
        return AgentMessage(self.name, payload)


class PredictiveMaintenanceAgent:
    name = "Predictive Maintenance Agent"

    def __init__(self, clf):
        self.clf = clf

    def run(self, machine_row) -> AgentMessage:
        explanation = xai.explain_instance(self.clf, machine_row)
        payload = {
            "failure_probability": round(explanation["failure_probability"], 4),
            "risk_band": (
                "critical" if explanation["failure_probability"] >= 0.5 else
                "elevated" if explanation["failure_probability"] >= 0.2 else "low"
            ),
            "top_contributors": explanation["top_contributors"],
            "xai_method": explanation["method"],
            "narrative": explanation["narrative"],
        }
        return AgentMessage(self.name, payload)


class KnowledgeAgent:
    name = "Knowledge Agent"

    def __init__(self, index: rag.RAGIndex):
        self.index = index

    def run(self, query: str) -> AgentMessage:
        result = rag.generate_answer(query, self.index)
        payload = {
            "answer": result["answer"],
            "sources": [f"{e['doc']} :: {e['section']}" for e in result["evidence"]],
        }
        return AgentMessage(self.name, payload)


class PlanningDecisionAgent:
    name = "Planning/Decision Agent"

    def run(self, vision_msg: AgentMessage, maint_msg: AgentMessage,
            knowledge_msg: AgentMessage, twin_state: dt.TwinState) -> AgentMessage:
        scenarios = dt.run_all_scenarios(twin_state)
        best = dt.recommend(scenarios)

        reasons = []
        if maint_msg.payload["risk_band"] in ("critical", "elevated"):
            reasons.append(f"Predictive model flags {maint_msg.payload['risk_band']} failure risk "
                            f"({maint_msg.payload['failure_probability']:.1%}).")
        if vision_msg.payload.get("available") and vision_msg.payload.get("is_defective"):
            reasons.append(f"Vision agent detected a {vision_msg.payload['severity']} defect "
                            f"(area {vision_msg.payload['defect_fraction']:.1%}).")
        if knowledge_msg.payload["sources"]:
            reasons.append(f"SOP guidance retrieved from: {', '.join(knowledge_msg.payload['sources'][:2])}.")

        recommendation = {
            "recommended_action": best.name,
            "expected_cost": best.expected_cost,
            "residual_risk": best.residual_risk,
            "downtime_hours": best.downtime_hours,
            "reasoning": reasons or ["No strong risk signals; proceeding with lowest-cost safe option."],
            "all_scenarios": [asdict(s) for s in scenarios],
            "status": "PENDING_HUMAN_APPROVAL",
        }
        return AgentMessage(self.name, recommendation)


def run_pipeline(clf, machine_row, image=None, query: Optional[str] = None,
                  rag_index: Optional[rag.RAGIndex] = None):
    """Orchestrates all four agents end to end and returns every structured
    message (for the Streamlit UI to display agent-by-agent)."""
    rag_index = rag_index or rag.RAGIndex()
    default_query = query or "What action should be taken for this alarm?"

    v_agent, m_agent = VisionAgent(), PredictiveMaintenanceAgent(clf)
    k_agent, p_agent = KnowledgeAgent(rag_index), PlanningDecisionAgent()

    v_msg = v_agent.run(image)
    m_msg = m_agent.run(machine_row)
    k_msg = k_agent.run(default_query)

    twin_state = dt.TwinState(
        tool_wear_min=float(machine_row["Tool wear [min]"]),
        torque_nm=float(machine_row["Torque [Nm]"]),
        rpm=float(machine_row["Rotational speed [rpm]"]),
        failure_probability=m_msg.payload["failure_probability"],
    )
    p_msg = p_agent.run(v_msg, m_msg, k_msg, twin_state)

    return {"vision": v_msg, "predictive": m_msg, "knowledge": k_msg, "planning": p_msg}


if __name__ == "__main__":
    from data_pipeline import build_dataset
    from models import train_baseline

    feat, train, val, test = build_dataset()
    clf, _ = train_baseline(train, val)
    row = test.iloc[0]
    out = run_pipeline(clf, row)
    for k, v in out.items():
        print(k, "->", v.payload)
