"""
Stage VIII (HITL half) + Stage IX (Automated Report)
- log_human_decision: records APPROVE / REJECT / MODIFY + reason to a CSV
  audit trail (the AI recommendation is never auto-executed).
- generate_pdf_report: builds a downloadable factory incident/decision PDF
  summarizing prediction, confidence, XAI, RAG evidence, digital-twin
  scenarios, the agent recommendation, and the human decision.
"""

import os
import csv
import time
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib import colors

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DECISION_LOG = f"{ARTIFACT_DIR}/human_decisions.csv"


def log_human_decision(product_id: str, recommended_action: str, decision: str,
                        modified_action: str = "", reason: str = ""):
    """decision in {APPROVE, REJECT, MODIFY}."""
    is_new = not os.path.exists(DECISION_LOG)
    with open(DECISION_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "product_id", "recommended_action",
                              "decision", "modified_action", "reason"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), product_id,
                          recommended_action, decision, modified_action, reason])


def load_decisions():
    if not os.path.exists(DECISION_LOG):
        return []
    with open(DECISION_LOG) as f:
        return list(csv.DictReader(f))


def generate_pdf_report(path: str, context: dict):
    """context expects keys: product_id, machine_reading (dict), predictive,
    vision, knowledge, planning (agent payload dicts), human_decision (dict),
    xai_narrative (str)."""
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph("AI Factory Intelligence — Incident &amp; Decision Report", h1))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(f"Product / Machine ID: {context.get('product_id','N/A')}", h2))

    mr = context.get("machine_reading", {})
    rows = [["Sensor", "Value"]] + [[k, f"{v:.2f}" if isinstance(v, float) else str(v)]
                                     for k, v in mr.items()]
    t = Table(rows, colWidths=[8 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    pred = context.get("predictive", {})
    story.append(Paragraph("Predictive Maintenance Agent", h2))
    story.append(Paragraph(
        f"Failure probability: {pred.get('failure_probability', 0):.1%} "
        f"(risk band: {pred.get('risk_band','N/A')})", body))
    story.append(Paragraph(f"Explainability ({pred.get('xai_method','N/A')}): "
                            f"{pred.get('narrative','')}", body))
    story.append(Spacer(1, 0.3 * cm))

    vis = context.get("vision", {})
    if vis.get("available"):
        story.append(Paragraph("Vision Agent", h2))
        story.append(Paragraph(
            f"Defect severity: {vis.get('severity')}, area: {vis.get('defect_fraction', 0):.1%}, "
            f"confidence: {vis.get('confidence')}", body))
        story.append(Spacer(1, 0.3 * cm))

    know = context.get("knowledge", {})
    story.append(Paragraph("Knowledge Agent (RAG)", h2))
    story.append(Paragraph(know.get("answer", "").replace("\n", "<br/>"), body))
    story.append(Paragraph("Sources: " + ", ".join(know.get("sources", [])), small))
    story.append(Spacer(1, 0.3 * cm))

    plan = context.get("planning", {})
    story.append(Paragraph("Planning / Decision Agent — Digital Twin What-If", h2))
    scen_rows = [["Scenario", "Produced Units", "Downtime (h)", "Expected Cost ($)", "Residual Risk"]]
    for s in plan.get("all_scenarios", []):
        scen_rows.append([s["name"], f"{s['produced_units']:.0f}", f"{s['downtime_hours']:.2f}",
                           f"{s['expected_cost']:.2f}", f"{s['residual_risk']:.1%}"])
    st = Table(scen_rows, colWidths=[5 * cm, 3 * cm, 2.5 * cm, 3 * cm, 2.5 * cm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"<b>AI Recommendation:</b> {plan.get('recommended_action','N/A')}", body))
    story.append(Paragraph("Reasoning: " + " ".join(plan.get("reasoning", [])), body))
    story.append(Spacer(1, 0.3 * cm))

    hd = context.get("human_decision", {})
    story.append(Paragraph("Human-in-the-Loop Decision", h2))
    story.append(Paragraph(
        f"Decision: <b>{hd.get('decision','PENDING')}</b>  |  "
        f"Modified action: {hd.get('modified_action','-')}  |  "
        f"Reason: {hd.get('reason','-')}", body))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Note: This AI recommendation is decision support only. It is not an "
        "autonomous operational directive; the recorded human decision above "
        "is authoritative per SAFETY-001.", small))

    doc.build(story)
    return path


if __name__ == "__main__":
    demo_ctx = {
        "product_id": "M14860",
        "machine_reading": {"Torque [Nm]": 58.2, "Tool wear [min]": 190, "Rotational speed [rpm]": 1350},
        "predictive": {"failure_probability": 0.42, "risk_band": "elevated",
                        "xai_method": "Permutation importance",
                        "narrative": "Risk driven mainly by high tool wear and torque."},
        "vision": {"available": True, "severity": "major", "defect_fraction": 0.045, "confidence": 0.63},
        "knowledge": {"answer": "Stop machine on OSF alarm per SOP-103.", "sources": ["sop_power_overstrain.txt :: Section 3"]},
        "planning": {"recommended_action": "Stop for maintenance", "reasoning": ["Elevated risk + major defect."],
                      "all_scenarios": [{"name": "Continue operation", "produced_units": 480, "downtime_hours": 1,
                                          "expected_cost": 900, "residual_risk": 0.3}]},
        "human_decision": {"decision": "APPROVE", "modified_action": "-", "reason": "Confirmed by supervisor"},
    }
    generate_pdf_report(f"{ARTIFACT_DIR}/demo_report.pdf", demo_ctx)
    log_human_decision("M14860", "Stop for maintenance", "APPROVE", reason="Confirmed by supervisor")
    print("Report + decision log written.")
