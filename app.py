"""
Stage IX - Web Application
AI Factory Intelligence Command Center — Streamlit UI wiring together every
stage: data/EDA, ML+DL, CV, NLP, RAG, multi-agent orchestration, XAI,
digital twin what-if simulation, human-in-the-loop approval, MLflow
evidence, and PDF report generation.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from data_pipeline import build_dataset, eda_summary, FEATURE_COLS, TARGET_COL
import models
import text_nlp
import vision
import rag
import xai
import digital_twin as dt
import agents
import mlops
import report

st.set_page_config(page_title="AI Factory Intelligence Command Center", layout="wide")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


@st.cache_data(show_spinner=False)
def _load_data():
    return build_dataset()


@st.cache_resource(show_spinner="Training baseline + deep learning models...")
def _get_models():
    feat, train, val, test = _load_data()
    rf, rf_metrics = models.train_baseline(train, val)
    dl, scaler, dl_metrics = models.train_deep_model(train, val)
    mlops.log_training_run("baseline_rf", {"n_estimators": 300, "max_depth": 10}, rf_metrics)
    mlops.log_training_run("deep_learning", {"backend": dl_metrics.get("backend")}, dl_metrics)
    return rf, dl, scaler, rf_metrics, dl_metrics


@st.cache_resource(show_spinner=False)
def _get_rag_index():
    return rag.RAGIndex()


feat, train, val, test = _load_data()
rf, dl, scaler, rf_metrics, dl_metrics = _get_models()
rag_index = _get_rag_index()

st.title("🏭 AI Factory Intelligence Command Center")
st.caption("Autonomous Manufacturing Intelligence & Digital Twin — integrated demo "
           "on the AI4I 2020 Predictive Maintenance dataset")

tabs = st.tabs([
    "1. Data & EDA", "2. ML vs DL", "3. Vision + NLP", "4. RAG Knowledge",
    "5. Multi-Agent Command Center", "6. Explainable AI", "7. Digital Twin",
    "8. Human-in-the-Loop & MLOps", "9. Report",
])

# ---------------------------------------------------------------- Tab 1: EDA
with tabs[0]:
    st.subheader("Data Engineering & EDA")
    summary = eda_summary(feat)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", summary["n_rows"])
    c2.metric("Overall failure rate", f"{summary['failure_rate']:.2%}")
    c3.metric("Train / Val / Test", f"{len(train)}/{len(val)}/{len(test)}")

    st.write("**Failure rate by machine type**")
    st.bar_chart(pd.Series(summary["by_type_failure_rate"]))

    st.write("**Failure mode counts (TWF/HDF/PWF/OSF/RNF)**")
    st.bar_chart(pd.Series(summary["failure_mode_counts"]))

    st.write("**Sensor distributions**")
    sensor_pick = st.selectbox("Sensor", ["Air temperature [K]", "Process temperature [K]",
                                           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
    st.line_chart(feat[sensor_pick].reset_index(drop=True))

    with st.expander("Preprocessing & assumptions (documented)"):
        st.markdown(data_pipeline_doc := (
            "- Source: AI4I 2020 Predictive Maintenance dataset (public).\n"
            "- Duplicates dropped by Product ID; sensor values clipped at 0.5/99.5 percentile "
            "(documented, not silently deleted).\n"
            "- Rolling mean/std (window=5) computed **per machine Type**, sorted by UDI as a "
            "chronological proxy — simulates a streaming sensor setting.\n"
            "- Stratified train/val/test split (70/15/15) on the target; rolling features are "
            "causal (backward-looking) so no leakage across the split.\n"
        ))
    with st.expander("Raw sample"):
        st.dataframe(feat.head(20))

# ---------------------------------------------------------- Tab 2: ML vs DL
with tabs[1]:
    st.subheader("Machine Learning & Deep Learning")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Baseline: Random Forest")
        st.json(rf_metrics)
    with c2:
        st.markdown(f"### Deep Learning ({dl_metrics.get('backend')})")
        st.json(dl_metrics)

    st.write("**Comparison (Precision / Recall / F1 / ROC-AUC)**")
    comp_df = pd.DataFrame({
        "RandomForest": {k: rf_metrics[k] for k in ["precision", "recall", "f1", "roc_auc"]},
        "DeepLearning": {k: dl_metrics[k] for k in ["precision", "recall", "f1", "roc_auc"]},
    }).T
    st.dataframe(comp_df)
    st.bar_chart(comp_df[["precision", "recall", "f1"]])

    st.markdown("### Error analysis")
    fn, fp = models.error_analysis(rf, test)
    st.write(f"Top missed failures (false negatives): {len(fn)}")
    st.dataframe(fn[["Product ID", "Type", "Tool wear [min]", "Torque [Nm]", "proba"]])
    st.write(f"Top false alarms (false positives): {len(fp)}")
    st.dataframe(fp[["Product ID", "Type", "Tool wear [min]", "Torque [Nm]", "proba"]])

# ------------------------------------------------------ Tab 3: Vision + NLP
with tabs[2]:
    st.subheader("Computer Vision — Defect Detection")
    st.caption("Training-free classical CV anomaly scorer (Otsu-threshold on high-frequency "
               "residual). For real MVTec-AD data, use `vision.train_cnn_on_mvtec()` "
               "(MobileNetV2 transfer learning) locally with TensorFlow.")
    up = st.file_uploader("Upload a component image (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = np.array(Image.open(up).convert("RGB"))
        result = vision.classical_defect_score(img)
        overlay = vision.overlay_mask(img, result["mask"])
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(overlay, caption="Flagged regions (evidence overlay)", use_container_width=True)
        st.metric("Severity (per SOP-105)", result["severity"])
        st.metric("Defect area", f"{result['defect_fraction']:.2%}")
        st.metric("Confidence", result["confidence"])
        st.session_state["last_vision_result"] = result
        st.session_state["last_vision_image"] = img
    else:
        st.info("Upload an image to run defect detection, or continue — vision is optional per record.")

    st.divider()
    st.subheader("NLP — Maintenance Note Analysis")
    n_notes = st.slider("Generate synthetic maintenance notes", 20, 300, 100)
    if st.button("Generate & classify notes"):
        notes = text_nlp.generate_notes(feat, n=n_notes)
        classified = text_nlp.classify_batch(notes)
        st.session_state["notes_df"] = classified
    if "notes_df" in st.session_state:
        cdf = st.session_state["notes_df"]
        st.metric("Urgency classification accuracy (vs. synthetic ground truth)",
                   f"{cdf['urgency_correct'].mean():.1%}")
        st.dataframe(cdf[["Product ID", "failure_mode", "urgency_pred", "component", "note"]].head(30))

# --------------------------------------------------------- Tab 4: RAG
with tabs[3]:
    st.subheader("Generative AI + RAG — SOP / Manual Knowledge Base")
    st.caption("Knowledge base: rag_knowledge/*.txt (SOP-101..105, SAFETY-001). "
               "Retrieval = TF-IDF cosine similarity; answer is grounded ONLY in retrieved evidence.")
    q = st.text_input("Ask the Knowledge Agent",
                       "What should I do if a machine shows an overstrain fault?")
    if st.button("Retrieve & answer"):
        result = rag.generate_answer(q, rag_index)
        st.markdown("**Answer (evidence-grounded):**")
        st.write(result["answer"])
        st.markdown("**Retrieved evidence:**")
        for e in result["evidence"]:
            st.caption(f"{e['doc']} — {e['section']} (similarity {e['score']:.2f})")

    with st.expander("Why RAG beats unsupported generation (demo)"):
        demo = rag.demo_retrieval_vs_unsupported()
        st.write("**Query:**", demo["query"])
        st.write("**Without retrieval:**", demo["unsupported_baseline"])
        st.write("**With RAG:**", demo["rag_answer"]["answer"])

# ---------------------------------------------- Tab 5: Multi-Agent Command Center
with tabs[4]:
    st.subheader("Multi-Agent Command Center")
    st.caption("Select a machine reading; Vision, Predictive Maintenance, Knowledge and "
               "Planning/Decision agents run in sequence, passing structured messages.")

    idx = st.number_input("Test-set row index", 0, len(test) - 1, 0)
    row = test.iloc[int(idx)]
    st.dataframe(row[["Product ID", "Type"] + FEATURE_COLS[:5]].to_frame().T)

    use_uploaded_image = st.checkbox("Use image uploaded in Vision tab (if any)", value=True)
    image_arg = st.session_state.get("last_vision_image") if use_uploaded_image else None
    query = st.text_input("Question for the Knowledge Agent", "What action should be taken for this alarm?")

    if st.button("▶ Run agent pipeline"):
        out = agents.run_pipeline(rf, row, image=image_arg, query=query, rag_index=rag_index)
        st.session_state["agent_out"] = out
        st.session_state["agent_row"] = row

    if "agent_out" in st.session_state:
        out = st.session_state["agent_out"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👁 Vision Agent")
            st.json(out["vision"].payload)
            st.markdown("#### 🔧 Predictive Maintenance Agent")
            st.json(out["predictive"].payload)
        with c2:
            st.markdown("#### 📚 Knowledge Agent")
            st.json(out["knowledge"].payload)
            st.markdown("#### 🧭 Planning / Decision Agent")
            plan = out["planning"].payload
            st.success(f"Recommended action: **{plan['recommended_action']}**  "
                       f"(status: {plan['status']})")
            st.write(plan["reasoning"])

# -------------------------------------------------------- Tab 6: XAI
with tabs[5]:
    st.subheader("Explainable AI")
    st.caption(f"Method available: {'SHAP' if xai.HAS_SHAP else 'Permutation / perturbation fallback (install `shap` for TreeExplainer)'}")

    gfi = xai.global_feature_importance(rf, val[FEATURE_COLS], val[TARGET_COL])
    st.write("**Global feature importance**")
    st.bar_chart(pd.Series(gfi["importance"][:10], index=gfi["features"][:10]))

    if "agent_row" in st.session_state:
        row = st.session_state["agent_row"]
        exp = xai.explain_instance(rf, row)
        st.write("**Local explanation for selected machine (from Agent tab)**")
        st.info(exp["narrative"])
        st.dataframe(pd.DataFrame(exp["top_contributors"]))
    else:
        st.info("Run the agent pipeline in Tab 5 first to see a local (per-instance) explanation.")

# ------------------------------------------------------- Tab 7: Digital Twin
with tabs[6]:
    st.subheader("Digital Twin & What-If Simulation")
    c1, c2, c3 = st.columns(3)
    wear = c1.slider("Tool wear (min)", 0, 260, 180)
    torque = c2.slider("Torque (Nm)", 10.0, 80.0, 58.0)
    rpm = c3.slider("Rotational speed (rpm)", 1000, 2500, 1350)
    risk = st.slider("Predicted failure probability (from ML model, or override)", 0.0, 1.0, 0.42)

    state = dt.TwinState(tool_wear_min=wear, torque_nm=torque, rpm=rpm, failure_probability=risk)
    scenarios = dt.run_all_scenarios(state)
    best = dt.recommend(scenarios)

    sc_df = pd.DataFrame([s.__dict__ for s in scenarios])
    st.dataframe(sc_df)
    st.bar_chart(sc_df.set_index("name")[["expected_cost"]])
    st.success(f"Digital twin recommendation: **{best.name}** (lowest expected cost among safe options)")
    st.session_state["twin_scenarios"] = scenarios
    st.session_state["twin_best"] = best

# ------------------------------------------------- Tab 8: HITL + MLOps
with tabs[7]:
    st.subheader("Human-in-the-Loop Approval")
    if "agent_out" in st.session_state:
        plan = st.session_state["agent_out"]["planning"].payload
        pid = st.session_state["agent_row"]["Product ID"]
        st.write(f"Machine **{pid}** — AI recommendation: **{plan['recommended_action']}**")
        decision = st.radio("Supervisor decision", ["APPROVE", "REJECT", "MODIFY"], horizontal=True)
        modified = ""
        if decision == "MODIFY":
            modified = st.selectbox("Modified action", [s["name"] for s in plan["all_scenarios"]])
        reason = st.text_area("Reason / notes", "")
        if st.button("Submit decision"):
            report.log_human_decision(pid, plan["recommended_action"], decision, modified, reason)
            st.session_state["human_decision"] = {
                "decision": decision, "modified_action": modified or "-", "reason": reason or "-"}
            st.success("Decision recorded to audit log.")
    else:
        st.info("Run the agent pipeline in Tab 5 first.")

    st.divider()
    st.subheader("MLOps — Experiment Tracking")
    st.caption(f"Backend: {mlops.backend_name()}")
    runs = mlops.load_local_runs()
    if runs:
        runs_df = pd.DataFrame([
            {"run_name": r["run_name"], **r["params"], **r["metrics"]} for r in runs
        ])
        st.dataframe(runs_df)
    else:
        st.info("No local-fallback runs logged yet (or MLflow backend is active — run `mlflow ui`).")

    st.subheader("Decision Audit Log")
    decisions = report.load_decisions()
    if decisions:
        st.dataframe(pd.DataFrame(decisions))

# --------------------------------------------------------- Tab 9: Report
with tabs[8]:
    st.subheader("Automated Incident / Decision Report")
    if "agent_out" in st.session_state and "human_decision" in st.session_state:
        row = st.session_state["agent_row"]
        out = st.session_state["agent_out"]
        ctx = {
            "product_id": row["Product ID"],
            "machine_reading": {k: float(row[k]) for k in
                                 ["Air temperature [K]", "Process temperature [K]",
                                  "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]},
            "predictive": out["predictive"].payload,
            "vision": out["vision"].payload,
            "knowledge": out["knowledge"].payload,
            "planning": out["planning"].payload,
            "human_decision": st.session_state["human_decision"],
        }
        if st.button("📄 Generate PDF report"):
            path = f"{ARTIFACT_DIR}/report_{row['Product ID']}.pdf"
            report.generate_pdf_report(path, ctx)
            with open(path, "rb") as f:
                st.download_button("⬇ Download report", f, file_name=os.path.basename(path),
                                    mime="application/pdf")
            st.success("Report generated.")
    else:
        st.info("Run the agent pipeline (Tab 5) and submit a human decision (Tab 8) first.")

st.divider()
st.caption("AI recommendations in this system are decision support only — every action requires "
           "explicit human approval (SAFETY-001). Built for the AI Factory 2.0 hackathon task.")
