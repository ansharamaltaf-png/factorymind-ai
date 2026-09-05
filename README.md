# 🏭 AI Factory Intelligence Command Center

Integrated AI Factory Intelligence System for the **AI Factory 2.0** hackathon
task (Advance track). Combines data engineering, ML + DL, computer vision,
NLP, RAG/GenAI, multi-agent orchestration, explainable AI, digital-twin
what-if simulation, human-in-the-loop approval, MLOps tracking, and an
automated PDF report — in one Streamlit app.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch trains the baseline + deep-learning models (cached after that
via `st.cache_resource`) — takes a few seconds to ~1 minute depending on
whether TensorFlow is installed.

## Datasets used

| Modality | Source | Notes |
|---|---|---|
| Tabular + Time-series | `data/ai4i2020.csv` — **AI4I 2020 Predictive Maintenance Dataset** (public, UCI/Kaggle) | Real sensor data: air/process temperature, rotational speed, torque, tool wear, failure labels (TWF/HDF/PWF/OSF/RNF). Rolling features simulate the time-series/streaming setting. |
| Image | User-uploaded, built to work with **MVTec-AD** (https://www.kaggle.com/datasets/ipythonx/mvtec-ad) | See "Vision module" below — MVTec-AD wasn't downloadable in the build sandbox (no internet), so a training-free classical-CV detector is wired into the live app, and a CNN transfer-learning training script (`vision.train_cnn_on_mvtec`) is included for you to run locally against the real dataset. |
| Text | Synthetic maintenance notes, generated in `text_nlp.py` from each row's real failure-mode flags | Documented as synthetic — not claimed to be real logs. |
| PDF/SOP knowledge base | Synthetic SOP/safety documents in `rag_knowledge/*.txt` | Stand-in for machine manuals/SOPs; same chunking/retrieval approach works on real PDFs (swap in `pypdf` text extraction — see `rag.py` docstring). |

## Architecture

```
Streamlit App (app.py)
 ├─ data_pipeline.py   Stage I  – cleaning, EDA, feature engineering, leak-safe split
 ├─ models.py           Stage II – RandomForest baseline + Keras ANN (or sklearn MLP fallback)
 ├─ vision.py            Stage III (CV) – classical anomaly scorer (+ optional MVTec CNN training path)
 ├─ text_nlp.py           Stage III (NLP) – synthetic notes + rule/keyword extraction & urgency classification
 ├─ rag.py                 Stage IV – TF-IDF retrieval over SOP knowledge base + grounded answer generation
 ├─ agents.py               Stage V – Vision / Predictive-Maintenance / Knowledge / Planning agents, structured messages
 ├─ xai.py                   Stage VI – SHAP (or permutation-importance fallback) global + local explanations
 ├─ digital_twin.py           Stage VII – 4-scenario what-if simulator (continue / stop / reduce load / reschedule)
 ├─ mlops.py                   Stage VIII (MLOps) – MLflow wrapper (local JSONL fallback if MLflow absent)
 └─ report.py                   Stage VIII (HITL) + IX – decision audit log + PDF report generator
```

## Why the fallbacks?

The build/sandbox environment used to write this project has **no internet
access and no TensorFlow/MLflow/SHAP/Streamlit installed**. Every optional
dependency (`tensorflow`, `shap`, `mlflow`, `anthropic`) is wrapped in a
`try/except` so:
- the app **always runs end-to-end** even on a minimal install,
- but installing the full `requirements.txt` upgrades each stage to the
  "real" tool the rubric asks for (Keras ANN instead of MLPClassifier, SHAP
  TreeExplainer instead of permutation importance, MLflow UI instead of a
  JSONL log).

Nothing here is hard-coded to pass the demo — the RandomForest, MLP/ANN,
TF-IDF retrieval, digital-twin cost model, and classical-CV detector all run
on the actual data/inputs given.

## Wiring in real MVTec-AD images

1. Download MVTec-AD locally, arrange as `class_name/{good,defect}/*.png`.
2. `from vision import train_cnn_on_mvtec; train_cnn_on_mvtec("path/to/mvtec")`
   — trains a MobileNetV2 transfer-learning classifier (requires TensorFlow).
3. Swap the call in `agents.VisionAgent.run` from `classical_defect_score`
   to your loaded CNN's `.predict`.

## Wiring in a real LLM for RAG phrasing

Set `ANTHROPIC_API_KEY` and call `rag.generate_answer(query, index,
use_llm=True)`. The retrieval step (which evidence gets used) is unchanged
either way — the LLM is only ever used to phrase retrieved evidence, never
as the primary predictive engine (per rubric rule).

## Human-in-the-loop & safety

Every AI recommendation is labeled `PENDING_HUMAN_APPROVAL` and cannot reach
the "report" stage without an explicit APPROVE / REJECT / MODIFY decision
(Tab 8), which is appended to `artifacts/human_decisions.csv` as an audit
trail. This mirrors SAFETY-001 in the SOP knowledge base: AI output here is
decision support, never autonomous authority.

## Continuous-learning loop (Grand Challenge, sketch)

`artifacts/human_decisions.csv` + `artifacts/mlflow_fallback_log.jsonl`
already capture (prediction → human decision → outcome). To close the loop:
append approved/rejected rows back into a labeled retraining set, re-run
`models.train_and_compare()`, and log the new run via `mlops.log_training_run`
so each retraining iteration is tracked as a new MLflow candidate model.
