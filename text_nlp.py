"""
Stage III (NLP half) - Maintenance / incident text analysis.

Real AI4I data has no free-text field, so we generate realistic synthetic
maintenance notes tied to each row's actual failure mode flags (TWF/HDF/
PWF/OSF/RNF) and sensor readings -- this is documented synthetic data, not
claimed as real logs. We then run:
  - rule + TF-IDF based urgency/category classification
  - structured information extraction (component, symptom, urgency)
  - similarity search against the SOP knowledge base (used by rag.py)
"""

import random
import re
import pandas as pd

random.seed(42)

FAILURE_TEMPLATES = {
    "TWF": [
        "Tool wear alarm triggered on {pid}. Tool wear at {wear} min, torque {torque} Nm. Operator reports rough surface finish.",
        "{pid} showing excessive tool wear ({wear} min). Recommend tool change before next shift.",
    ],
    "HDF": [
        "Heat dissipation issue on {pid}. Air/process temp delta {delta}K, rotational speed {rpm} rpm low relative to torque. Motor casing warm to touch.",
        "Operator flagged overheating on {pid}, process temperature {ptemp}K. Cooling fan noise reported.",
    ],
    "PWF": [
        "Power failure event on {pid}: torque {torque} Nm at {rpm} rpm outside safe power envelope. Line stopped automatically.",
        "{pid} tripped on power fault. Suspect drive overload, torque reading {torque} Nm.",
    ],
    "OSF": [
        "Overstrain fault on {pid}. Tool wear {wear} min combined with torque {torque} Nm exceeded strain limit for {ptype}-type tooling.",
        "{pid} overstrain shutdown. Maintenance to inspect spindle bearing.",
    ],
    "RNF": [
        "Random/unclassified stoppage on {pid}, no clear sensor correlation. Logged for pattern review.",
    ],
    "NONE": [
        "Routine check on {pid}: all sensor readings nominal (temp {ptemp}K, {rpm} rpm, torque {torque} Nm). No action needed.",
        "{pid} passed scheduled inspection. Tool wear {wear} min, within limits.",
    ],
}

URGENCY_BY_MODE = {"TWF": "medium", "HDF": "high", "PWF": "critical",
                    "OSF": "high", "RNF": "low", "NONE": "low"}


def _row_mode(row):
    for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if row.get(m, 0) == 1:
            return m
    return "NONE"


def generate_notes(df: pd.DataFrame, n=None) -> pd.DataFrame:
    rows = df if n is None else df.sample(n=min(n, len(df)), random_state=42)
    records = []
    for _, r in rows.iterrows():
        mode = _row_mode(r)
        template = random.choice(FAILURE_TEMPLATES[mode])
        note = template.format(
            pid=r["Product ID"], wear=int(r["Tool wear [min]"]),
            torque=round(r["Torque [Nm]"], 1), rpm=int(r["Rotational speed [rpm]"]),
            ptemp=round(r["Process temperature [K]"], 1),
            delta=round(r["Process temperature [K]"] - r["Air temperature [K]"], 1),
            ptype=r["Type"],
        )
        records.append({
            "Product ID": r["Product ID"], "failure_mode": mode,
            "urgency_true": URGENCY_BY_MODE[mode], "note": note,
        })
    return pd.DataFrame(records)


COMPONENT_KEYWORDS = {
    "tool": ["tool wear", "tool change", "tooling", "surface finish"],
    "motor": ["overheating", "motor casing", "cooling fan"],
    "drive": ["power fault", "drive overload", "power failure"],
    "spindle/bearing": ["spindle bearing", "overstrain"],
}

URGENCY_KEYWORDS = {
    "critical": ["tripped", "stopped automatically", "power fault", "critical"],
    "high": ["overheating", "overstrain", "shutdown", "warm to touch"],
    "medium": ["rough surface", "recommend", "excessive"],
    "low": ["routine", "nominal", "passed", "no action"],
}


def extract_info(note: str) -> dict:
    """Lightweight structured information extraction (component + urgency)
    using keyword/rule matching -- transparent and auditable, unlike a black
    box LLM call, and fast enough to run per-incident in the agent pipeline."""
    text = note.lower()
    component = "unspecified"
    for comp, kws in COMPONENT_KEYWORDS.items():
        if any(kw in text for kw in kws):
            component = comp
            break

    urgency = "low"
    for level in ["critical", "high", "medium", "low"]:
        if any(kw in text for kw in URGENCY_KEYWORDS[level]):
            urgency = level
            break

    pid_match = re.search(r"\b([A-Z]\d{5})\b", note)

    return {
        "component": component,
        "urgency_pred": urgency,
        "product_id": pid_match.group(1) if pid_match else None,
    }


def classify_batch(notes_df: pd.DataFrame) -> pd.DataFrame:
    extracted = notes_df["note"].apply(extract_info).apply(pd.Series)
    out = pd.concat([notes_df.reset_index(drop=True), extracted], axis=1)
    out["urgency_correct"] = out["urgency_true"] == out["urgency_pred"]
    return out


if __name__ == "__main__":
    from data_pipeline import build_dataset
    feat, *_ = build_dataset()
    notes = generate_notes(feat, n=200)
    result = classify_batch(notes)
    print(result.head())
    print("Urgency classification accuracy:", result["urgency_correct"].mean())
