"""
Stage II - Machine Learning & Deep Learning
Baseline: RandomForestClassifier (justified: tabular + engineered rolling
features, robust to feature scale, gives feature_importances_ for XAI).
Deep Learning: Keras ANN (MLP) trained on the same feature set to predict
failure probability. If TensorFlow is unavailable in the runtime, we
automatically fall back to sklearn's MLPClassifier so the app still runs
end-to-end (clearly logged so it is never silently mistaken for the primary
predictive engine requirement -- both are still classic ML/DL, not an LLM).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

from data_pipeline import FEATURE_COLS, TARGET_COL, build_dataset

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as kmodels
    HAS_TF = True
except Exception:
    HAS_TF = False
    from sklearn.neural_network import MLPClassifier

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def train_baseline(train, val):
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, proba)
    joblib.dump(clf, f"{ARTIFACT_DIR}/baseline_rf.joblib")
    return clf, metrics


def build_ann(input_dim):
    model = kmodels.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def train_deep_model(train, val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_val = scaler.transform(val[FEATURE_COLS])
    y_train, y_val = train[TARGET_COL].values, val[TARGET_COL].values

    joblib.dump(scaler, f"{ARTIFACT_DIR}/dl_scaler.joblib")

    if HAS_TF:
        model = build_ann(X_train.shape[1])
        model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=30, batch_size=64, verbose=0,
            class_weight={0: 1.0, 1: (y_train == 0).sum() / max((y_train == 1).sum(), 1)},
        )
        model.save(f"{ARTIFACT_DIR}/dl_model.keras")
        proba = model.predict(X_val, verbose=0).ravel()
    else:
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, f"{ARTIFACT_DIR}/dl_model.joblib")
        proba = model.predict_proba(X_val)[:, 1]

    metrics = evaluate(y_val, proba)
    metrics["backend"] = "keras_ann" if HAS_TF else "sklearn_mlp_fallback"
    return model, scaler, metrics


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def error_analysis(clf, test_df, top_n=10):
    """Return the worst false-negatives (missed failures) and false-positives
    for the report / XAI discussion."""
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].values
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = test_df.copy()
    out["pred"] = pred
    out["proba"] = proba
    fn = out[(y_test == 1) & (pred == 0)].sort_values("proba").head(top_n)
    fp = out[(y_test == 0) & (pred == 1)].sort_values("proba", ascending=False).head(top_n)
    return fn, fp


def train_and_compare():
    feat, train, val, test = build_dataset()
    rf, rf_metrics = train_baseline(train, val)
    dl, scaler, dl_metrics = train_deep_model(train, val)

    comparison = {"baseline_rf": rf_metrics, "deep_learning": dl_metrics}
    with open(f"{ARTIFACT_DIR}/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    fn, fp = error_analysis(rf, test)
    fn.to_csv(f"{ARTIFACT_DIR}/false_negatives.csv", index=False)
    fp.to_csv(f"{ARTIFACT_DIR}/false_positives.csv", index=False)

    return rf, dl, scaler, comparison, (train, val, test, feat)


if __name__ == "__main__":
    rf, dl, scaler, comparison, splits = train_and_compare()
    print(json.dumps(comparison, indent=2, default=str))
