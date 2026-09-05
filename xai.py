"""
Stage VI - Explainable AI

Uses SHAP if available for RandomForest explanations.
Handles different SHAP output formats across versions.

If SHAP is unavailable or produces an incompatible output,
the system automatically falls back to permutation importance
or the model's built-in feature_importances_.

The app therefore always has a working XAI explanation.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from data_pipeline import FEATURE_COLS, TARGET_COL


# ============================================================
# OPTIONAL SHAP
# ============================================================

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def global_feature_importance(clf, X: pd.DataFrame, y=None):

    # --------------------------------------------------------
    # Try SHAP first
    # --------------------------------------------------------
    if HAS_SHAP:

        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X)

            # Convert output safely to numpy
            if isinstance(shap_values, list):

                # Binary classification:
                # use positive/failure class
                if len(shap_values) > 1:
                    vals = np.asarray(shap_values[1])
                else:
                    vals = np.asarray(shap_values[0])

            else:
                vals = np.asarray(shap_values)

                # Newer SHAP versions may return:
                # (samples, features, classes)
                if vals.ndim == 3:
                    vals = vals[:, :, -1]

            # Handle unusual dimensions
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)

            # Calculate mean absolute SHAP value
            importance = np.abs(vals).mean(axis=0)

            # Force to 1D
            importance = np.asarray(importance).reshape(-1)

            # Make sure number of features matches
            if len(importance) == len(X.columns):

                method = "SHAP (TreeExplainer)"

            else:
                raise ValueError(
                    "SHAP output dimensions do not match feature count."
                )

        except Exception:
            # If SHAP fails, use permutation importance
            importance = None
            method = None

    else:
        importance = None
        method = None


    # --------------------------------------------------------
    # Fallback: Permutation Importance
    # --------------------------------------------------------

    if importance is None:

        if y is not None:

            try:
                result = permutation_importance(
                    clf,
                    X,
                    y,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )

                importance = np.asarray(
                    result.importances_mean
                ).reshape(-1)

                method = "Permutation importance"

            except Exception:
                importance = None

        else:
            importance = None


    # --------------------------------------------------------
    # Final fallback: RandomForest built-in importance
    # --------------------------------------------------------

    if importance is None:

        importance = np.asarray(
            clf.feature_importances_
        ).reshape(-1)

        method = "Impurity-based feature_importances_"


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    importance = np.asarray(importance).reshape(-1)

    if len(importance) != len(X.columns):
        raise ValueError(
            f"Feature importance length ({len(importance)}) "
            f"does not match feature count ({len(X.columns)})."
        )


    # --------------------------------------------------------
    # Sort features by importance
    # --------------------------------------------------------

    order = np.argsort(importance)[::-1]


    return {
        "method": method,

        "features": [
            X.columns[i]
            for i in order
        ],

        "importance": [
            float(importance[i])
            for i in order
        ],
    }


# ============================================================
# LOCAL / INSTANCE EXPLANATION
# ============================================================

def explain_instance(clf, x_row: pd.Series):
    """Local explanation for a single machine reading."""

    X = pd.DataFrame([x_row[FEATURE_COLS]])

    proba = float(clf.predict_proba(X)[0, 1])

    # ========================================================
    # SHAP explanation
    # ========================================================

    if HAS_SHAP:

        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X)

            # Convert every possible SHAP output format
            # into ONE numeric value per feature.
            if isinstance(shap_values, list):

                # Binary classifier -> failure class
                if len(shap_values) > 1:
                    vals = np.asarray(shap_values[1])
                else:
                    vals = np.asarray(shap_values[0])

            else:
                vals = np.asarray(shap_values)

                # New SHAP format:
                # (samples, features, classes)
                if vals.ndim == 3:
                    vals = vals[0, :, -1]

                # Standard format:
                # (samples, features)
                elif vals.ndim == 2:
                    vals = vals[0]

                # Already one value per feature
                elif vals.ndim == 1:
                    vals = vals

                else:
                    raise ValueError(
                        f"Unsupported SHAP shape: {vals.shape}"
                    )

            # IMPORTANT:
            # Force every feature contribution to be a scalar.
            vals = np.asarray(vals).reshape(-1)

            # Check feature count
            if len(vals) != len(FEATURE_COLS):
                raise ValueError(
                    f"SHAP returned {len(vals)} values "
                    f"but there are {len(FEATURE_COLS)} features."
                )

            contribs = [
                (
                    feature,
                    float(value)
                )
                for feature, value in zip(
                    FEATURE_COLS,
                    vals
                )
            ]

            contribs = sorted(
                contribs,
                key=lambda t: abs(float(t[1])),
                reverse=True
            )

            method = "SHAP"

        except Exception:

            # =================================================
            # SHAP failed -> perturbation fallback
            # =================================================

            contribs = []

            base = proba

            for feature in FEATURE_COLS:

                perturbed = X.copy()

                # Remove this feature's signal
                perturbed[feature] = 0

                new_proba = float(
                    clf.predict_proba(
                        perturbed
                    )[0, 1]
                )

                contribution = float(
                    base - new_proba
                )

                contribs.append(
                    (
                        feature,
                        contribution
                    )
                )

            contribs = sorted(
                contribs,
                key=lambda t: abs(float(t[1])),
                reverse=True
            )

            method = (
                "Leave-one-out perturbation "
                "(SHAP unavailable)"
            )

    # ========================================================
    # SHAP not installed -> perturbation fallback
    # ========================================================

    else:

        contribs = []

        base = proba

        for feature in FEATURE_COLS:

            perturbed = X.copy()

            perturbed[feature] = 0

            new_proba = float(
                clf.predict_proba(
                    perturbed
                )[0, 1]
            )

            contribution = float(
                base - new_proba
            )

            contribs.append(
                (
                    feature,
                    contribution
                )
            )

        contribs = sorted(
            contribs,
            key=lambda t: abs(float(t[1])),
            reverse=True
        )

        method = (
            "Leave-one-out perturbation "
            "(SHAP unavailable)"
        )

    # ========================================================
    # Top 5 contributors
    # ========================================================

    top = contribs[:5]

    narrative = human_readable_explanation(
        proba,
        top
    )

    return {
        "failure_probability": proba,

        "method": method,

        "top_contributors": [
            {
                "feature": feature,
                "contribution": float(contribution)
            }
            for feature, contribution in top
        ],

        "narrative": narrative,
    }

# ============================================================
# HUMAN READABLE EXPLANATION
# ============================================================

def human_readable_explanation(
    proba: float,
    top_contributors
) -> str:

    if proba >= 0.5:
        risk = "HIGH"

    elif proba >= 0.2:
        risk = "ELEVATED"

    else:
        risk = "LOW"


    drivers = ", ".join(
        f"{f} "
        f"({'+' if c > 0 else ''}{c:.3f})"
        for f, c in top_contributors[:3]
    )


    if not drivers:
        drivers = "no dominant feature identified"


    return (
        f"Predicted failure risk is {risk} "
        f"({proba:.1%}). "
        f"The strongest signals pushing this prediction "
        f"were: {drivers}. "
        f"Positive contribution values increase failure "
        f"risk; negative values decrease it."
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from data_pipeline import build_dataset
    from models import train_baseline

    feat, train, val, test = build_dataset()

    rf, _ = train_baseline(
        train,
        val
    )

    gfi = global_feature_importance(
        rf,
        val[FEATURE_COLS],
        val[TARGET_COL]
    )

    print(
        "XAI method:",
        gfi["method"]
    )

    print(
        "Top global features:",
        list(
            zip(
                gfi["features"][:5],
                gfi["importance"][:5]
            )
        )
    )

    row = test.iloc[0]

    explanation = explain_instance(
        rf,
        row
    )

    print(
        explanation["narrative"]
    )