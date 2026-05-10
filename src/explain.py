# ============================================================
# explain.py — SHAP Explainability Engine
# ============================================================
# PURPOSE:
#   Makes the AI's decisions transparent using SHAP values.
#   Explains both the overall model behavior (global) and
#   individual patient predictions (local).
#
# WHAT IS SHAP?
#   SHAP = SHapley Additive exPlanations
#   Based on Shapley values from cooperative game theory.
#   For each prediction, SHAP assigns each feature a value
#   that represents its contribution to pushing the prediction
#   AWAY from the average baseline.
#
#   Example: For a patient predicted to have disease (confidence 80%):
#     + thal=2 contributed +15% (strong positive push)
#     + ca=2 contributed +12%
#     - thalach=162 contributed -5% (high max HR is protective)
#     ... etc.
#
# WHY SHAP OVER FEATURE IMPORTANCE?
#   Feature importance tells you "which features mattered overall"
#   SHAP tells you "HOW and HOW MUCH each feature mattered for THIS patient"
#   This is critical for medical explainability.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.utils import (
    RF_MODEL_PATH, XGB_MODEL_PATH,
    SHAP_SUMMARY_PATH, OUTPUTS_DIR,
    FEATURE_LABELS, setup_logger, save_figure
)

logger = setup_logger("explain")


def get_explainer(model, X_background: pd.DataFrame):
    """
    Creates a SHAP TreeExplainer for tree-based models.

    WHY TreeExplainer:
        Specifically designed for tree-based models (RF, XGBoost, LightGBM).
        Much faster than the generic KernelExplainer and provides exact
        SHAP values rather than approximations.

    X_background is used as the reference distribution — SHAP measures
    how much each feature shifts the prediction away from the average
    prediction on this background dataset.

    Args:
        model        : Trained RF or XGBoost model
        X_background : A sample of training data (used as baseline)

    Returns:
        SHAP TreeExplainer object
    """
    logger.info("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, data=X_background)
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Computes SHAP values for a dataset.

    Args:
        explainer: SHAP explainer object
        X        : Feature DataFrame to explain

    Returns:
        shap_values: Array of SHAP values, shape (n_samples, n_features)
                     For binary classification we take index [1] (positive class)
    """
    logger.info(f"Computing SHAP values for {len(X)} samples...")
    shap_values = explainer.shap_values(X)

    # For tree models with binary classification, shap_values may be a list
    # [values_for_class_0, values_for_class_1]
    # We always want class 1 (heart disease)
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> plt.Figure:
    """
    Global SHAP summary plot showing feature importance across all patients.

    The beeswarm plot shows:
      - X-axis: SHAP value (impact on model output)
      - Y-axis: Features sorted by importance
      - Color: Feature value (red=high, blue=low)
      - Each dot: One patient

    This gives a holistic view of which features drive predictions globally.

    Args:
        shap_values: 2D array of SHAP values
        X          : Feature DataFrame (for feature names + values)

    Returns:
        matplotlib Figure
    """
    logger.info("Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use human-readable labels
    X_labeled = X.rename(columns=FEATURE_LABELS)

    shap.summary_plot(
        shap_values,
        X_labeled,
        plot_type="dot",       # Beeswarm dot plot
        show=False,            # Don't display — we return the figure
        max_display=15,        # Show top 15 features
        color_bar_label="Feature Value"
    )

    plt.title("SHAP Feature Impact — Heart Disease Prediction",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    save_figure(fig, SHAP_SUMMARY_PATH)
    logger.info(f"SHAP summary saved to: {SHAP_SUMMARY_PATH}")
    return fig


def plot_shap_waterfall(
    explainer,
    X_single: pd.DataFrame,
    patient_label: str = "Patient"
) -> plt.Figure:
    """
    Local SHAP waterfall plot for a single patient prediction.

    The waterfall plot shows how each feature VALUE for THIS patient
    pushes the prediction up (red) or down (blue) from the base value.

    Base value = average model output across the training set.
    Final output = base + sum of all SHAP contributions.

    This is the KEY explainability tool for individual predictions.

    Args:
        explainer    : SHAP explainer
        X_single     : Single-row DataFrame (one patient)
        patient_label: Label for the plot title

    Returns:
        matplotlib Figure
    """
    logger.info(f"Generating waterfall plot for: {patient_label}")

    # Compute SHAP explanation object for this single row
    explanation = explainer(X_single)

    # Get values for the positive class (heart disease)
    if len(explanation.shape) == 3:
        # Multi-output: select class 1 (disease)
        exp_single = shap.Explanation(
            values=explanation.values[0, :, 1],
            base_values=explanation.base_values[0, 1],
            data=explanation.data[0],
            feature_names=list(X_single.columns),
        )
    else:
        exp_single = explanation[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(exp_single, max_display=13, show=False)
    plt.title(f"SHAP Explanation — {patient_label}",
              fontsize=13, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_shap_bar(shap_values: np.ndarray, feature_names: list) -> plt.Figure:
    """
    Simple bar chart of mean absolute SHAP values (global feature importance).

    Easier to read than the beeswarm plot for viva presentations.

    Args:
        shap_values  : 2D array of SHAP values (n_samples, n_features)
        feature_names: List of feature column names

    Returns:
        matplotlib Figure
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1][:15]   # Top 15

    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in indices]
    values = mean_shap[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(values)))
    bars = ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Prediction)", fontsize=11)
    ax.set_title("Global Feature Importance via SHAP", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "shap_bar.png")
    save_figure(fig, path)
    return fig


def generate_text_explanation(shap_values_single: np.ndarray, feature_names: list, prediction: int) -> str:
    """
    Generates a human-readable text explanation of a SHAP prediction.
    Used in the Streamlit dashboard for viva-friendly output.

    Args:
        shap_values_single: 1D SHAP values for one patient (n_features,)
        feature_names     : Feature column names
        prediction        : 0 or 1

    Returns:
        Formatted explanation string
    """
    outcome = "HEART DISEASE DETECTED" if prediction == 1 else "NO HEART DISEASE"

    # Sort features by absolute SHAP value (most impactful first)
    sorted_idx = np.argsort(np.abs(shap_values_single))[::-1]

    lines = [f"🔍 Prediction: {outcome}\n", "Top contributing factors:\n"]

    for i, idx in enumerate(sorted_idx[:5]):
        feat_name = FEATURE_LABELS.get(feature_names[idx], feature_names[idx])
        shap_val  = shap_values_single[idx]
        direction = "↑ INCREASED risk" if shap_val > 0 else "↓ DECREASED risk"
        lines.append(f"  {i+1}. {feat_name}: {direction} (SHAP={shap_val:+.3f})")

    return "\n".join(lines)


def run_explanation_pipeline(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list
) -> dict:
    """
    Full explainability pipeline:
      1. Create SHAP explainer using training data as background
      2. Compute SHAP values on test set
      3. Generate and save summary plot
      4. Generate bar chart
      5. Return explainer + shap_values for further use

    Args:
        model       : Trained sklearn/XGBoost model
        X_train     : Training data (used as SHAP background)
        X_test      : Test data (explained)
        feature_names: Column names

    Returns:
        Dict with: explainer, shap_values, fig_summary, fig_bar
    """
    logger.info("Running full SHAP explanation pipeline...")

    # Use a sample of training data as background (faster computation)
    # 100 samples is standard — enough to represent the distribution
    background_size = min(100, len(X_train))
    X_background = X_train.sample(n=background_size, random_state=42)

    explainer   = get_explainer(model, X_background)
    shap_values = compute_shap_values(explainer, X_test)

    fig_summary = plot_shap_summary(shap_values, X_test)
    fig_bar     = plot_shap_bar(shap_values, feature_names)

    logger.info("Explanation pipeline complete ✓")

    return {
        "explainer":   explainer,
        "shap_values": shap_values,
        "fig_summary": fig_summary,
        "fig_bar":     fig_bar,
    }


if __name__ == "__main__":
    from src.preprocessing import preprocess
    from src.utils import XGB_MODEL_PATH
    import joblib

    X_train, X_test, y_train, y_test, feature_names = preprocess()
    model = joblib.load(XGB_MODEL_PATH)

    results = run_explanation_pipeline(model, X_train, X_test, feature_names)
    print("\n✓ SHAP explanation complete")
    print(f"  Summary plot saved to: {SHAP_SUMMARY_PATH}")
    print(f"  SHAP values shape: {results['shap_values'].shape}")
