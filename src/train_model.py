# ============================================================
# train_model.py — Model Training & Evaluation
# ============================================================
# PURPOSE:
#   Trains two classifiers (Random Forest & XGBoost) on the
#   preprocessed heart disease data. Evaluates both, saves
#   models to disk, and generates evaluation plots
#   (confusion matrix + ROC curve) for the Streamlit dashboard.
#
# LOGIC FLOW:
#   1. Preprocess data (calls preprocessing.py)
#   2. Train Random Forest
#   3. Train XGBoost
#   4. Evaluate both classifiers with multiple metrics
#   5. Plot confusion matrices and ROC curves
#   6. Save both models + metrics to disk
#
# EXPECTED OUTPUT:
#   - models/rf_model.pkl
#   - models/xgb_model.pkl
#   - models/model_metrics.json
#   - outputs/confusion_matrix.png
#   - outputs/roc_curve.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
from xgboost import XGBClassifier

# Import shared utilities
from src.utils import (
    RF_MODEL_PATH, XGB_MODEL_PATH, OUTPUTS_DIR,
    RF_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS,
    compute_metrics, save_metrics, save_figure,
    setup_logger, FEATURE_LABELS
)
from src.preprocessing import preprocess

logger = setup_logger("train_model")


# ─────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train, params: dict = None):
    """
    Train a Random Forest classifier.

    WHY RANDOM FOREST:
        An ensemble of decision trees where each tree is trained on a
        random subset of the data (bagging) and a random subset of features
        (feature randomness). This prevents overfitting and makes the model
        robust to noisy medical data.

    Args:
        X_train : Training features
        y_train : Training labels
        params  : Hyperparameter dict (uses RF_DEFAULT_PARAMS if None)

    Returns:
        Trained RandomForestClassifier
    """
    p = params or RF_DEFAULT_PARAMS
    logger.info(f"Training Random Forest with params: {p}")

    model = RandomForestClassifier(**p)
    model.fit(X_train, y_train)

    logger.info("Random Forest training complete ✓")
    return model


def train_xgboost(X_train, y_train, params: dict = None):
    """
    Train an XGBoost classifier.

    WHY XGBOOST:
        Gradient boosting builds trees sequentially — each tree corrects
        the errors of the previous one. XGBoost adds regularization
        (L1/L2), which prevents overfitting. It typically outperforms
        Random Forest on tabular data with proper tuning.

    WHY USE BOTH:
        Comparing two different algorithm families gives us insight into
        which approach works better for this specific dataset. It also
        allows us to select the best model for deployment.

    Args:
        X_train : Training features
        y_train : Training labels
        params  : Hyperparameter dict (uses XGB_DEFAULT_PARAMS if None)

    Returns:
        Trained XGBClassifier
    """
    p = params or XGB_DEFAULT_PARAMS.copy()
    # Remove keys not valid for newer XGBoost versions
    p.pop("use_label_encoder", None)

    logger.info(f"Training XGBoost with params: {p}")

    model = XGBClassifier(**p)
    model.fit(X_train, y_train)

    logger.info("XGBoost training complete ✓")
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model and print a classification report.

    Args:
        model      : Trained sklearn-compatible model
        X_test     : Test features
        y_test     : True test labels
        model_name : Name string for logging/display

    Returns:
        Dict of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]   # Probability for class=1

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["model"] = model_name

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation — {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy : {metrics['accuracy']}")
    logger.info(f"  Precision: {metrics['precision']}")
    logger.info(f"  Recall   : {metrics['recall']}")
    logger.info(f"  F1 Score : {metrics['f1']}")
    logger.info(f"  ROC-AUC  : {metrics['roc_auc']}")

    # Full classification report (shows per-class stats)
    report = classification_report(y_test, y_pred,
                                   target_names=["No Disease", "Heart Disease"])
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_confusion_matrices(
    rf_model, xgb_model, X_test, y_test
) -> plt.Figure:
    """
    Plots side-by-side confusion matrices for both models.

    A confusion matrix shows:
      - True Positives (TP): Correctly predicted disease
      - True Negatives (TN): Correctly predicted no disease
      - False Positives (FP): Predicted disease when healthy (overdiagnosis)
      - False Negatives (FN): Missed disease (more dangerous in healthcare!)

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

    models = [("Random Forest", rf_model), ("XGBoost", xgb_model)]

    for ax, (name, model) in zip(axes, models):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,          # Show count numbers in each cell
            fmt="d",             # Integer format
            cmap="Blues",        # Blue colormap (medical feel)
            ax=ax,
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    return fig


def plot_roc_curves(
    rf_model, xgb_model, X_test, y_test
) -> plt.Figure:
    """
    Plots ROC curves for both models on the same axes.

    ROC (Receiver Operating Characteristic) curve shows
    the trade-off between True Positive Rate and False Positive Rate
    at different classification thresholds.
    AUC (Area Under the Curve) closer to 1.0 means better discrimination.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#FF5722"]
    models = [("Random Forest", rf_model), ("XGBoost", xgb_model)]

    for color, (name, model) in zip(colors, models):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

    # Diagonal reference line (random classifier baseline)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC=0.5)")
    ax.fill_between(fpr, tpr, alpha=0.05, color=colors[-1])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(rf_model, feature_names: list) -> plt.Figure:
    """
    Plots top-15 feature importances from the Random Forest model.

    Feature importance = average decrease in impurity (Gini/entropy)
    across all trees when a feature is used for splitting.
    Higher importance → feature has more predictive power.

    Returns:
        matplotlib Figure
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]   # Top 15

    # Use human-readable labels where available
    labels = [
        FEATURE_LABELS.get(feature_names[i], feature_names[i])
        for i in indices
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(indices)),
        importances[indices],
        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(indices)))[::-1]
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
    ax.set_title("Top Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.invert_yaxis()   # Highest importance at top
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────

def save_models(rf_model, xgb_model):
    """
    Persist both trained models to disk using Joblib.

    WHY JOBLIB (not pickle):
        Joblib is optimized for large NumPy arrays, which are
        the internal structure of scikit-learn models.
        It's faster and more memory-efficient than pickle for ML models.
    """
    os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    logger.info(f"Models saved: {RF_MODEL_PATH}, {XGB_MODEL_PATH}")


# ─────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────

def train_pipeline(rf_params=None, xgb_params=None) -> dict:
    """
    Full training pipeline:
      1. Preprocess data
      2. Train RF + XGBoost
      3. Evaluate both
      4. Save models, metrics, and plots

    Args:
        rf_params  : Optional override for RF hyperparameters
        xgb_params : Optional override for XGBoost hyperparameters

    Returns:
        dict with both models, metrics, and feature_names
    """
    logger.info("=" * 55)
    logger.info("  HEALTHCARE AI — MODEL TRAINING PIPELINE")
    logger.info("=" * 55)

    # ── Step 1: Preprocess ──
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    # ── Step 2: Train models ──
    rf_model  = train_random_forest(X_train, y_train, rf_params)
    xgb_model = train_xgboost(X_train, y_train, xgb_params)

    # ── Step 3: Evaluate ──
    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # ── Step 4: Save metrics ──
    combined_metrics = {
        "random_forest": rf_metrics,
        "xgboost":       xgb_metrics,
        "feature_names": feature_names,
    }
    save_metrics(combined_metrics)

    # ── Step 5: Save plots ──
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig_cm = plot_confusion_matrices(rf_model, xgb_model, X_test, y_test)
    save_figure(fig_cm, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))

    fig_roc = plot_roc_curves(rf_model, xgb_model, X_test, y_test)
    save_figure(fig_roc, os.path.join(OUTPUTS_DIR, "roc_curve.png"))

    fig_fi = plot_feature_importance(rf_model, feature_names)
    save_figure(fig_fi, os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    logger.info("All plots saved to outputs/ ✓")

    # ── Step 6: Save models ──
    save_models(rf_model, xgb_model)

    logger.info("Training pipeline complete ✓")
    logger.info(f"  RF  Accuracy: {rf_metrics['accuracy']}")
    logger.info(f"  XGB Accuracy: {xgb_metrics['accuracy']}")

    return {
        "rf_model":      rf_model,
        "xgb_model":     xgb_model,
        "feature_names": feature_names,
        "X_test":        X_test,
        "y_test":        y_test,
        "rf_metrics":    rf_metrics,
        "xgb_metrics":   xgb_metrics,
    }


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = train_pipeline()
    print("\n✓ Training complete!")
    print(f"  Random Forest Accuracy : {results['rf_metrics']['accuracy']}")
    print(f"  XGBoost Accuracy       : {results['xgb_metrics']['accuracy']}")
    print(f"  Models saved to        : models/")
    print(f"  Plots saved to         : outputs/")
