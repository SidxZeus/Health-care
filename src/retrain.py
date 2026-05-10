# ============================================================
# retrain.py — Retraining & Performance Comparison
# ============================================================
# PURPOSE:
#   Takes the corrected training data and improved hyperparameters
#   from correction_engine.py, retrains the models, then compares
#   old vs new performance side-by-side.
#
# SELF-CORRECTION LOOP SUMMARY:
#   1. Train initial model
#   2. Detect errors
#   3. Analyze errors
#   4. Apply corrections
#   5. RETRAIN ← this module
#   6. Compare → decide if improvement is achieved
#
# EXPECTED OUTPUT:
#   - Updated models saved to models/
#   - outputs/performance_comparison.png
#   - Updated model_metrics.json with "v2" entry
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils import (
    RF_MODEL_PATH, XGB_MODEL_PATH,
    PERF_COMPARE_PATH, OUTPUTS_DIR,
    compute_metrics, save_metrics, load_metrics,
    save_figure, setup_logger
)

logger = setup_logger("retrain")


def retrain_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_params: dict,
    xgb_params: dict,
) -> tuple:
    """
    Retrains both models on the corrected training data.

    Args:
        X_train   : Corrected (possibly SMOTE-augmented) training features
        y_train   : Corrected training labels
        rf_params : Hyperparameters for Random Forest
        xgb_params: Hyperparameters for XGBoost

    Returns:
        (rf_model_new, xgb_model_new) — freshly trained models
    """
    logger.info(f"Retraining RF on {len(X_train)} samples with params: {rf_params}")
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)

    # Clean xgb_params — remove any non-XGB kwargs
    xgb_clean = {k: v for k, v in xgb_params.items()
                 if k not in ("use_label_encoder",)}
    logger.info(f"Retraining XGBoost with params: {xgb_clean}")
    xgb_model = XGBClassifier(**xgb_clean)
    xgb_model.fit(X_train, y_train)

    logger.info("Retraining complete ✓")
    return rf_model, xgb_model


def evaluate_and_compare(
    old_metrics: dict,
    rf_new: object,
    xgb_new: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluates newly trained models and compares against old metrics.

    Args:
        old_metrics: Dict with "random_forest" and "xgboost" keys from previous run
        rf_new     : New Random Forest model
        xgb_new    : New XGBoost model
        X_test     : Test features
        y_test     : True test labels

    Returns:
        comparison dict with old/new metrics and delta values
    """
    logger.info("Evaluating new models...")

    rf_pred  = rf_new.predict(X_test)
    rf_prob  = rf_new.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_new.predict(X_test)
    xgb_prob = xgb_new.predict_proba(X_test)[:, 1]

    rf_new_metrics  = compute_metrics(y_test, rf_pred, rf_prob)
    xgb_new_metrics = compute_metrics(y_test, xgb_pred, xgb_prob)
    rf_new_metrics["model"]  = "Random Forest (Retrained)"
    xgb_new_metrics["model"] = "XGBoost (Retrained)"

    # Compute improvement deltas
    rf_old  = old_metrics.get("random_forest", {})
    xgb_old = old_metrics.get("xgboost", {})

    def delta(new_val, old_val):
        if new_val is None or old_val is None:
            return None
        return round(new_val - old_val, 4)

    comparison = {
        "rf": {
            "old": rf_old,
            "new": rf_new_metrics,
            "delta": {
                "accuracy":  delta(rf_new_metrics["accuracy"],  rf_old.get("accuracy")),
                "f1":        delta(rf_new_metrics["f1"],        rf_old.get("f1")),
                "roc_auc":   delta(rf_new_metrics["roc_auc"],   rf_old.get("roc_auc")),
            }
        },
        "xgb": {
            "old": xgb_old,
            "new": xgb_new_metrics,
            "delta": {
                "accuracy":  delta(xgb_new_metrics["accuracy"],  xgb_old.get("accuracy")),
                "f1":        delta(xgb_new_metrics["f1"],        xgb_old.get("f1")),
                "roc_auc":   delta(xgb_new_metrics["roc_auc"],   xgb_old.get("roc_auc")),
            }
        },
    }

    logger.info("Performance Comparison:")
    for m_name, m_data in comparison.items():
        acc_delta = m_data["delta"]["accuracy"]
        f1_delta  = m_data["delta"]["f1"]
        arrow = "↑" if (acc_delta or 0) >= 0 else "↓"
        logger.info(
            f"  {m_name.upper()}: Accuracy {m_data['old'].get('accuracy','?')} → "
            f"{m_data['new']['accuracy']} ({arrow}{abs(acc_delta or 0):.4f}) | "
            f"F1: {m_data['old'].get('f1','?')} → {m_data['new']['f1']}"
        )

    return comparison


def plot_performance_comparison(comparison: dict) -> plt.Figure:
    """
    Side-by-side bar chart comparing old vs new model performance
    on key metrics (Accuracy, F1, ROC-AUC).

    Args:
        comparison: Dict from evaluate_and_compare()

    Returns:
        matplotlib Figure (also saved to outputs/)
    """
    metrics_to_show = ["accuracy", "f1", "roc_auc"]
    metric_labels   = ["Accuracy", "F1 Score", "ROC-AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Before vs After Self-Correction",
                 fontsize=16, fontweight="bold")

    model_keys   = ["rf",    "xgb"]
    model_labels = ["Random Forest", "XGBoost"]
    bar_colors   = {"Before": "#78909C", "After": "#2196F3"}

    for ax, m_key, m_label in zip(axes, model_keys, model_labels):
        m_data = comparison[m_key]

        old_vals = [m_data["old"].get(m, 0) or 0 for m in metrics_to_show]
        new_vals = [m_data["new"].get(m, 0) or 0 for m in metrics_to_show]

        x = np.arange(len(metric_labels))
        width = 0.35

        bars_old = ax.bar(x - width/2, old_vals, width, label="Before",
                          color=bar_colors["Before"], edgecolor="white", alpha=0.85)
        bars_new = ax.bar(x + width/2, new_vals, width, label="After",
                          color=bar_colors["After"], edgecolor="white")

        # Add value labels on top of bars
        for bar in list(bars_old) + list(bars_new):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

        # Highlight improvement arrows
        for i, (old_v, new_v) in enumerate(zip(old_vals, new_vals)):
            delta = new_v - old_v
            color = "green" if delta >= 0 else "red"
            symbol = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            ax.annotate(symbol, xy=(x[i], max(old_v, new_v) + 0.04),
                        ha="center", fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(m_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, PERF_COMPARE_PATH)
    logger.info(f"Performance comparison saved: {PERF_COMPARE_PATH}")
    return fig


def save_retrained_models(rf_model, xgb_model):
    """Overwrites the saved models with newly retrained versions."""
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    logger.info(f"Retrained models saved: {RF_MODEL_PATH}, {XGB_MODEL_PATH}")


def retrain_pipeline(
    X_train_corrected: pd.DataFrame,
    y_train_corrected: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rf_params: dict,
    xgb_params: dict,
    corrections_applied: list,
) -> dict:
    """
    Full retraining pipeline:
      1. Retrain both models
      2. Evaluate and compare
      3. Save plots + updated models + metrics

    Args:
        X_train_corrected : Corrected training features
        y_train_corrected : Corrected training labels
        X_test            : Original test features (unchanged)
        y_test            : True test labels
        rf_params         : Updated RF hyperparameters
        xgb_params        : Updated XGB hyperparameters
        corrections_applied: List of applied strategies (for logging)

    Returns:
        Dict with new models, comparison, and performance figure
    """
    logger.info("=" * 55)
    logger.info("  RETRAINING PIPELINE")
    logger.info(f"  Corrections: {', '.join(corrections_applied)}")
    logger.info("=" * 55)

    # Load old metrics for comparison
    old_metrics_all = load_metrics()
    # Get the most recent version's metrics
    if old_metrics_all:
        latest_key = sorted(old_metrics_all.keys())[-1]
        old_metrics = old_metrics_all[latest_key]
    else:
        old_metrics = {}

    # Retrain
    rf_new, xgb_new = retrain_models(
        X_train_corrected, y_train_corrected,
        rf_params, xgb_params
    )

    # Compare
    comparison = evaluate_and_compare(old_metrics, rf_new, xgb_new, X_test, y_test)

    # Save comparison plot
    fig_compare = plot_performance_comparison(comparison)

    # Save new metrics
    new_metrics = {
        "random_forest":    comparison["rf"]["new"],
        "xgboost":          comparison["xgb"]["new"],
        "corrections":      corrections_applied,
        "training_samples": len(X_train_corrected),
    }
    save_metrics(new_metrics)

    # Overwrite saved models
    save_retrained_models(rf_new, xgb_new)

    # Summarize improvement
    acc_delta = comparison["xgb"]["delta"]["accuracy"] or 0
    improved  = acc_delta > 0

    logger.info(f"\n{'='*40}")
    logger.info(f"  Retraining Result: {'IMPROVED ✓' if improved else 'No improvement'}")
    logger.info(f"  XGB Accuracy delta: {acc_delta:+.4f}")
    logger.info(f"{'='*40}")

    return {
        "rf_model_new":    rf_new,
        "xgb_model_new":   xgb_new,
        "comparison":      comparison,
        "fig_comparison":  fig_compare,
        "improved":        improved,
        "acc_delta":       acc_delta,
    }


if __name__ == "__main__":
    print("retrain.py: Call retrain_pipeline() after correction_engine.apply_corrections()")
