# ============================================================
# error_analysis.py — Root-Cause Analysis of Prediction Errors
# ============================================================
# PURPOSE:
#   Digs into the error cases found by error_detection.py
#   and identifies WHY the model is failing. Produces
#   analysis charts and a structured report that feeds
#   directly into the correction_engine.py.
#
# ANALYSIS DIMENSIONS:
#   1. Class imbalance — Is one class underrepresented?
#   2. Feature distributions — Are error cases "edge cases"?
#   3. Confidence analysis — Are errors low-confidence?
#   4. Recurring patterns — Do errors share common features?
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    NUMERICAL_FEATURES, OUTPUTS_DIR,
    FEATURE_LABELS, setup_logger, save_figure
)

logger = setup_logger("error_analysis")


def analyze_class_imbalance(y_train: pd.Series) -> dict:
    """
    Checks whether the training set has a class imbalance.

    WHY THIS MATTERS:
        If 80% of patients are healthy and 20% have disease,
        a model can achieve 80% accuracy by ALWAYS predicting "healthy".
        The model learns a majority-class bias. SMOTE oversampling
        in correction_engine.py directly addresses this.

    Args:
        y_train: Training labels Series

    Returns:
        Dict with class counts, ratio, and imbalance flag
    """
    counts = y_train.value_counts()
    total  = len(y_train)

    no_disease_pct  = counts.get(0, 0) / total
    has_disease_pct = counts.get(1, 0) / total
    ratio = no_disease_pct / has_disease_pct if has_disease_pct > 0 else float("inf")

    # Flag as imbalanced if majority is > 60% of data
    is_imbalanced = max(no_disease_pct, has_disease_pct) > 0.60

    analysis = {
        "no_disease_count":  int(counts.get(0, 0)),
        "disease_count":     int(counts.get(1, 0)),
        "no_disease_pct":    round(no_disease_pct, 3),
        "has_disease_pct":   round(has_disease_pct, 3),
        "imbalance_ratio":   round(ratio, 2),
        "is_imbalanced":     is_imbalanced,
        "recommendation":    "Apply SMOTE oversampling" if is_imbalanced else "Class balance is acceptable",
    }

    logger.info(f"Class balance: {no_disease_pct:.1%} no-disease | {has_disease_pct:.1%} disease")
    if is_imbalanced:
        logger.warning("Class imbalance detected! Recommend SMOTE.")

    return analysis


def analyze_error_feature_distributions(
    error_df: pd.DataFrame,
    correct_df: pd.DataFrame,
    feature_names: list,
) -> plt.Figure:
    """
    Compares feature distributions between correctly-classified
    and misclassified patients using violin plots.

    Key insight: If a feature's distribution is very different
    in error cases vs correct cases, that feature is likely
    contributing to model confusion. This informs which features
    to focus on during retraining.

    Args:
        error_df  : DataFrame of misclassified patients
        correct_df: DataFrame of correctly classified patients
        feature_names: All feature column names

    Returns:
        matplotlib Figure
    """
    # Focus on numerical features that are meaningful to compare
    num_cols = [c for c in NUMERICAL_FEATURES if c in error_df.columns]
    if not num_cols:
        logger.warning("No numerical features found for distribution analysis.")
        return None

    logger.info(f"Analyzing feature distributions for {len(num_cols)} numerical features")

    n_cols = min(len(num_cols), 3)   # Max 3 per row
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        label = FEATURE_LABELS.get(col, col)

        # Combine for plotting
        plot_df = pd.concat([
            error_df[[col]].assign(Category="Errors"),
            correct_df[[col]].assign(Category="Correct"),
        ], ignore_index=True)

        sns.violinplot(
            data=plot_df, x="Category", y=col,
            palette={"Errors": "#FF5722", "Correct": "#2196F3"},
            inner="quartile", ax=ax
        )
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution: Errors vs Correctly Classified",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "error_feature_dist.png")
    save_figure(fig, path)
    logger.info(f"Feature distribution plot saved: {path}")
    return fig


def analyze_confidence_in_errors(error_df: pd.DataFrame) -> dict:
    """
    Examines confidence scores of misclassified predictions.

    Insight: If errors are concentrated at high confidence,
    the model is confidently wrong — a sign of overfitting or
    distribution shift. If errors are at low confidence,
    adjusting the decision threshold might help.

    Args:
        error_df: DataFrame of error cases with 'confidence' column

    Returns:
        Dict with confidence statistics for error cases
    """
    if "confidence" not in error_df.columns or len(error_df) == 0:
        return {"error": "No confidence data available"}

    confs = error_df["confidence"]

    analysis = {
        "mean_confidence":     round(confs.mean(), 3),
        "median_confidence":   round(confs.median(), 3),
        "pct_low_confidence":  round((confs < 0.60).mean(), 3),
        "pct_high_confidence": round((confs >= 0.80).mean(), 3),
        "recommendation": "",
    }

    if analysis["pct_low_confidence"] > 0.5:
        analysis["recommendation"] = (
            "Most errors occur at low confidence — "
            "consider adjusting the decision threshold."
        )
    elif analysis["pct_high_confidence"] > 0.4:
        analysis["recommendation"] = (
            "Model is confidently wrong on many cases — "
            "possible overfitting; consider more regularization."
        )
    else:
        analysis["recommendation"] = (
            "Confidence pattern is mixed — "
            "focus on feature engineering and retraining."
        )

    logger.info(f"Error confidence: mean={analysis['mean_confidence']:.3f} | "
                f"low_conf_errors={analysis['pct_low_confidence']:.1%}")
    logger.info(f"Recommendation: {analysis['recommendation']}")

    return analysis


def analyze_error_patterns(error_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Finds recurring patterns in error cases by computing
    feature-level statistics (mean, std) and comparing
    to population-level stats.

    Args:
        error_df    : Misclassified patient DataFrame
        feature_names: Feature column names

    Returns:
        DataFrame summarizing error-pattern statistics per feature
    """
    if len(error_df) == 0:
        return pd.DataFrame()

    feature_cols = [c for c in feature_names if c in error_df.columns]

    stats = []
    for col in feature_cols:
        label = FEATURE_LABELS.get(col, col)
        try:
            stats.append({
                "Feature":     label,
                "Mean (Errors)": round(error_df[col].mean(), 3),
                "Std (Errors)":  round(error_df[col].std(), 3),
                "Min":           round(error_df[col].min(), 3),
                "Max":           round(error_df[col].max(), 3),
            })
        except Exception:
            pass

    return pd.DataFrame(stats)


def plot_error_type_breakdown(stats: dict) -> plt.Figure:
    """
    Bar chart showing FP vs FN breakdown and accuracy.

    Args:
        stats: Dict from detect_errors() stats key

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Error type pie chart ──
    ax = axes[0]
    if stats["total_errors"] > 0:
        labels = ["False Positives\n(Overdiagnosis)", "False Negatives\n(Missed Disease)"]
        values = [stats["false_positives"], stats["false_negatives"]]
        colors = ["#FF9800", "#F44336"]
        ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
        ax.set_title("Error Type Distribution", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No Errors!", ha="center", va="center",
                fontsize=16, color="green")
        ax.set_title("Error Type Distribution")

    # ── Right: Overall accuracy bar ──
    ax2 = axes[1]
    total = stats["total_predictions"]
    correct = total - stats["total_errors"]
    bars = ax2.bar(
        ["Correct", "Errors"],
        [correct, stats["total_errors"]],
        color=["#4CAF50", "#F44336"],
        edgecolor="white", linewidth=1.5,
        width=0.5
    )
    for bar, val in zip(bars, [correct, stats["total_errors"]]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=12)
    ax2.set_ylabel("Number of Predictions", fontsize=11)
    ax2.set_title(f"Prediction Breakdown (n={total})", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "error_breakdown.png")
    save_figure(fig, path)
    return fig


def generate_analysis_report(
    y_train: pd.Series,
    error_df: pd.DataFrame,
    correct_df: pd.DataFrame,
    stats: dict,
    feature_names: list,
) -> dict:
    """
    Master function that runs all analyses and returns a
    structured report consumed by the correction engine.

    Returns:
        report dict with all analysis results + recommendations
    """
    logger.info("Generating full error analysis report...")

    imbalance = analyze_class_imbalance(y_train)
    confidence_analysis = analyze_confidence_in_errors(error_df)
    pattern_df = analyze_error_patterns(error_df, feature_names)

    # Generate plots
    fig_dist = analyze_error_feature_distributions(error_df, correct_df, feature_names)
    fig_breakdown = plot_error_type_breakdown(stats)

    # Compile correction recommendations
    recommendations = []
    if imbalance["is_imbalanced"]:
        recommendations.append("smote")
    if confidence_analysis.get("pct_low_confidence", 0) > 0.5:
        recommendations.append("threshold_tuning")
    if stats["error_rate"] > 0.15:
        recommendations.append("hyperparameter_tuning")

    report = {
        "class_imbalance":       imbalance,
        "confidence_analysis":   confidence_analysis,
        "error_pattern_table":   pattern_df,
        "correction_triggers":   recommendations,
        "fig_distribution":      fig_dist,
        "fig_breakdown":         fig_breakdown,
        "summary":               (
            f"Error rate: {stats['error_rate']:.1%} | "
            f"FP: {stats['false_positives']} | FN: {stats['false_negatives']} | "
            f"Triggers: {', '.join(recommendations) or 'None'}"
        ),
    }

    logger.info(f"Analysis report: {report['summary']}")
    logger.info(f"Correction triggers: {recommendations}")
    return report


if __name__ == "__main__":
    from src.preprocessing import preprocess
    from src.predict import predict_batch
    from src.error_detection import detect_errors
    import joblib
    from src.utils import XGB_MODEL_PATH

    X_train, X_test, y_train, y_test, feature_names = preprocess()
    model  = joblib.load(XGB_MODEL_PATH)
    batch  = predict_batch(X_test)
    errors = detect_errors(X_test, y_test, batch["predictions"],
                           batch["confidences"], batch["probabilities"])

    correct_df = errors["all_df"][~errors["all_df"]["is_error"]]
    report = generate_analysis_report(
        y_train, errors["error_df"], correct_df,
        errors["stats"], feature_names
    )
    print(f"\n✓ Analysis complete: {report['summary']}")
    print(f"  Correction triggers: {report['correction_triggers']}")
