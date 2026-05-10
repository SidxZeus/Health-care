# ============================================================
# error_detection.py — Detecting Prediction Errors
# ============================================================
# PURPOSE:
#   Compares model predictions against actual (true) labels
#   to identify where and how the model fails. Logs all
#   mispredictions to a CSV file for downstream analysis.
#
# LOGIC:
#   - Iterate over (prediction, actual) pairs
#   - Mismatch → error; flag as FP or FN
#   - Record the patient's feature values for each error
#   - Write to outputs/error_log.csv
#   - Return error statistics
#
# TYPES OF ERRORS IN HEALTHCARE:
#   False Positive (FP): Predicted disease, actually healthy
#     → Patient may undergo unnecessary tests/worry
#   False Negative (FN): Predicted healthy, actually has disease
#     → MUCH MORE DANGEROUS — missed diagnosis
# ============================================================

import os
import numpy as np
import pandas as pd

from src.utils import (
    ERROR_LOG_PATH, CONFIDENCE_THRESHOLD,
    setup_logger
)

logger = setup_logger("error_detection")


def detect_errors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray = None,
) -> dict:
    """
    Identifies all misclassifications and logs them to CSV.

    Args:
        X_test       : Test feature DataFrame
        y_test       : True labels (Series)
        predictions  : Array of model predictions (0/1)
        confidences  : Array of confidence scores per prediction
        probabilities: Optional 2D array [[P(0), P(1)], ...] per row

    Returns:
        Dict with:
          - error_df       : DataFrame of all error cases
          - fp_df          : False Positives subset
          - fn_df          : False Negatives subset
          - stats          : Summary statistics
          - error_rate     : Fraction of total predictions that are wrong
    """
    logger.info("=" * 50)
    logger.info("Running error detection...")
    logger.info("=" * 50)

    # Reset indices to align X_test, y_test, predictions arrays
    X_reset = X_test.reset_index(drop=True)
    y_reset = pd.Series(y_test.values, name="actual")

    # Build a combined DataFrame for easy comparison
    error_data = X_reset.copy()
    error_data["actual"]      = y_reset.values
    error_data["predicted"]   = predictions
    error_data["confidence"]  = confidences
    error_data["is_error"]    = (predictions != y_reset.values)

    if probabilities is not None:
        error_data["prob_no_disease"]   = probabilities[:, 0]
        error_data["prob_heart_disease"] = probabilities[:, 1]

    # Filter to errors only
    error_df = error_data[error_data["is_error"]].copy()

    # Classify error type
    # FP: predicted=1, actual=0 (false alarm)
    # FN: predicted=0, actual=1 (missed detection)
    error_df["error_type"] = np.where(
        (error_df["predicted"] == 1) & (error_df["actual"] == 0),
        "False Positive (FP)",
        "False Negative (FN)"
    )

    # Flag low-confidence errors (these are the "uncertain" mistakes)
    error_df["low_confidence"] = error_df["confidence"] < CONFIDENCE_THRESHOLD

    fp_df = error_df[error_df["error_type"] == "False Positive (FP)"]
    fn_df = error_df[error_df["error_type"] == "False Negative (FN)"]

    # ── Statistics ──
    total       = len(error_data)
    n_errors    = len(error_df)
    n_fp        = len(fp_df)
    n_fn        = len(fn_df)
    error_rate  = n_errors / total if total > 0 else 0
    n_low_conf_errors = error_df["low_confidence"].sum()

    stats = {
        "total_predictions": total,
        "total_errors":      n_errors,
        "false_positives":   n_fp,
        "false_negatives":   n_fn,
        "error_rate":        round(error_rate, 4),
        "fp_rate":           round(n_fp / total, 4),
        "fn_rate":           round(n_fn / total, 4),
        "low_conf_errors":   int(n_low_conf_errors),
    }

    logger.info(f"  Total predictions : {total}")
    logger.info(f"  Correct           : {total - n_errors}")
    logger.info(f"  Errors            : {n_errors} ({error_rate:.1%})")
    logger.info(f"  False Positives   : {n_fp}")
    logger.info(f"  False Negatives   : {n_fn}  ← Most critical!")
    logger.info(f"  Low-confidence errors: {n_low_conf_errors}")

    # ── Save error log ──
    _save_error_log(error_df)

    return {
        "error_df":   error_df,
        "fp_df":      fp_df,
        "fn_df":      fn_df,
        "all_df":     error_data,
        "stats":      stats,
        "error_rate": error_rate,
    }


def _save_error_log(error_df: pd.DataFrame):
    """Saves the error DataFrame to CSV, appending if file exists."""
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)

    if os.path.exists(ERROR_LOG_PATH) and os.path.getsize(ERROR_LOG_PATH) > 0:
        existing = pd.read_csv(ERROR_LOG_PATH)
        combined = pd.concat([existing, error_df], ignore_index=True)
    else:
        combined = error_df

    combined.to_csv(ERROR_LOG_PATH, index=False)
    logger.info(f"Error log saved: {ERROR_LOG_PATH} ({len(error_df)} new errors)")


def load_error_log() -> pd.DataFrame:
    """Loads the persisted error log CSV."""
    if not os.path.exists(ERROR_LOG_PATH):
        return pd.DataFrame()
    return pd.read_csv(ERROR_LOG_PATH)


def get_error_summary_string(stats: dict) -> str:
    """Returns a printable summary of error statistics."""
    return (
        f"Error Detection Summary\n"
        f"{'─'*35}\n"
        f"Total Predictions : {stats['total_predictions']}\n"
        f"Total Errors      : {stats['total_errors']} ({stats['error_rate']:.1%})\n"
        f"False Positives   : {stats['false_positives']}\n"
        f"False Negatives   : {stats['false_negatives']}\n"
        f"Low-Conf Errors   : {stats['low_conf_errors']}\n"
    )


if __name__ == "__main__":
    from src.preprocessing import preprocess
    from src.predict import predict_batch
    import joblib
    from src.utils import XGB_MODEL_PATH

    X_train, X_test, y_train, y_test, feature_names = preprocess()
    model = joblib.load(XGB_MODEL_PATH)

    batch = predict_batch(X_test, model_type="xgb")
    results = detect_errors(
        X_test, y_test,
        batch["predictions"],
        batch["confidences"],
        batch["probabilities"]
    )

    print(get_error_summary_string(results["stats"]))
