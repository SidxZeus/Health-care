# ============================================================
# correction_engine.py — Self-Correction Strategies
# ============================================================
# PURPOSE:
#   Based on the error analysis report, selects and applies
#   the right correction strategies to the training data
#   and/or model hyperparameters before retraining.
#
# CORRECTION STRATEGIES:
#   1. SMOTE  — Synthetic oversampling of minority class
#   2. Threshold Tuning — Find optimal decision boundary
#   3. Hyperparameter Tuning — GridSearchCV on best model
#   4. Default fallback — Increase trees / lower learning rate
#
# WHY RULE-BASED CORRECTIONS?
#   The engine reads the analysis report and applies targeted
#   fixes — matching specific problems to specific solutions.
#   This is the "self" in "self-correcting AI".
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_curve
from imblearn.over_sampling import SMOTE

from src.utils import (
    RF_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS,
    CONFIDENCE_THRESHOLD, RANDOM_SEED,
    setup_logger
)

logger = setup_logger("correction_engine")


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Applies SMOTE (Synthetic Minority Oversampling Technique) to
    balance the class distribution in the training set.

    HOW SMOTE WORKS:
        For each minority-class sample (heart disease patient),
        SMOTE finds its k-nearest neighbors within the same class
        and generates NEW synthetic samples along the line segments
        connecting them. This avoids simple duplication and
        creates more representative minority-class examples.

    WHY NOT JUST DUPLICATE:
        Simply duplicating minority samples makes the model memorize
        those exact patients rather than learning generalizable patterns.
        SMOTE generates novel, realistic synthetic patients.

    Args:
        X_train: Training features
        y_train: Training labels (may be imbalanced)

    Returns:
        (X_resampled, y_resampled) — balanced dataset
    """
    logger.info("Applying SMOTE oversampling...")
    logger.info(f"  Before — Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")

    smote = SMOTE(
        k_neighbors=5,          # Use 5 nearest neighbors for synthesis
        random_state=RANDOM_SEED,
        sampling_strategy="auto"  # Auto-balance to 1:1 ratio
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(f"  After  — Class 0: {(y_resampled==0).sum()} | Class 1: {(y_resampled==1).sum()}")
    logger.info(f"  Synthetic samples added: {len(X_resampled) - len(X_train)}")

    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def tune_threshold(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Finds the optimal classification threshold that maximizes F1 score.

    WHY THRESHOLD TUNING:
        By default, models use a 0.5 threshold: P(disease) >= 0.5 → predict disease.
        But in healthcare, we might prefer higher recall (fewer missed diagnoses)
        at the cost of precision. Tuning the threshold gives us this control.

    HOW:
        We compute the F1 score for thresholds from 0.1 to 0.9.
        The threshold with the highest F1 is chosen.

    Args:
        model : Trained model
        X_test: Test features
        y_test: True labels

    Returns:
        optimal_threshold: Float between 0.0 and 1.0
    """
    logger.info("Tuning decision threshold...")

    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    # Try 50 threshold values between 0.2 and 0.8
    for threshold in np.linspace(0.20, 0.80, 60):
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_at_threshold, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"  Optimal threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    return round(best_threshold, 3)


def tune_hyperparameters_rf(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Grid search for optimal Random Forest hyperparameters.

    WHY GRIDSEARCHCV:
        Exhaustively tests all combinations of specified parameters
        using cross-validation. Finds the combination that generalizes
        best — not just performs best on training data.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        best_params: Dict of optimal hyperparameters
    """
    logger.info("Tuning Random Forest hyperparameters via GridSearchCV...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        rf, param_grid,
        cv=cv,
        scoring="f1",        # Optimize for F1 (balance precision/recall)
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_params["random_state"] = RANDOM_SEED
    best_params["n_jobs"] = -1

    logger.info(f"  Best RF params: {best_params}")
    logger.info(f"  Best CV F1: {grid_search.best_score_:.4f}")
    return best_params


def tune_hyperparameters_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Grid search for optimal XGBoost hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        best_params: Dict of optimal hyperparameters
    """
    logger.info("Tuning XGBoost hyperparameters via GridSearchCV...")

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample":     [0.8, 1.0],
    }

    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        xgb, param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_params["eval_metric"]    = "logloss"
    best_params["random_state"]   = RANDOM_SEED

    logger.info(f"  Best XGB params: {best_params}")
    logger.info(f"  Best CV F1: {grid_search.best_score_:.4f}")
    return best_params


def apply_corrections(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    correction_triggers: list,
    current_model=None,
) -> dict:
    """
    Master correction function. Reads the triggers from error_analysis
    and applies the appropriate corrections.

    Args:
        X_train            : Training features
        y_train            : Training labels
        X_test             : Test features
        y_test             : Test labels
        correction_triggers: List from error_analysis (e.g. ["smote", "threshold_tuning"])
        current_model      : The currently trained model (for threshold tuning)

    Returns:
        correction_result dict with:
          - X_train_corrected: Potentially resampled training features
          - y_train_corrected: Potentially resampled training labels
          - rf_params        : Hyperparameters for RF retraining
          - xgb_params       : Hyperparameters for XGB retraining
          - optimal_threshold: Float (0.5 default if not tuned)
          - corrections_applied: List of applied strategies
    """
    logger.info("=" * 50)
    logger.info("CORRECTION ENGINE — Applying Fixes")
    logger.info(f"Triggers received: {correction_triggers}")
    logger.info("=" * 50)

    X_corrected  = X_train.copy()
    y_corrected  = y_train.copy()
    rf_params    = RF_DEFAULT_PARAMS.copy()
    xgb_params   = XGB_DEFAULT_PARAMS.copy()
    threshold    = 0.5
    applied      = []

    # ── Correction 1: SMOTE ──
    if "smote" in correction_triggers:
        try:
            X_corrected, y_corrected = apply_smote(X_corrected, y_corrected)
            applied.append("SMOTE Oversampling")
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")

    # ── Correction 2: Threshold Tuning ──
    if "threshold_tuning" in correction_triggers and current_model is not None:
        try:
            threshold = tune_threshold(current_model, X_test, y_test)
            applied.append(f"Threshold → {threshold}")
        except Exception as e:
            logger.error(f"Threshold tuning failed: {e}")

    # ── Correction 3: Hyperparameter Tuning ──
    if "hyperparameter_tuning" in correction_triggers:
        try:
            rf_params  = tune_hyperparameters_rf(X_corrected, y_corrected)
            xgb_params = tune_hyperparameters_xgb(X_corrected, y_corrected)
            applied.append("Hyperparameter Tuning (RF + XGBoost)")
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            # Fallback: slightly adjust defaults
            rf_params["n_estimators"]  = 200
            xgb_params["n_estimators"] = 150
            xgb_params["learning_rate"] = 0.08
            applied.append("Default Parameter Adjustment (fallback)")

    # If no triggers, apply conservative default improvements
    if not applied:
        logger.info("No triggers — applying conservative default improvements")
        rf_params["n_estimators"]  = 150
        xgb_params["n_estimators"] = 120
        applied.append("Conservative Parameter Boost")

    logger.info(f"Corrections applied: {applied}")

    return {
        "X_train_corrected": X_corrected,
        "y_train_corrected": y_corrected,
        "rf_params":         rf_params,
        "xgb_params":        xgb_params,
        "optimal_threshold": threshold,
        "corrections_applied": applied,
    }


if __name__ == "__main__":
    print("Correction engine loaded. Call apply_corrections() with triggers.")
