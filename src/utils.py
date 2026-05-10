# ============================================================
# utils.py — Shared Utilities & Configuration
# ============================================================
# PURPOSE:
#   This module acts as the backbone of the project.
#   It holds all shared config (paths, feature names),
#   the logging setup, and reusable helper functions.
#   Every other module imports from here to stay consistent.
#
# WHY THIS APPROACH:
#   Centralizing config prevents "magic strings" scattered
#   across files. If you rename a file path, you change it
#   once here — not in 10 different places.
# ============================================================

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────
# 1. PROJECT ROOT & FOLDER PATHS
# ─────────────────────────────────────────────
# os.path.dirname(__file__) → folder containing utils.py  → src/
# Going one level up (..) gives the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All key directories derived from the root
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR    = os.path.join(PROJECT_ROOT, "outputs")   # logs go inside outputs/

# Specific file paths used across modules
DATASET_PATH      = os.path.join(DATA_DIR,    "heart.csv")
RF_MODEL_PATH     = os.path.join(MODELS_DIR,  "rf_model.pkl")
XGB_MODEL_PATH    = os.path.join(MODELS_DIR,  "xgb_model.pkl")
SCALER_PATH       = os.path.join(MODELS_DIR,  "scaler.pkl")
METRICS_PATH      = os.path.join(MODELS_DIR,  "model_metrics.json")
ERROR_LOG_PATH    = os.path.join(OUTPUTS_DIR, "error_log.csv")
SHAP_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "shap_summary.png")
PERF_COMPARE_PATH = os.path.join(OUTPUTS_DIR, "performance_comparison.png")

# Public URL for the UCI Heart Disease dataset (Cleveland subset, cleaned)
DATASET_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)

# ─────────────────────────────────────────────
# 2. FEATURE DEFINITIONS
# ─────────────────────────────────────────────
# These are the 13 input features + 1 target column
# from the Cleveland Heart Disease dataset.

FEATURE_NAMES = [
    "age",        # Patient age in years
    "sex",        # 1 = male, 0 = female
    "cp",         # Chest pain type (0-3)
    "trestbps",   # Resting blood pressure (mm Hg)
    "chol",       # Serum cholesterol (mg/dl)
    "fbs",        # Fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",    # Resting ECG results (0-2)
    "thalach",    # Maximum heart rate achieved
    "exang",      # Exercise-induced angina (1 = yes)
    "oldpeak",    # ST depression induced by exercise
    "slope",      # Slope of peak exercise ST segment (0-2)
    "ca",         # Number of major vessels (0-3)
    "thal",       # Thalassemia (0-3)
]

TARGET_COL = "target"   # 1 = heart disease, 0 = no heart disease

# Categorical features that need encoding
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Numerical features that need scaling
NUMERICAL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# Human-readable feature labels (used in SHAP plots)
FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar > 120",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope",
    "ca":       "Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia",
}

# ─────────────────────────────────────────────
# 3. MODEL CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────
# These are the hyperparameter defaults for both models.
# The correction engine may override these after analysis.

RF_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,       # None = grow until pure leaves
    "random_state": 42,
    "n_jobs": -1,            # Use all CPU cores
}

XGB_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

# Default prediction confidence threshold
# Predictions below this are considered "low confidence"
CONFIDENCE_THRESHOLD = 0.60

# Train/test split ratio
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 4. LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logger(name: str = "healthcare_ai") -> logging.Logger:
    """
    Creates and returns a logger that writes to both:
      - The console (so you see output while running)
      - A file  outputs/healthcare_ai.log  (persistent record)

    Args:
        name: Logger name (use module name for easy tracking)

    Returns:
        Configured Logger instance
    """
    # Ensure the outputs directory exists before writing the log file
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)   # Capture everything DEBUG and above

    # ── Console handler ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)   # Show INFO+ on console

    # ── File handler ──
    log_file = os.path.join(LOGS_DIR, "healthcare_ai.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)     # Store DEBUG+ in file

    # Format: timestamp | level | message
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ─────────────────────────────────────────────
# 5. METRICS HELPER
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true : True labels (array-like)
        y_pred : Predicted labels (array-like)
        y_prob : Predicted probabilities for the positive class (optional)

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ROC-AUC only makes sense if we have probability scores
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["roc_auc"] = None

    return metrics


# ─────────────────────────────────────────────
# 6. METRICS PERSISTENCE
# ─────────────────────────────────────────────
def save_metrics(metrics: dict, path: str = METRICS_PATH):
    """
    Save metrics dictionary to a JSON file.
    Appends under a versioned key so history is preserved.

    Args:
        metrics : Dictionary of metric values
        path    : File path for the JSON file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing data if the file already exists
    existing = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

    # Create a version key like "v1", "v2", etc.
    version = f"v{len(existing) + 1}"
    existing[version] = metrics

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    return version


def load_metrics(path: str = METRICS_PATH) -> dict:
    """
    Load all saved metrics from the JSON file.

    Returns:
        Dictionary of versioned metrics, or empty dict if file doesn't exist
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 7. PLOT HELPER
# ─────────────────────────────────────────────
def save_figure(fig, path: str, dpi: int = 150):
    """
    Save a matplotlib figure to disk, creating parent directories as needed.

    Args:
        fig  : matplotlib Figure object
        path : Full file path to save to
        dpi  : Dots per inch (image quality)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)   # Free memory — important in long-running Streamlit apps


# ─────────────────────────────────────────────
# 8. ENSURE DIRECTORIES EXIST
# ─────────────────────────────────────────────
def ensure_dirs():
    """
    Create all project directories if they don't already exist.
    Called at the start of run_pipeline.py.
    """
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# 9. SAMPLE PATIENT (for Streamlit default values)
# ─────────────────────────────────────────────
# A realistic patient record for demo purposes
SAMPLE_PATIENT = {
    "age":      55,
    "sex":      1,       # Male
    "cp":       2,       # Non-anginal pain
    "trestbps": 140,
    "chol":     241,
    "fbs":      0,
    "restecg":  1,
    "thalach":  150,
    "exang":    0,
    "oldpeak":  1.2,
    "slope":    1,
    "ca":       0,
    "thal":     2,
}
