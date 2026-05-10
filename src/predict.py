# ============================================================
# predict.py — Prediction & Confidence Scoring
# ============================================================
# PURPOSE:
#   Loads a trained model and makes predictions on new patient
#   data. Returns both the binary prediction (0/1) and a
#   confidence score (probability %) so clinicians can gauge
#   how certain the AI is about its decision.
# ============================================================

import joblib
import numpy as np
import pandas as pd

from src.utils import (
    RF_MODEL_PATH, XGB_MODEL_PATH,
    CONFIDENCE_THRESHOLD, setup_logger
)
from src.preprocessing import preprocess_patient

logger = setup_logger("predict")


def load_model(model_type: str = "xgb"):
    """Load a saved model from disk. model_type: 'rf' or 'xgb'"""
    path = RF_MODEL_PATH if model_type == "rf" else XGB_MODEL_PATH
    try:
        model = joblib.load(path)
        logger.debug(f"Model loaded: {path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Run train_model.py first."
        )


def predict_patient(patient_dict: dict, feature_names: list, model_type: str = "xgb") -> dict:
    """
    Predicts heart disease risk for a single patient.

    Args:
        patient_dict : Raw patient data dict (keys = FEATURE_NAMES)
        feature_names: Training columns after one-hot encoding
        model_type   : 'rf' or 'xgb'

    Returns:
        Result dict with: prediction, confidence, risk_level,
        low_confidence, label, probabilities, model_used
    """
    logger.info(f"Predicting for patient: {patient_dict}")

    model = load_model(model_type)
    X_patient = preprocess_patient(patient_dict, feature_names)

    # Binary prediction (0 = no disease, 1 = disease)
    prediction = int(model.predict(X_patient)[0])

    # Probability scores: predict_proba returns [[P(0), P(1)]]
    probabilities = model.predict_proba(X_patient)[0]
    confidence = float(probabilities[prediction])   # Confidence in the predicted class
    disease_prob = float(probabilities[1])          # Always P(heart disease)

    # 3-tier risk classification beyond binary yes/no
    if disease_prob < 0.35:
        risk_level = "Low"
    elif disease_prob < 0.65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Flag predictions near the decision boundary (uncertain)
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    result = {
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "disease_prob":  round(disease_prob, 4),
        "risk_level":    risk_level,
        "low_confidence": low_confidence,
        "label":         "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "probabilities": {
            "no_disease":    round(float(probabilities[0]), 4),
            "heart_disease": round(float(probabilities[1]), 4),
        },
        "model_used": model_type.upper(),
    }

    logger.info(
        f"Result: {result['label']} | Confidence: {result['confidence']:.1%} | "
        f"Risk: {result['risk_level']}"
        + (" [LOW CONFIDENCE]" if low_confidence else "")
    )
    return result


def predict_batch(X: pd.DataFrame, model_type: str = "xgb") -> dict:
    """
    Batch predictions for the full test set.
    Used by error_detection.py.

    Returns:
        Dict with: predictions, probabilities, confidences
    """
    model = load_model(model_type)
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)
    confidences   = probabilities.max(axis=1)

    logger.info(f"Batch: {len(predictions)} samples | Avg confidence: {confidences.mean():.3f}")
    return {
        "predictions":   predictions,
        "probabilities": probabilities,
        "confidences":   confidences,
    }


def confidence_band_summary(confidences: np.ndarray) -> dict:
    """Returns counts/percentages of predictions in high/medium/low confidence bands."""
    total = len(confidences)
    high   = (confidences >= 0.80).sum()
    medium = ((confidences >= 0.60) & (confidences < 0.80)).sum()
    low    = (confidences < 0.60).sum()
    return {
        "total":  total,
        "high":   {"count": int(high),   "pct": round(high / total, 3)},
        "medium": {"count": int(medium), "pct": round(medium / total, 3)},
        "low":    {"count": int(low),    "pct": round(low / total, 3)},
    }


if __name__ == "__main__":
    from src.utils import SAMPLE_PATIENT
    from src.preprocessing import preprocess
    _, _, _, _, feature_names = preprocess()
    result = predict_patient(SAMPLE_PATIENT, feature_names)
    print(f"\n  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Risk Level : {result['risk_level']}")
