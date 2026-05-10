# ============================================================
# run_pipeline.py — End-to-End Pipeline Runner
# ============================================================
# PURPOSE:
#   One-click script to run the ENTIRE system:
#   preprocess → train → explain → detect errors →
#   analyze → correct → retrain → compare
#
# HOW TO RUN:
#   python run_pipeline.py
#
# This is the entry point for full system execution.
# ============================================================

import sys
import os
import time

# Add project root to path so imports work from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import ensure_dirs, setup_logger, load_metrics
from src.preprocessing import preprocess
from src.train_model import train_pipeline
from src.predict import predict_batch
from src.explain import run_explanation_pipeline
from src.error_detection import detect_errors, get_error_summary_string
from src.error_analysis import generate_analysis_report
from src.correction_engine import apply_corrections
from src.retrain import retrain_pipeline

logger = setup_logger("pipeline")


def print_header(title: str):
    """Prints a formatted section header."""
    width = 55
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_full_pipeline():
    """
    Executes the complete self-correcting AI pipeline.

    PIPELINE STAGES:
      1.  Setup
      2.  Train initial models
      3.  SHAP explainability
      4.  Error detection
      5.  Error analysis
      6.  Correction strategies
      7.  Retraining
      8.  Final comparison
    """
    start_time = time.time()

    print_header("SELF-CORRECTING HEALTHCARE AI — PIPELINE START")

    # ── Stage 0: Setup ──
    print_header("Stage 0: Environment Setup")
    ensure_dirs()
    print("✓ Directories ready")

    # ── Stage 1: Train Initial Models ──
    print_header("Stage 1: Training Initial Models")
    train_results = train_pipeline()
    X_test        = train_results["X_test"]
    y_test        = train_results["y_test"]
    feature_names = train_results["feature_names"]
    rf_model      = train_results["rf_model"]
    xgb_model     = train_results["xgb_model"]

    print(f"✓ Random Forest Accuracy : {train_results['rf_metrics']['accuracy']:.4f}")
    print(f"✓ XGBoost Accuracy       : {train_results['xgb_metrics']['accuracy']:.4f}")

    # ── Stage 2: SHAP Explanations ──
    print_header("Stage 2: Generating SHAP Explanations")

    # Re-run preprocessing to get X_train for SHAP background
    X_train, _, y_train, _, _ = preprocess()

    explain_results = run_explanation_pipeline(
        xgb_model, X_train, X_test, feature_names
    )
    print(f"✓ SHAP summary plot saved: outputs/shap_summary.png")

    # ── Stage 3: Error Detection ──
    print_header("Stage 3: Detecting Prediction Errors")
    batch = predict_batch(X_test, model_type="xgb")

    error_results = detect_errors(
        X_test, y_test,
        batch["predictions"],
        batch["confidences"],
        batch["probabilities"],
    )
    print(get_error_summary_string(error_results["stats"]))

    # ── Stage 4: Error Analysis ──
    print_header("Stage 4: Analyzing Error Patterns")
    correct_df = error_results["all_df"][~error_results["all_df"]["is_error"]]

    analysis_report = generate_analysis_report(
        y_train,
        error_results["error_df"],
        correct_df,
        error_results["stats"],
        feature_names,
    )
    print(f"✓ {analysis_report['summary']}")
    print(f"✓ Correction triggers: {analysis_report['correction_triggers']}")

    # ── Stage 5: Apply Corrections ──
    print_header("Stage 5: Applying Self-Correction Strategies")
    correction_result = apply_corrections(
        X_train, y_train,
        X_test, y_test,
        correction_triggers=analysis_report["correction_triggers"],
        current_model=xgb_model,
    )
    print(f"✓ Corrections applied: {correction_result['corrections_applied']}")
    print(f"  Training samples after correction: {len(correction_result['X_train_corrected'])}")

    # ── Stage 6: Retrain ──
    print_header("Stage 6: Retraining with Corrections")
    retrain_results = retrain_pipeline(
        X_train_corrected=correction_result["X_train_corrected"],
        y_train_corrected=correction_result["y_train_corrected"],
        X_test=X_test,
        y_test=y_test,
        rf_params=correction_result["rf_params"],
        xgb_params=correction_result["xgb_params"],
        corrections_applied=correction_result["corrections_applied"],
    )

    # ── Stage 7: Final Report ──
    print_header("Stage 7: Final Report")
    cmp = retrain_results["comparison"]
    xgb_delta = retrain_results["acc_delta"]
    rf_delta  = cmp["rf"]["delta"]["accuracy"] or 0

    print(f"  Random Forest:  {cmp['rf']['old'].get('accuracy','N/A')} → {cmp['rf']['new']['accuracy']}  (Δ {rf_delta:+.4f})")
    print(f"  XGBoost:        {cmp['xgb']['old'].get('accuracy','N/A')} → {cmp['xgb']['new']['accuracy']} (Δ {xgb_delta:+.4f})")
    print(f"\n  Result: {'🎉 IMPROVEMENT ACHIEVED' if retrain_results['improved'] else '⚠ No significant improvement'}")
    print(f"\n  Performance comparison chart: outputs/performance_comparison.png")

    elapsed = time.time() - start_time
    print_header(f"PIPELINE COMPLETE — {elapsed:.1f} seconds")
    print("\n  Next step: Launch the dashboard")
    print("  → streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    run_full_pipeline()
