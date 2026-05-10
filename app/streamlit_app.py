import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark medical theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0e1a; color: #e0e6f0; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 100%); }
.metric-card {
    background: linear-gradient(135deg, #1a2744, #162036);
    border: 1px solid #2a3f6f;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.3rem;
}
.prediction-box-disease {
    background: linear-gradient(135deg, #4a0a0a, #6b1414);
    border: 2px solid #ff4444;
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.prediction-box-safe {
    background: linear-gradient(135deg, #0a2a1a, #0d3d22);
    border: 2px solid #00c853;
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.disclaimer {
    background: #1a1a0a; border: 1px solid #b8860b;
    border-radius: 8px; padding: 1rem;
    color: #ffd700; font-size: 0.85rem;
}
.section-header {
    font-size: 1.4rem; font-weight: 700;
    color: #4fc3f7; border-bottom: 2px solid #2a3f6f;
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}
div[data-testid="metric-container"] {
    background: #1a2744; border-radius: 10px;
    border: 1px solid #2a3f6f; padding: 0.5rem 1rem;
}
.stButton>button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 2rem;
    width: 100%; transition: all 0.3s;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(21,101,192,0.4); }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    from src.utils import RF_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH
    models = {}
    for key, path in [("rf", RF_MODEL_PATH), ("xgb", XGB_MODEL_PATH)]:
        if os.path.exists(path):
            models[key] = joblib.load(path)
    if os.path.exists(SCALER_PATH):
        models["scaler"] = joblib.load(SCALER_PATH)
    return models

@st.cache_data(show_spinner=False)
def get_feature_names():
    try:
        from src.preprocessing import preprocess
        _, _, _, _, fn = preprocess()
        return fn
    except Exception:
        return []

def models_exist():
    from src.utils import XGB_MODEL_PATH
    return os.path.exists(XGB_MODEL_PATH)


# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 Healthcare AI")
    st.markdown("**Self-Correcting Explainable Assistant**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict", "📊 Model Performance",
         "🔍 Error Analysis", "🔄 Self-Correction", "📚 About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
    ⚕️ <b>Medical Disclaimer</b><br>
    This tool is for educational purposes only.<br>
    It does NOT replace professional medical advice.<br>
    Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🫀 Self-Correcting Explainable Healthcare AI")
    st.markdown("### *An Intelligent Clinical Decision-Support Assistant*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algorithm", "XGBoost + RF")
    with col2:
        st.metric("Explainability", "SHAP")
    with col3:
        st.metric("Self-Correction", "SMOTE + Tuning")
    with col4:
        st.metric("Dataset", "UCI Heart Disease")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### How It Works")
        st.markdown("""
        ```
        Patient Data Input
              ↓
        Data Preprocessing
              ↓
        ML Prediction (RF + XGBoost)
              ↓
        Confidence Scoring
              ↓
        SHAP Explanation
              ↓
        Error Detection
              ↓
        Root-Cause Analysis
              ↓
        Self-Correction (SMOTE)
              ↓
        Retraining → Improved Model
        ```
        """)
    with col_b:
        st.markdown("### Core Features")
        features = [
            ("🤖", "ML Prediction", "Random Forest + XGBoost ensemble"),
            ("🔍", "Explainability", "SHAP values for every prediction"),
            ("⚠️", "Error Detection", "FP/FN identification and logging"),
            ("📊", "Error Analysis", "Root-cause pattern detection"),
            ("🔧", "Self-Correction", "SMOTE + hyperparameter tuning"),
            ("🔄", "Retraining", "Iterative model improvement"),
        ]
        for icon, title, desc in features:
            st.markdown(f"**{icon} {title}** — {desc}")

    if not models_exist():
        st.warning("⚠️ Models not yet trained. Run `python run_pipeline.py` first.")
    else:
        st.success("✅ Models loaded and ready. Use the sidebar to navigate.")


# ════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("## 🔮 Patient Risk Prediction")

    if not models_exist():
        st.error("Models not found. Please run: `python run_pipeline.py`")
        st.stop()

    from src.utils import SAMPLE_PATIENT, FEATURE_LABELS
    from src.predict import predict_patient
    from src.explain import get_explainer, compute_shap_values, plot_shap_waterfall, generate_text_explanation
    from src.preprocessing import preprocess

    st.markdown("### Enter Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age (years)", 20, 80, SAMPLE_PATIENT["age"])
        sex      = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp       = st.selectbox("Chest Pain Type", [(0,"Typical Angina"),(1,"Atypical Angina"),(2,"Non-Anginal"),(3,"Asymptomatic")], format_func=lambda x: x[1])[0]
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, SAMPLE_PATIENT["trestbps"])
        chol     = st.slider("Cholesterol (mg/dL)", 100, 600, SAMPLE_PATIENT["chol"])

    with col2:
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])[0]
        restecg  = st.selectbox("Resting ECG", [(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")], format_func=lambda x: x[1])[0]
        thalach  = st.slider("Max Heart Rate Achieved", 60, 220, SAMPLE_PATIENT["thalach"])
        exang    = st.selectbox("Exercise-Induced Angina", [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])[0]
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, SAMPLE_PATIENT["oldpeak"], step=0.1)

    with col3:
        slope    = st.selectbox("ST Slope", [(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")], format_func=lambda x: x[1])[0]
        ca       = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
        thal     = st.selectbox("Thalassemia", [(0,"Normal"),(1,"Fixed Defect"),(2,"Reversable Defect"),(3,"Unknown")], format_func=lambda x: x[1])[0]
        model_choice = st.radio("Model", ["xgb", "rf"], format_func=lambda x: "XGBoost" if x=="xgb" else "Random Forest")

    patient = dict(
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope, ca=ca, thal=thal
    )

    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
        with st.spinner("Analysing patient data..."):
            try:
                X_train, X_test, y_train, y_test, feature_names = preprocess()
                result = predict_patient(patient, feature_names, model_type=model_choice)

                st.markdown("---")
                st.markdown("### Prediction Result")

                if result["prediction"] == 1:
                    st.markdown(f"""
                    <div class='prediction-box-disease'>
                        <h2 style='color:#ff4444'>⚠️ {result['label']}</h2>
                        <h3 style='color:#ffaaaa'>Risk Level: {result['risk_level']}</h3>
                        <p style='color:#ffcccc'>Confidence: {result['confidence']:.1%}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='prediction-box-safe'>
                        <h2 style='color:#00c853'>✅ {result['label']}</h2>
                        <h3 style='color:#aaffcc'>Risk Level: {result['risk_level']}</h3>
                        <p style='color:#ccffdd'>Confidence: {result['confidence']:.1%}</p>
                    </div>""", unsafe_allow_html=True)

                if result["low_confidence"]:
                    st.warning("⚠️ Low confidence prediction — please consult a physician.")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("P(Heart Disease)", f"{result['probabilities']['heart_disease']:.1%}")
                col_b.metric("P(No Disease)", f"{result['probabilities']['no_disease']:.1%}")
                col_c.metric("Model Used", result["model_used"])

                # SHAP Explanation
                st.markdown("---")
                st.markdown("### 🔍 SHAP Explanation — Why this prediction?")
                with st.spinner("Computing SHAP values..."):
                    try:
                        model = joblib.load(__import__('src.utils', fromlist=['XGB_MODEL_PATH']).XGB_MODEL_PATH if model_choice == "xgb" else __import__('src.utils', fromlist=['RF_MODEL_PATH']).RF_MODEL_PATH)
                        from src.preprocessing import preprocess_patient
                        import shap
                        X_bg = X_train.sample(n=min(100, len(X_train)), random_state=42)
                        explainer = get_explainer(model, X_bg)
                        X_pat = preprocess_patient(patient, feature_names)
                        shap_vals = compute_shap_values(explainer, X_pat)
                        text_exp = generate_text_explanation(shap_vals[0], feature_names, result["prediction"])
                        st.code(text_exp)
                        fig_wf = plot_shap_waterfall(explainer, X_pat, "Patient")
                        st.pyplot(fig_wf)
                    except Exception as e:
                        st.info(f"SHAP visualization: {e}")

                st.warning("⚕️ This is an AI assistant. Please consult a qualified doctor for diagnosis.")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")

    from src.utils import OUTPUTS_DIR, load_metrics, METRICS_PATH
    metrics_all = load_metrics()

    if not metrics_all:
        st.warning("No metrics found. Run `python run_pipeline.py` first.")
        st.stop()

    latest = metrics_all[sorted(metrics_all.keys())[-1]]

    st.markdown("### Performance Metrics")
    col1, col2 = st.columns(2)

    for col, (mname, mdata) in zip([col1, col2], [
        ("Random Forest", latest.get("random_forest", {})),
        ("XGBoost",       latest.get("xgboost", {}))
    ]):
        with col:
            st.markdown(f"#### {mname}")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy",  f"{mdata.get('accuracy', 'N/A')}")
            m2.metric("F1 Score",  f"{mdata.get('f1', 'N/A')}")
            m3, m4 = st.columns(2)
            m3.metric("Precision", f"{mdata.get('precision', 'N/A')}")
            m4.metric("Recall",    f"{mdata.get('recall', 'N/A')}")
            st.metric("ROC-AUC", f"{mdata.get('roc_auc', 'N/A')}")

    st.markdown("---")
    for fname, title in [
        ("confusion_matrix.png",  "Confusion Matrix"),
        ("roc_curve.png",         "ROC Curves"),
        ("feature_importance.png","Feature Importance"),
        ("shap_summary.png",      "SHAP Summary"),
        ("shap_bar.png",          "SHAP Bar Chart"),
    ]:
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f"#### {title}")
            st.image(fpath)


# ════════════════════════════════════════
# PAGE: ERROR ANALYSIS
# ════════════════════════════════════════
elif page == "🔍 Error Analysis":
    st.markdown("## 🔍 Error Detection & Analysis")

    from src.utils import ERROR_LOG_PATH, OUTPUTS_DIR
    from src.error_detection import load_error_log

    error_df = load_error_log()

    if error_df.empty:
        st.warning("No error log found. Run `python run_pipeline.py` first.")
        st.stop()

    total   = len(error_df)
    fp_count = (error_df.get("error_type","") == "False Positive (FP)").sum() if "error_type" in error_df.columns else 0
    fn_count = (error_df.get("error_type","") == "False Negative (FN)").sum() if "error_type" in error_df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Errors Logged", total)
    col2.metric("False Positives (FP)", fp_count, help="Predicted disease, was healthy")
    col3.metric("False Negatives (FN)", fn_count, help="Missed disease — more critical!")

    st.markdown("---")
    st.markdown("### Error Log Table")
    st.dataframe(error_df.tail(50), use_container_width=True)

    for fname, title in [
        ("error_breakdown.png",    "Error Type Breakdown"),
        ("error_feature_dist.png", "Feature Distribution: Errors vs Correct"),
    ]:
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f"#### {title}")
            st.image(fpath)

    st.markdown("### Understanding Error Types")
    col_a, col_b = st.columns(2)
    with col_a:
        st.error("**False Positive (FP)** — Model predicted DISEASE but patient is HEALTHY\n\nRisk: Unnecessary tests, patient anxiety")
    with col_b:
        st.warning("**False Negative (FN)** — Model predicted HEALTHY but patient HAS DISEASE\n\nRisk: Missed diagnosis — more dangerous!")


# ════════════════════════════════════════
# PAGE: SELF-CORRECTION
# ════════════════════════════════════════
elif page == "🔄 Self-Correction":
    st.markdown("## 🔄 Self-Correction Engine")

    from src.utils import PERF_COMPARE_PATH, OUTPUTS_DIR, load_metrics

    st.markdown("""
    The self-correction engine:
    1. **Analyzes** error patterns from the error log
    2. **Selects** correction strategies (SMOTE, threshold tuning, hyperparameter optimization)
    3. **Retrains** the model with corrections applied
    4. **Compares** performance before and after
    """)

    metrics_all = load_metrics()
    if len(metrics_all) >= 2:
        versions = sorted(metrics_all.keys())
        v1, v2 = versions[-2], versions[-1]
        m1, m2 = metrics_all[v1], metrics_all[v2]

        st.markdown("### Performance: Before vs After Correction")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{v1} (Before)**")
            xgb1 = m1.get("xgboost", {})
            st.metric("XGBoost Accuracy", xgb1.get("accuracy", "N/A"))
            st.metric("XGBoost F1",       xgb1.get("f1", "N/A"))
            st.metric("XGBoost ROC-AUC",  xgb1.get("roc_auc", "N/A"))
        with col2:
            st.markdown(f"**{v2} (After)**")
            xgb2 = m2.get("xgboost", {})
            acc_delta = round((xgb2.get("accuracy",0) or 0) - (xgb1.get("accuracy",0) or 0), 4)
            st.metric("XGBoost Accuracy", xgb2.get("accuracy","N/A"), delta=acc_delta)
            st.metric("XGBoost F1",       xgb2.get("f1","N/A"),
                      delta=round((xgb2.get("f1",0) or 0) - (xgb1.get("f1",0) or 0), 4))
            st.metric("XGBoost ROC-AUC",  xgb2.get("roc_auc","N/A"),
                      delta=round((xgb2.get("roc_auc",0) or 0) - (xgb1.get("roc_auc",0) or 0), 4))

        corrections = m2.get("corrections", [])
        if corrections:
            st.info(f"✅ Corrections applied: {', '.join(corrections)}")

    if os.path.exists(PERF_COMPARE_PATH):
        st.markdown("### Performance Comparison Chart")
        st.image(PERF_COMPARE_PATH)

    st.markdown("---")
    st.markdown("### Run Self-Correction Now")
    if st.button("🔄 Run Full Pipeline (Train → Detect → Correct → Retrain)", use_container_width=True):
        with st.spinner("Running full pipeline... (this may take 1-2 minutes)"):
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "run_pipeline.py"],
                    capture_output=True, text=True, timeout=300,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                if result.returncode == 0:
                    st.success("✅ Pipeline completed successfully!")
                    st.code(result.stdout[-3000:])
                    st.cache_resource.clear()
                    st.cache_data.clear()
                else:
                    st.error("Pipeline encountered an error:")
                    st.code(result.stderr[-2000:])
            except Exception as e:
                st.error(f"Failed to run pipeline: {e}")
                st.info("Run manually: `python run_pipeline.py`")


# ════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════
elif page == "📚 About":
    st.markdown("## 📚 About This Project")

    tab1, tab2, tab3 = st.tabs(["📖 Architecture", "🎓 Viva Q&A", "⚖️ Ethics & Future"])

    with tab1:
        st.markdown("### System Architecture")
        st.code("""
Patient Data
    ↓ preprocessing.py
Data Cleaning + Feature Encoding + Scaling
    ↓ train_model.py
Random Forest + XGBoost Training
    ↓ predict.py
Prediction + Confidence Score
    ↓ explain.py
SHAP Explanations (Global + Local)
    ↓ error_detection.py
FP/FN Detection + Error Logging
    ↓ error_analysis.py
Root Cause Analysis + Correction Triggers
    ↓ correction_engine.py
SMOTE + Threshold Tuning + Hyperparameter Search
    ↓ retrain.py
Retrained Model + Performance Comparison
        """)

        st.markdown("### Tech Stack")
        tech = {
            "pandas / numpy": "Data manipulation and numerical computing",
            "scikit-learn":   "Random Forest, preprocessing, evaluation metrics",
            "XGBoost":        "Gradient boosting classifier",
            "SHAP":           "Explainable AI (SHapley Additive exPlanations)",
            "imbalanced-learn": "SMOTE oversampling for class imbalance",
            "Streamlit":      "Interactive web dashboard",
            "Joblib":         "Model serialization and persistence",
            "Matplotlib/Seaborn": "Visualization and plotting",
        }
        st.table(pd.DataFrame({"Library": list(tech.keys()), "Purpose": list(tech.values())}))

    with tab2:
        st.markdown("### Viva Questions & Answers")
        qas = [
            ("What is SHAP?",
             "SHAP (SHapley Additive exPlanations) is a game-theory-based method that explains ML predictions. For each prediction, it assigns each feature a value showing how much it pushed the output above or below the baseline average prediction. Positive SHAP = increases disease risk; Negative SHAP = decreases it."),
            ("Why use two models (RF and XGBoost)?",
             "Random Forest uses bagging (parallel trees on random data subsets). XGBoost uses boosting (sequential trees correcting each other). Comparing both helps identify which paradigm works better for our data and provides a more robust system."),
            ("What is SMOTE and why is it used?",
             "SMOTE (Synthetic Minority Oversampling Technique) generates synthetic data points for the minority class by interpolating between existing minority samples. It prevents the model from being biased toward predicting the majority class — critical in medical data where disease cases may be fewer."),
            ("What makes this AI 'self-correcting'?",
             "The system detects its own errors by comparing predictions to actual outcomes. It then analyzes WHY errors occurred (class imbalance, low confidence, feature patterns) and applies targeted corrections (SMOTE, threshold tuning, hyperparameter search), then retrains itself — forming an autonomous improvement loop."),
            ("What is a False Negative and why is it dangerous in healthcare?",
             "A False Negative means the model predicted 'healthy' but the patient actually has heart disease. This is a missed diagnosis, which is far more dangerous than a False Positive because the patient receives no treatment. This is why we monitor FN rate carefully and may tune the threshold to reduce them."),
            ("How does confidence scoring help?",
             "predict_proba() returns P(disease). We use this as confidence. Low confidence (near 0.5) means the model is uncertain — these cases are flagged for mandatory doctor review rather than relying solely on the AI decision."),
            ("What are the ethical considerations?",
             "1) The AI must not replace doctors — it assists. 2) False negatives must be minimized. 3) Model should be fair across demographics (age, sex). 4) Predictions must be explainable (SHAP). 5) Patient data privacy must be protected. 6) The system includes a mandatory medical disclaimer."),
            ("What is the purpose of train/test split?",
             "We train on 80% of data and evaluate on the remaining 20% that the model has NEVER seen. This simulates real-world deployment and gives an honest estimate of generalization performance. Using all data for training would give inflated accuracy."),
        ]
        for q, a in qas:
            with st.expander(f"❓ {q}"):
                st.markdown(a)

    with tab3:
        st.markdown("### Ethical Considerations")
        st.markdown("""
        - **Non-replacement**: This AI assists clinicians — it does NOT replace doctors
        - **Transparency**: Every prediction has a SHAP explanation (no black box)
        - **Fairness**: Model should be validated across age groups and sexes
        - **Privacy**: Patient data must be anonymized and not stored without consent
        - **Accountability**: Errors are logged; the system admits uncertainty
        - **Mandatory disclaimer**: Displayed on every page
        """)

        st.markdown("### Future Scope")
        future = [
            ("🏥", "Multi-disease prediction", "Extend to diabetes, stroke, kidney disease"),
            ("🔗", "EHR Integration", "Connect to hospital Electronic Health Records"),
            ("🌐", "Federated Learning", "Train across hospitals without sharing patient data"),
            ("🧬", "Genetic Features", "Include genomic markers for precision medicine"),
            ("📱", "Mobile App", "Patient-facing risk monitoring on smartphones"),
            ("🤖", "LLM Reports", "Use GPT/Gemini to generate natural language reports"),
            ("⏱️", "Real-time Monitoring", "Continuous patient vitals monitoring with alerts"),
        ]
        for icon, title, desc in future:
            st.markdown(f"**{icon} {title}** — {desc}")

        st.markdown("---")
        st.markdown("### Dataset Information")
        st.info("""
        **Source**: UCI Heart Disease Dataset (Cleveland Clinic Foundation)
        **Samples**: ~303 patients
        **Features**: 13 clinical attributes
        **Target**: Binary (0 = No Heart Disease, 1 = Heart Disease)
        **Citation**: Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.
        """)
