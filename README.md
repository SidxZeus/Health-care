# Self-Correcting Explainable Healthcare AI Assistant

A production-style final-year project demonstrating **Explainable AI + Self-Correction** for heart disease risk prediction.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (trains, explains, detects errors, corrects, retrains)
python run_pipeline.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
Final year/
├── data/                  ← Dataset (auto-downloaded on first run)
├── models/                ← Saved RF + XGBoost models + scaler
├── src/
│   ├── utils.py           ← Config, logging, shared helpers
│   ├── preprocessing.py   ← Data cleaning, encoding, scaling, splitting
│   ├── train_model.py     ← RF + XGBoost training + evaluation plots
│   ├── predict.py         ← Prediction + confidence scoring
│   ├── explain.py         ← SHAP explainability engine
│   ├── error_detection.py ← FP/FN detection + error logging
│   ├── error_analysis.py  ← Root-cause analysis of errors
│   ├── correction_engine.py ← SMOTE + threshold + hyperparameter tuning
│   └── retrain.py         ← Retraining + before/after comparison
├── app/
│   └── streamlit_app.py   ← Full interactive dashboard (5 pages)
├── outputs/               ← Generated plots and error logs
├── run_pipeline.py        ← One-click full pipeline runner
└── requirements.txt
```

---

## Architecture Flow

```
Patient Data → Preprocessing → RF + XGBoost Prediction → Confidence Score
     → SHAP Explanation → Error Detection → Root-Cause Analysis
     → Correction (SMOTE / Threshold / Hyperparameter)
     → Retraining → Improved Model
```

---

## Modules Explained

| Module | Purpose |
|--------|---------|
| `preprocessing.py` | Loads UCI dataset, handles missing values, one-hot encodes categoricals, MinMax scales numericals, stratified 80/20 split |
| `train_model.py` | Trains Random Forest + XGBoost, saves confusion matrix + ROC curve + feature importance plots |
| `predict.py` | Returns prediction (0/1) + confidence score + risk level (Low/Medium/High) |
| `explain.py` | SHAP TreeExplainer for global summary + individual waterfall plots |
| `error_detection.py` | Compares predictions vs actuals, logs FP/FN to CSV |
| `error_analysis.py` | Analyzes class imbalance, feature distributions, confidence patterns |
| `correction_engine.py` | Applies SMOTE, threshold tuning, GridSearchCV |
| `retrain.py` | Retrains with corrections, saves performance comparison chart |

---

## Dashboard Pages

1. **Home** — Overview, architecture, feature list
2. **Predict** — Patient input form → prediction + SHAP explanation
3. **Model Performance** — Confusion matrix, ROC curve, SHAP plots
4. **Error Analysis** — Error log table, FP/FN breakdown, feature distributions
5. **Self-Correction** — Before/after comparison, run full pipeline
6. **About** — Architecture, viva Q&A, ethics, future scope

---

## Key Concepts

### SHAP (SHapley Additive exPlanations)
- Based on cooperative game theory (Shapley values)
- Assigns each feature a contribution score for each prediction
- Positive SHAP value → increases predicted disease risk
- Negative SHAP value → decreases predicted disease risk
- **Waterfall plot**: shows individual patient explanation
- **Summary plot**: shows global feature behavior across all patients

### Self-Correction Loop
1. Model makes predictions on test set
2. Errors (FP/FN) are detected and logged
3. Root-cause analysis identifies triggers:
   - Class imbalance → **SMOTE** oversampling
   - Low confidence errors → **Threshold tuning**
   - High error rate → **Hyperparameter GridSearchCV**
4. Corrections applied, model retrained
5. Performance compared before vs after

### Confidence Score
- `predict_proba()` returns `[P(no disease), P(disease)]`
- We use `P(disease)` as the disease probability
- Confidence = probability assigned to the predicted class
- Predictions below 60% confidence are flagged for doctor review

---

## Medical Disclaimer

> ⚕️ **This tool is for EDUCATIONAL PURPOSES ONLY.**  
> It does NOT provide medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare professional.  
> The AI assistant is a clinical DECISION-SUPPORT tool, not a replacement for doctors.

---

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

---

## Dataset

- **Source**: UCI Heart Disease Dataset (Cleveland Clinic)
- **Auto-downloaded** on first run if not present
- 303 patients, 13 features, binary target

---

## Author

Final Year Project — AI/ML Department  
Self-Correcting Explainable Healthcare AI Assistant
