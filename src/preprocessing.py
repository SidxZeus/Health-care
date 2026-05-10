# ============================================================
# preprocessing.py — Data Loading & Preparation
# ============================================================
# PURPOSE:
#   Loads the heart disease dataset, cleans it, encodes
#   categorical variables, scales numerical features,
#   and returns train/test splits ready for model training.
#
# LOGIC FLOW:
#   1. Download dataset if not present
#   2. Load CSV into a pandas DataFrame
#   3. Handle missing/invalid values
#   4. Encode categorical features
#   5. Scale numerical features with MinMaxScaler
#   6. Split into X_train, X_test, y_train, y_test
#   7. Save the fitted scaler for later use (predict.py)
#
# EXPECTED OUTPUT:
#   Tuple: (X_train, X_test, y_train, y_test, feature_names)
# ============================================================

import os
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import project-wide configuration
from src.utils import (
    DATASET_PATH, DATASET_URL, SCALER_PATH,
    FEATURE_NAMES, TARGET_COL,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_SEED, setup_logger
)

# Initialize logger for this module
logger = setup_logger("preprocessing")


# ─────────────────────────────────────────────
# STEP 1: DATASET DOWNLOAD
# ─────────────────────────────────────────────
def download_dataset():
    """
    Downloads the heart disease CSV from a public GitHub URL
    if it doesn't already exist in the data/ directory.

    WHY AUTO-DOWNLOAD:
        Students should be able to run the project immediately
        without manually sourcing and placing data files.
    """
    if os.path.exists(DATASET_PATH):
        logger.info(f"Dataset already exists at: {DATASET_PATH}")
        return

    logger.info(f"Downloading dataset from: {DATASET_URL}")
    try:
        # Stream=True allows downloading large files in chunks
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()   # Raises an error for HTTP 4xx/5xx

        # Write the downloaded content to disk
        with open(DATASET_PATH, "wb") as f:
            f.write(response.content)

        logger.info(f"Dataset saved to: {DATASET_PATH}")

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(
            "Could not download dataset. Please manually place 'heart.csv' "
            f"in the data/ folder. Error: {e}"
        )


# ─────────────────────────────────────────────
# STEP 2: LOAD & VALIDATE
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Loads the CSV into a pandas DataFrame and performs basic validation.

    Returns:
        df: Raw DataFrame with all rows and columns
    """
    download_dataset()

    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Validate expected columns exist
    expected_cols = FEATURE_NAMES + [TARGET_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


# ─────────────────────────────────────────────
# STEP 3: HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values using appropriate strategies:
      - Numerical columns → median (robust to outliers)
      - Categorical columns → mode (most frequent value)

    WHY MEDIAN for numerical:
        Mean is distorted by outliers. In medical data,
        a single extreme cholesterol reading would skew the mean.
        Median is more representative of the 'typical' patient.

    Args:
        df: Raw DataFrame (may contain NaN)

    Returns:
        df: DataFrame with no missing values
    """
    missing_before = df.isnull().sum().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")

    # Fill numerical features with column median
    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"  Filled '{col}' with median={median_val:.2f}")

    # Fill categorical features with column mode (most common value)
    for col in CATEGORICAL_FEATURES + ["sex", "fbs", "exang", "ca", "thal"]:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug(f"  Filled '{col}' with mode={mode_val}")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after cleaning: {missing_after}")

    return df


# ─────────────────────────────────────────────
# STEP 4: ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts categorical integer codes into one-hot encoded columns.

    WHY ONE-HOT ENCODING (not label encoding) for multi-class categoricals:
        Features like 'cp' (chest pain type: 0,1,2,3) are nominal —
        there's no natural ordering. Label encoding would imply
        cp=3 > cp=2 > cp=1, which is medically incorrect.
        One-hot encoding treats each value as an independent binary flag.

    NOTE: We use pd.get_dummies which automatically handles the number
    of unique categories without needing to fit a separate encoder.
    drop_first=True prevents the "dummy variable trap" (multicollinearity).

    Args:
        df: DataFrame after missing value handling

    Returns:
        df: DataFrame with one-hot encoded categorical columns
    """
    logger.info(f"Encoding categorical features: {CATEGORICAL_FEATURES}")

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,     # Drops one redundant column per feature
        dtype=int            # Use integers (0/1) not booleans
    )

    logger.info(f"Shape after encoding: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 5: SCALE NUMERICAL FEATURES
# ─────────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit_new: bool = True
) -> tuple:
    """
    Normalizes numerical features to the [0, 1] range using MinMaxScaler.

    WHY MINMAX (not StandardScaler):
        MinMaxScaler maps all values to [0,1], preserving relative distances.
        Medical values like blood pressure (80-180) and cholesterol (100-600)
        are on very different scales. Without scaling, the model would give
        far more weight to cholesterol simply because its numbers are larger.

    WHY FIT ONLY ON TRAINING DATA:
        We fit the scaler on X_train only, then TRANSFORM both train and test.
        If we fit on the full dataset, information from the test set would
        "leak" into the training process — this inflates performance metrics.

    Args:
        X_train   : Training features DataFrame
        X_test    : Testing features DataFrame
        fit_new   : If True, fit a new scaler; if False, load saved scaler

    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """
    # Find which numerical columns actually exist after encoding
    numerical_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    logger.info(f"Scaling numerical columns: {numerical_cols}")

    if fit_new:
        scaler = MinMaxScaler()
        # Fit: compute min/max from training data only
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    else:
        # Load previously fitted scaler (used during prediction)
        scaler = joblib.load(SCALER_PATH)
        X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])

    # Apply the SAME scaler (fitted on train) to the test data
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Save the scaler so predict.py can use it for new patients
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to: {SCALER_PATH}")

    return X_train, X_test, scaler


# ─────────────────────────────────────────────
# STEP 6: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame) -> tuple:
    """
    Separates features (X) from the target (y), then splits into
    training and testing sets.

    WHY STRATIFIED SPLIT:
        Heart disease data can be slightly imbalanced.
        stratify=y ensures both train and test sets have the
        same proportion of positive/negative cases as the full dataset.
        Without this, by chance, the test set might have mostly one class.

    Args:
        df: Fully preprocessed DataFrame

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    logger.info(
        f"Split → Train: {len(X_train)} rows | Test: {len(X_test)} rows"
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────
def preprocess() -> tuple:
    """
    Runs the full preprocessing pipeline in one call.

    This is the function called by train_model.py and run_pipeline.py.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
        where feature_names is the list of column names after encoding
    """
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 50)

    # Step 1: Load raw data
    df = load_data()

    # Step 2: Clean missing values
    df = handle_missing_values(df)

    # Step 3: Encode categorical features
    df = encode_features(df)

    # Step 4: Split into train/test before scaling
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5: Scale numerical features
    # Important: scale AFTER split to prevent data leakage
    X_train, X_test, scaler = scale_features(X_train, X_test, fit_new=True)

    # Capture feature names after encoding (used by SHAP)
    feature_names = list(X_train.columns)
    logger.info(f"Final feature count: {len(feature_names)}")
    logger.info("Preprocessing complete ✓")

    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────
# PREPROCESS A SINGLE NEW PATIENT
# ─────────────────────────────────────────────
def preprocess_patient(patient_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    Prepares a single patient's raw data for prediction.

    This replicates the encoding + scaling steps but for one row,
    using the already-fitted scaler from disk.

    Args:
        patient_dict : Dict with raw patient values (matching FEATURE_NAMES)
        feature_names: Column names from training (after encoding)

    Returns:
        DataFrame with a single row, ready for model.predict()
    """
    # Create a one-row DataFrame from the patient dict
    df_patient = pd.DataFrame([patient_dict])

    # Apply the same encoding steps as training data
    df_patient = pd.get_dummies(
        df_patient,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=int
    )

    # Reindex to match training columns exactly (fills 0 for missing dummies)
    # This handles cases where the patient's categorical values don't cover
    # all the dummy columns that appeared during training
    df_patient = df_patient.reindex(columns=feature_names, fill_value=0)

    # Load the saved scaler and apply it
    scaler = joblib.load(SCALER_PATH)
    numerical_cols = [c for c in NUMERICAL_FEATURES if c in df_patient.columns]
    df_patient[numerical_cols] = scaler.transform(df_patient[numerical_cols])

    return df_patient


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Run preprocessing and print a summary
    X_train, X_test, y_train, y_test, features = preprocess()
    print(f"\n✓ Preprocessing complete")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Feature count    : {len(features)}")
    print(f"  Features         : {features}")
