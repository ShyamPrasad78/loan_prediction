# main.py
import joblib
from src.feature_engineering import load_data, feature_engineering, clean_data, encode_categoricals
from src.model_training import train_and_save_model
from src.evaluation import evaluate_model
import pandas as pd

# =============================
# CONFIG
# =============================
DATA_PATH = "E:/data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
SAMPLE_ROWS = 200000  # Set to None for full data

# =============================
# MAIN PIPELINE
# =============================
if __name__ == "__main__":
    # 1. Load and preprocess
    df = load_data(DATA_PATH, n_rows=SAMPLE_ROWS)
    df = feature_engineering(df)
    df = clean_data(df)
    df = encode_categoricals(df)

    # 2. Split features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # 3. Train model and save
    X_test_scaled, y_test = train_and_save_model(X, y, save_dir="models_fast")

    # 4. Load model and evaluate
    from sklearn.ensemble import StackingClassifier
    model = joblib.load("models_fast/stacking_model.pkl")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 5. Evaluate
    results = evaluate_model(
        y_test=y_test,
        y_proba=y_proba,
        X_test_flat=X_test_scaled,  # Scaled test features (DataFrame)
        model=model  # Trained stacking model
    )


# Load model
model = joblib.load("models_fast/stacking_model.pkl")

# Predict
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate with SHAP
results = evaluate_model(
    y_test=y_test,
    y_proba=y_proba,
    X_test_flat=X_test_scaled,   # <-- Pass scaled test DataFrame
    model=model                  # <-- Pass trained model
)