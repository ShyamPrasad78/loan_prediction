# src/model_training.py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
import pandas as pd
import numpy as np


def create_pipeline():
    """Define base models and stacking ensemble."""
    xgb_clf = XGBClassifier(
        learning_rate=0.1,
        max_depth=4,
        n_estimators=100,
        scale_pos_weight=2.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    lgbm_clf = LGBMClassifier(
        learning_rate=0.1,
        max_depth=4,
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    cat_clf = CatBoostClassifier(
        depth=4,
        learning_rate=0.1,
        iterations=100,
        verbose=0,
        random_state=42
    )
    final_estimator = LogisticRegression(max_iter=1000)

    stack_model = StackingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('lgbm', lgbm_clf),
            ('cat', cat_clf)
        ],
        final_estimator=final_estimator,
        cv=3,
        passthrough=True,
        n_jobs=-1
    )
    return stack_model


def clean_column_names(df):
    """Make column names XGBoost-safe."""
    df = df.copy()
    df.columns = (
        df.columns
        .to_series()
        .str.replace(r'[\[\]<>\s]+', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    return df


def train_and_save_model(X, y, save_dir="models_fast"):
    """Train model and save to disk."""
    print("🏋️ Training fast model...")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Scale numericals
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include='number').columns
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[numerical_cols]),
        columns=numerical_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[numerical_cols]),
        columns=numerical_cols
    )

    # Add back categorical columns (already encoded)
    cat_cols = X_train.columns.difference(numerical_cols)
    X_train_scaled[cat_cols] = X_train[cat_cols].values
    X_test_scaled[cat_cols] = X_test[cat_cols].values

    # Clean column names
    X_train_scaled = clean_column_names(X_train_scaled)
    X_test_scaled = clean_column_names(X_test_scaled)

    # Train model
    model = create_pipeline()
    model.fit(X_train_scaled, y_train)

    # Save everything
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, f"{save_dir}/stacking_model.pkl")
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    joblib.dump({
        'numerical_columns': numerical_cols.tolist(),
        'categorical_columns': cat_cols.tolist(),
        'feature_names': X_train_scaled.columns.tolist()
    }, f"{save_dir}/config.pkl")

    print(f"✅ Model saved to {save_dir}/")
    return X_test_scaled, y_test