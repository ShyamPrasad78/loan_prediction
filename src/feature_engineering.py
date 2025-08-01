# src/feature_engineering.py
import pandas as pd
import numpy as np

def load_data(file_path, n_rows=None):
    """Load and return raw data."""
    print("📂 Loading data...")
    cols = [
        'loan_amnt', 'term', 'int_rate', 'emp_length', 'home_ownership',
        'annual_inc', 'verification_status', 'purpose', 'dti', 'delinq_2yrs',
        'open_acc', 'revol_util', 'total_acc', 'application_type', 'loan_status'
    ]
    df = pd.read_csv(file_path, usecols=cols, low_memory=False, nrows=n_rows)
    return df

def feature_engineering(df):
    """Apply feature engineering."""
    print("🔧 Feature engineering...")
    df = df.copy()

    # Basic ratios
    df['income_to_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1)
    df['dti_to_inc_ratio'] = df['dti'] / (df['annual_inc'] + 1)
    df['log_annual_inc'] = np.log1p(df['annual_inc'])

    # Extract term in months
    df['term_months'] = pd.to_numeric(
        df['term'].str.extract(r'(\d+)')[0], errors='coerce'
    ).astype('Int64')
    df['term_months'].fillna(df['term_months'].median(), inplace=True)

    # Interaction features
    df['dti_x_income'] = df['dti'] * df['annual_inc']
    df['revol_util_x_open_acc'] = df['revol_util'] * df['open_acc']

    return df

def clean_data(df):
    """Clean and filter target."""
    print("🧹 Cleaning data...")
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # Fill missing
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def encode_categoricals(df, cat_cols=None):
    """One-hot encode categorical variables."""
    if cat_cols is None:
        cat_cols = [
            'term', 'emp_length', 'home_ownership', 'verification_status',
            'purpose', 'application_type'
        ]
    print("🏷️ Encoding categoricals...")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df