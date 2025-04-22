import pandas as pd
from sklearn.impute import SimpleImputer


def detect_errors(df):
    """Identify incorrect data formats and inconsistencies."""
    errors = {}
    for col in df.columns:
        if df[col].dtype == "object":  # Check for text-based errors
            errors[col] = df[col].str.contains(r"[^a-zA-Z0-9\s]", na=False).sum()
        elif df[col].dtype in ["int64", "float64"]:  # Numeric inconsistencies
            errors[col] = df[col].isnull().sum()
    return errors


def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()


def handle_missing_values(df, strategy="mean"):
    """Fill missing values using AI-based imputation (mean, median, mode)."""
    imputer = SimpleImputer(strategy=strategy)
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = imputer.fit_transform(df[[col]])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def standardize_data(df):
    """Standardize date formats, text capitalization, and numerical formatting."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.title()  # Capitalizing names, etc.
    for col in df.select_dtypes(include=["datetime64"]):
        df[col] = pd.to_datetime(df[col], errors="coerce")  # Standardize dates
    return df
