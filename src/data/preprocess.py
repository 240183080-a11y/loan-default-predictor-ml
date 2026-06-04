import pandas as pd

ID_COLUMNS = ["LoanID"]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw loan data before feature engineering.

    Drops non-predictive ID columns. Categorical encoding lives in
    ``src.features.build_features``.
    """
    return df.drop(columns=ID_COLUMNS, errors="ignore")
