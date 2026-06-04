import pandas as pd

CATEGORICAL_COLUMNS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

TARGET_COLUMN = "Default"


def _present_categoricals(df: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_COLUMNS if c in df.columns]


def encode_binary_columns(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Map two-level categoricals to 0/1 (lexicographically larger label -> 1).

    Mutates ``df`` in place and returns it.
    """
    if columns is None:
        columns = [
            c
            for c in _present_categoricals(df)
            if df[c].dropna().nunique() == 2
        ]

    for col in columns:
        levels = sorted(df[col].dropna().unique(), key=str)
        mapping = {levels[0]: 0, levels[1]: 1}
        df[col] = df[col].map(mapping).astype("int8")

    return df


def encode_multi_category_columns(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    One-hot encode categoricals with more than two levels (drop_first=True).

    Mutates ``df`` in place and returns it.
    """
    if columns is None:
        columns = [
            c
            for c in _present_categoricals(df)
            if df[c].dropna().nunique() > 2
        ]

    if not columns:
        return df

    dummies = pd.get_dummies(df[columns], drop_first=True, dtype=int)
    df.drop(columns=columns, inplace=True)
    df[dummies.columns] = dummies

    return df


def normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces and slashes in column names (e.g. for tree models)."""
    df.columns = df.columns.str.replace(r"[\s/]+", "_", regex=True)
    return df


def ensure_ml_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to dtypes suitable for sklearn / LightGBM / XGBoost.

    Mutates ``df`` in place and returns it.
    """
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            df[col] = series.astype(int)
        elif pd.api.types.is_integer_dtype(series):
            continue
        elif pd.api.types.is_numeric_dtype(series):
            df[col] = series.astype(float)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals, normalize names, and set ML-ready dtypes.

    Applies all feature steps in place on ``df`` and returns the same object.
    """
    encode_binary_columns(df)
    encode_multi_category_columns(df)
    normalize_feature_names(df)
    ensure_ml_dtypes(df)
    return df
