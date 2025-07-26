import pandas as pd


def load_ai4i_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the AI4I 2020 predictive maintenance dataset."""
    df = pd.read_csv(csv_path)

    # Drop identifier columns if present
    for col in ["UDI", "ProductID"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Encode categorical failure type
    if "FailureType" in df.columns:
        df["FailureTypeLabel"], _ = pd.factorize(df["FailureType"])

    # Simple RUL approximation if not provided
    if "TWF" in df.columns and "RUL" not in df.columns:
        df["RUL"] = df.index.to_series().iloc[::-1]

    return df
