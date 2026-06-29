# src/ingest.py
# ClearTrace — Data Ingestion Layer
# Loads the PaySim CSV and passes the full dataframe to the filter.
# In MVP this is a batch load. The structure supports streaming in v2.

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"

REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig",
    "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud"
}


def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the PaySim transaction CSV into a dataframe.

    Args:
        path: Path to the PaySim CSV file. Defaults to data/ folder.

    Returns:
        Raw transaction dataframe with all original columns intact.

    Raises:
        FileNotFoundError: If the CSV is not found at the specified path.
        ValueError: If required columns are missing from the dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found at {path}. "
            "Download it from kaggle.com/datasets/ealaxi/paysim1 "
            "and place it in the data/ folder."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Confirm you are using the correct PaySim CSV file."
        )

    return df


def get_summary(df: pd.DataFrame) -> dict:
    """
    Return a quick summary of the loaded dataset.
    Used by the dashboard to confirm data loaded correctly.

    Args:
        df: Raw transaction dataframe.

    Returns:
        Dictionary with total rows, fraud count, fraud rate, and type counts.
    """
    return {
        "total_transactions": len(df),
        "fraud_count": int(df["isFraud"].sum()),
        "fraud_rate_pct": round(df["isFraud"].mean() * 100, 4),
        "transaction_type_counts": df["type"].value_counts().to_dict()
    }