# src/filter.py
# ClearTrace — Transaction Type Filter
# Core feature: routes only TRANSFER and CASH_OUT to the active pipeline.
# PAYMENT, CASH_IN, and DEBIT are dropped before any scoring or review begins.
# This single filter is the architectural foundation of the entire product.

import pandas as pd

FRAUD_ACTIVE_TYPES = {"TRANSFER", "CASH_OUT"}


def filter_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a transaction dataframe to only fraud-active transaction types.

    Args:
        df: Raw transaction dataframe with a 'type' column.

    Returns:
        Filtered dataframe containing only TRANSFER and CASH_OUT rows.
        PAYMENT, CASH_IN, and DEBIT rows are dropped entirely.
    """
    if "type" not in df.columns:
        raise ValueError("Dataframe must contain a 'type' column.")

    filtered = df[df["type"].isin(FRAUD_ACTIVE_TYPES)].copy()
    filtered = filtered.reset_index(drop=True)
    return filtered


def get_dropped_types(df: pd.DataFrame) -> pd.Series:
    """
    Return a count of transaction types that were dropped by the filter.
    Useful for dashboard metrics — shows how much noise was eliminated.

    Args:
        df: Raw transaction dataframe before filtering.

    Returns:
        Series of counts for dropped transaction types only.
    """
    dropped = df[~df["type"].isin(FRAUD_ACTIVE_TYPES)]
    return dropped["type"].value_counts()