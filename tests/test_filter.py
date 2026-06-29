# tests/test_filter.py
# ClearTrace — Filter Tests
# Passing these tests confirms the core hypothesis is running as code:
# TRANSFER and CASH_OUT are the only transaction types that carry fraud.
# No PAYMENT, CASH_IN, or DEBIT row should ever pass through the filter.

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.filter import filter_transactions, get_dropped_types


# --- Fixtures ---

@pytest.fixture
def sample_df():
    """Small representative sample of PaySim transaction types."""
    return pd.DataFrame({
        "type": [
            "TRANSFER", "CASH_OUT", "PAYMENT",
            "CASH_IN", "DEBIT", "TRANSFER", "CASH_OUT"
        ],
        "amount": [1000, 500, 200, 800, 150, 3000, 750],
        "isFraud": [1, 1, 0, 0, 0, 0, 1]
    })


@pytest.fixture
def zero_fraud_df():
    """Dataframe with only zero-risk transaction types — none should pass."""
    return pd.DataFrame({
        "type": ["PAYMENT", "CASH_IN", "DEBIT", "PAYMENT", "CASH_IN"],
        "amount": [100, 200, 50, 300, 400],
        "isFraud": [0, 0, 0, 0, 0]
    })


@pytest.fixture
def all_active_df():
    """Dataframe with only fraud-active types — all should pass."""
    return pd.DataFrame({
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
        "amount": [1000, 500, 2000, 750],
        "isFraud": [1, 0, 1, 1]
    })


# --- Core filter tests ---

def test_only_active_types_pass_through(sample_df):
    """TRANSFER and CASH_OUT rows pass. PAYMENT, CASH_IN, DEBIT do not."""
    result = filter_transactions(sample_df)
    assert set(result["type"].unique()) == {"TRANSFER", "CASH_OUT"}, (
        "Filter passed through a non-fraud-active transaction type."
    )


def test_payment_never_passes(sample_df):
    """PAYMENT produces zero fraud cases — must never enter the pipeline."""
    result = filter_transactions(sample_df)
    assert "PAYMENT" not in result["type"].values, (
        "PAYMENT passed through the filter. This should never happen."
    )


def test_cash_in_never_passes(sample_df):
    """CASH_IN produces zero fraud cases — must never enter the pipeline."""
    result = filter_transactions(sample_df)
    assert "CASH_IN" not in result["type"].values, (
        "CASH_IN passed through the filter. This should never happen."
    )


def test_debit_never_passes(sample_df):
    """DEBIT produces zero fraud cases — must never enter the pipeline."""
    result = filter_transactions(sample_df)
    assert "DEBIT" not in result["type"].values, (
        "DEBIT passed through the filter. This should never happen."
    )


def test_correct_row_count(sample_df):
    """Filter returns exactly the right number of rows."""
    result = filter_transactions(sample_df)
    expected_count = len(sample_df[sample_df["type"].isin({"TRANSFER", "CASH_OUT"})])
    assert len(result) == expected_count, (
        f"Expected {expected_count} rows, got {len(result)}."
    )


def test_zero_risk_types_return_empty(zero_fraud_df):
    """A dataframe of only PAYMENT/CASH_IN/DEBIT returns an empty dataframe."""
    result = filter_transactions(zero_fraud_df)
    assert len(result) == 0, (
        "Filter returned rows from a dataframe containing only zero-risk types."
    )


def test_all_active_types_pass(all_active_df):
    """A dataframe of only TRANSFER/CASH_OUT passes through completely."""
    result = filter_transactions(all_active_df)
    assert len(result) == len(all_active_df), (
        "Filter dropped rows it should have kept."
    )


def test_index_resets_after_filter(sample_df):
    """Filtered dataframe has a clean 0-based index."""
    result = filter_transactions(sample_df)
    assert list(result.index) == list(range(len(result))), (
        "Index was not reset after filtering."
    )


def test_missing_type_column_raises_error():
    """Dataframe without a 'type' column raises a clear ValueError."""
    bad_df = pd.DataFrame({"amount": [100, 200], "isFraud": [0, 1]})
    with pytest.raises(ValueError, match="'type' column"):
        filter_transactions(bad_df)


# --- Dropped types tests ---

def test_get_dropped_types_returns_correct_types(sample_df):
    """get_dropped_types returns counts only for dropped types."""
    dropped = get_dropped_types(sample_df)
    assert "PAYMENT" in dropped.index
    assert "CASH_IN" in dropped.index
    assert "DEBIT" in dropped.index
    assert "TRANSFER" not in dropped.index
    assert "CASH_OUT" not in dropped.index


def test_get_dropped_types_counts_are_correct(sample_df):
    """get_dropped_types returns accurate counts per dropped type."""
    dropped = get_dropped_types(sample_df)
    assert dropped["PAYMENT"] == 1
    assert dropped["CASH_IN"] == 1
    assert dropped["DEBIT"] == 1