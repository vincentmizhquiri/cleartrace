# tests/test_ingest.py
# ClearTrace — Ingest Tests
# Confirms the dataset loads correctly, has the right columns,
# and produces an accurate summary before hitting the filter.

import pandas as pd
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ingest import load_transactions, get_summary, REQUIRED_COLUMNS


# --- Fixtures ---

@pytest.fixture
def mock_df():
    """Minimal valid dataframe matching PaySim schema."""
    return pd.DataFrame({
        "step":            [1, 2, 3, 4, 5],
        "type":            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
        "amount":          [1000.0, 500.0, 200.0, 800.0, 150.0],
        "nameOrig":        ["C1", "C2", "C3", "C4", "C5"],
        "oldbalanceOrg":   [1000.0, 500.0, 200.0, 800.0, 150.0],
        "newbalanceOrig":  [0.0, 0.0, 0.0, 0.0, 0.0],
        "nameDest":        ["C6", "C7", "C8", "C9", "C10"],
        "oldbalanceDest":  [0.0, 0.0, 0.0, 0.0, 0.0],
        "newbalanceDest":  [1000.0, 500.0, 200.0, 800.0, 150.0],
        "isFraud":         [1, 1, 0, 0, 0]
    })


@pytest.fixture
def mock_csv(tmp_path, mock_df):
    """Write mock dataframe to a temp CSV and return its path."""
    csv_path = tmp_path / "test_paysim.csv"
    mock_df.to_csv(csv_path, index=False)
    return csv_path


# --- Load tests ---

def test_load_returns_dataframe(mock_csv):
    """load_transactions returns a pandas DataFrame."""
    df = load_transactions(mock_csv)
    assert isinstance(df, pd.DataFrame)


def test_load_returns_correct_row_count(mock_csv):
    """load_transactions returns all rows from the CSV."""
    df = load_transactions(mock_csv)
    assert len(df) == 5


def test_load_contains_required_columns(mock_csv):
    """All required columns are present after loading."""
    df = load_transactions(mock_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_load_raises_file_not_found():
    """load_transactions raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="PaySim dataset not found"):
        load_transactions(Path("/nonexistent/path/fake.csv"))


def test_load_raises_value_error_for_missing_columns(tmp_path):
    """load_transactions raises ValueError if required columns are missing."""
    bad_df = pd.DataFrame({"type": ["TRANSFER"], "amount": [100]})
    bad_path = tmp_path / "bad.csv"
    bad_df.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_transactions(bad_path)


# --- Summary tests ---

def test_get_summary_returns_dict(mock_df):
    """get_summary returns a dictionary."""
    summary = get_summary(mock_df)
    assert isinstance(summary, dict)


def test_get_summary_total_transactions(mock_df):
    """get_summary total_transactions matches dataframe length."""
    summary = get_summary(mock_df)
    assert summary["total_transactions"] == 5


def test_get_summary_fraud_count(mock_df):
    """get_summary fraud_count matches actual fraud rows."""
    summary = get_summary(mock_df)
    assert summary["fraud_count"] == 2


def test_get_summary_fraud_rate(mock_df):
    """get_summary fraud_rate_pct is correctly calculated."""
    summary = get_summary(mock_df)
    assert summary["fraud_rate_pct"] == 40.0


def test_get_summary_type_counts(mock_df):
    """get_summary transaction_type_counts has all five types."""
    summary = get_summary(mock_df)
    counts = summary["transaction_type_counts"]
    assert set(counts.keys()) == {"TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"}