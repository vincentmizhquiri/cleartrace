# tests/test_queue.py
# ClearTrace -- Queue Tests
# Confirms the full pipeline runs correctly end to end:
# filter -> score -> sequence detect -> rank.

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.alert_queue import build_queue, get_queue_summary


# --- Fixtures ---

@pytest.fixture
def raw_df():
    """Raw transaction dataframe with all five types -- mimics PaySim input."""
    return pd.DataFrame({
        "step":           [1,  2,  3,  4,  5,  6,  7,  8],
        "type":           ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT",
                           "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount":         [1000.0, 1000.0, 200.0, 500.0, 100.0, 800.0, 800.0, 300.0],
        "nameOrig":       ["C1", "C1", "C2", "C3", "C4", "C5", "C5", "C6"],
        "oldbalanceOrg":  [1000.0, 0.0, 200.0, 500.0, 100.0, 800.0, 0.0, 300.0],
        "newbalanceOrig": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "nameDest":       ["C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"],
        "oldbalanceDest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "newbalanceDest": [1000.0, 1000.0, 200.0, 500.0, 100.0, 800.0, 800.0, 300.0],
        "isFraud":        [1, 1, 0, 0, 0, 1, 1, 0]
    })


@pytest.fixture
def queue_df(raw_df):
    """Pre-built queue for use in multiple tests."""
    return build_queue(raw_df, window_steps=10)


# --- Pipeline integration tests ---

def test_queue_returns_dataframe(queue_df):
    """build_queue returns a pandas DataFrame."""
    assert isinstance(queue_df, pd.DataFrame)


def test_queue_contains_only_active_types(queue_df):
    """Queue contains only TRANSFER and CASH_OUT rows."""
    assert set(queue_df["type"].unique()) == {"TRANSFER", "CASH_OUT"}


def test_queue_drops_zero_risk_types(raw_df, queue_df):
    """PAYMENT, CASH_IN, and DEBIT are not present in the queue."""
    assert "PAYMENT" not in queue_df["type"].values
    assert "CASH_IN" not in queue_df["type"].values
    assert "DEBIT" not in queue_df["type"].values


def test_queue_has_fraud_risk_score(queue_df):
    """Queue contains fraud_risk_score column."""
    assert "fraud_risk_score" in queue_df.columns


def test_queue_has_sequence_flag(queue_df):
    """Queue contains is_sequence_flag column."""
    assert "is_sequence_flag" in queue_df.columns


def test_queue_has_queue_rank(queue_df):
    """Queue contains queue_rank column."""
    assert "queue_rank" in queue_df.columns


# --- Ranking tests ---

def test_queue_rank_starts_at_one(queue_df):
    """queue_rank starts at 1."""
    assert queue_df["queue_rank"].min() == 1


def test_queue_rank_is_sequential(queue_df):
    """queue_rank is sequential with no gaps."""
    assert list(queue_df["queue_rank"]) == list(range(1, len(queue_df) + 1))


def test_sequence_flagged_transactions_rank_first(queue_df):
    """All sequence-flagged transactions appear before non-flagged ones."""
    flagged_ranks   = queue_df[queue_df["is_sequence_flag"] == True]["queue_rank"]
    unflagged_ranks = queue_df[queue_df["is_sequence_flag"] == False]["queue_rank"]
    if len(flagged_ranks) > 0 and len(unflagged_ranks) > 0:
        assert flagged_ranks.max() < unflagged_ranks.min()


def test_scores_descending_within_unflagged(queue_df):
    """Within non-sequence rows, fraud_risk_score is descending."""
    unflagged = queue_df[queue_df["is_sequence_flag"] == False]["fraud_risk_score"]
    assert list(unflagged) == sorted(unflagged, reverse=True)


# --- Summary tests ---

def test_get_queue_summary_returns_dict(queue_df):
    """get_queue_summary returns a dictionary."""
    summary = get_queue_summary(queue_df)
    assert isinstance(summary, dict)


def test_get_queue_summary_has_required_keys(queue_df):
    """get_queue_summary contains all required keys."""
    summary = get_queue_summary(queue_df)
    assert "queue_size" in summary
    assert "sequence_flagged_count" in summary
    assert "high_risk_count" in summary
    assert "top_score" in summary
    assert "transfer_count" in summary
    assert "cashout_count" in summary


def test_queue_summary_size_matches_queue(raw_df, queue_df):
    """queue_size equals the number of TRANSFER and CASH_OUT rows in raw data."""
    expected = len(raw_df[raw_df["type"].isin(["TRANSFER", "CASH_OUT"])])
    summary  = get_queue_summary(queue_df)
    assert summary["queue_size"] == expected


def test_queue_summary_transfer_count(queue_df):
    """transfer_count matches actual TRANSFER rows in queue."""
    summary = get_queue_summary(queue_df)
    assert summary["transfer_count"] == len(queue_df[queue_df["type"] == "TRANSFER"])


def test_queue_summary_cashout_count(queue_df):
    """cashout_count matches actual CASH_OUT rows in queue."""
    summary = get_queue_summary(queue_df)
    assert summary["cashout_count"] == len(queue_df[queue_df["type"] == "CASH_OUT"])


def test_queue_summary_top_score_is_max(queue_df):
    """top_score matches the maximum fraud_risk_score in the queue."""
    summary = get_queue_summary(queue_df)
    assert summary["top_score"] == round(queue_df["fraud_risk_score"].max(), 2)
