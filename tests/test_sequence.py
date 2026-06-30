# tests/test_sequence.py
# ClearTrace -- Sequence Detector Tests
# Confirms the two-step TRANSFER -> CASH_OUT detection works correctly
# across window boundaries, account matching, and summary statistics.

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sequence import detect_sequences, get_sequence_summary, DEFAULT_WINDOW_STEPS


# --- Fixtures ---

@pytest.fixture
def clear_sequence_df():
    """One TRANSFER followed by one CASH_OUT on the same account within window."""
    return pd.DataFrame({
        "step":           [1, 5],
        "type":           ["TRANSFER", "CASH_OUT"],
        "amount":         [1000.0, 1000.0],
        "nameOrig":       ["C1", "C1"],
        "oldbalanceOrg":  [1000.0, 0.0],
        "newbalanceOrig": [0.0, 0.0],
        "nameDest":       ["C2", "C3"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [1000.0, 1000.0],
        "isFraud":        [1, 1],
        "fraud_risk_score": [85.0, 72.0]
    })


@pytest.fixture
def no_sequence_df():
    """TRANSFER and CASH_OUT on different accounts -- no sequence."""
    return pd.DataFrame({
        "step":           [1, 5],
        "type":           ["TRANSFER", "CASH_OUT"],
        "amount":         [1000.0, 500.0],
        "nameOrig":       ["C1", "C2"],
        "oldbalanceOrg":  [1000.0, 500.0],
        "newbalanceOrig": [0.0, 0.0],
        "nameDest":       ["C3", "C4"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [1000.0, 500.0],
        "isFraud":        [0, 0],
        "fraud_risk_score": [45.0, 30.0]
    })


@pytest.fixture
def outside_window_df():
    """TRANSFER and CASH_OUT on same account but outside the window."""
    return pd.DataFrame({
        "step":           [1, 50],
        "type":           ["TRANSFER", "CASH_OUT"],
        "amount":         [1000.0, 1000.0],
        "nameOrig":       ["C1", "C1"],
        "oldbalanceOrg":  [1000.0, 0.0],
        "newbalanceOrig": [0.0, 0.0],
        "nameDest":       ["C2", "C3"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [1000.0, 1000.0],
        "isFraud":        [1, 1],
        "fraud_risk_score": [85.0, 72.0]
    })


@pytest.fixture
def multi_sequence_df():
    """Two separate sequences on two different accounts."""
    return pd.DataFrame({
        "step":           [1,  3,  10, 12],
        "type":           ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
        "amount":         [1000.0, 1000.0, 500.0, 500.0],
        "nameOrig":       ["C1", "C1", "C2", "C2"],
        "oldbalanceOrg":  [1000.0, 0.0, 500.0, 0.0],
        "newbalanceOrig": [0.0, 0.0, 0.0, 0.0],
        "nameDest":       ["C3", "C4", "C5", "C6"],
        "oldbalanceDest": [0.0, 0.0, 0.0, 0.0],
        "newbalanceDest": [1000.0, 1000.0, 500.0, 500.0],
        "isFraud":        [1, 1, 1, 1],
        "fraud_risk_score": [85.0, 72.0, 78.0, 65.0]
    })


# --- Output structure tests ---

def test_detect_adds_sequence_flag_column(clear_sequence_df):
    """detect_sequences adds is_sequence_flag column."""
    result = detect_sequences(clear_sequence_df)
    assert "is_sequence_flag" in result.columns


def test_detect_adds_sequence_role_column(clear_sequence_df):
    """detect_sequences adds sequence_role column."""
    result = detect_sequences(clear_sequence_df)
    assert "sequence_role" in result.columns


def test_detect_preserves_row_count(clear_sequence_df):
    """detect_sequences does not add or drop rows."""
    result = detect_sequences(clear_sequence_df)
    assert len(result) == len(clear_sequence_df)


# --- Sequence detection tests ---

def test_same_account_within_window_flagged(clear_sequence_df):
    """TRANSFER and CASH_OUT on same account within window are both flagged."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    assert result["is_sequence_flag"].all()


def test_transfer_role_is_transfer_origin(clear_sequence_df):
    """Flagged TRANSFER gets sequence_role of TRANSFER_ORIGIN."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    transfer_row = result[result["type"] == "TRANSFER"]
    assert (transfer_row["sequence_role"] == "TRANSFER_ORIGIN").all()


def test_cashout_role_is_cashout_exit(clear_sequence_df):
    """Flagged CASH_OUT gets sequence_role of CASHOUT_EXIT."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    cashout_row = result[result["type"] == "CASH_OUT"]
    assert (cashout_row["sequence_role"] == "CASHOUT_EXIT").all()


def test_different_accounts_not_flagged(no_sequence_df):
    """TRANSFER and CASH_OUT on different accounts are not flagged."""
    result = detect_sequences(no_sequence_df, window_steps=10)
    assert not result["is_sequence_flag"].any()


def test_outside_window_not_flagged(outside_window_df):
    """CASH_OUT outside the step window is not flagged."""
    result = detect_sequences(outside_window_df, window_steps=10)
    assert not result["is_sequence_flag"].any()


def test_multiple_sequences_detected(multi_sequence_df):
    """Two sequences on two accounts are both detected."""
    result = detect_sequences(multi_sequence_df, window_steps=10)
    assert result["is_sequence_flag"].all()
    assert len(result[result["sequence_role"] == "TRANSFER_ORIGIN"]) == 2
    assert len(result[result["sequence_role"] == "CASHOUT_EXIT"]) == 2


def test_no_sequence_roles_are_none_when_unflagged(no_sequence_df):
    """All sequence_role values are NONE when no sequence is detected."""
    result = detect_sequences(no_sequence_df, window_steps=10)
    assert (result["sequence_role"] == "NONE").all()


def test_default_window_is_set():
    """DEFAULT_WINDOW_STEPS is defined and greater than zero."""
    assert DEFAULT_WINDOW_STEPS > 0


# --- Summary tests ---

def test_get_sequence_summary_returns_dict(clear_sequence_df):
    """get_sequence_summary returns a dictionary."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    summary = get_sequence_summary(result)
    assert isinstance(summary, dict)


def test_get_sequence_summary_has_required_keys(clear_sequence_df):
    """get_sequence_summary contains all required keys."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    summary = get_sequence_summary(result)
    assert "total_sequences_detected" in summary
    assert "transfer_origin_count" in summary
    assert "cashout_exit_count" in summary
    assert "total_flagged_transactions" in summary
    assert "fraud_in_flagged" in summary
    assert "fraud_overlap_pct" in summary


def test_sequence_summary_counts_correct(clear_sequence_df):
    """Summary counts match the detected sequence."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    summary = get_sequence_summary(result)
    assert summary["total_sequences_detected"] == 1
    assert summary["transfer_origin_count"] == 1
    assert summary["cashout_exit_count"] == 1
    assert summary["total_flagged_transactions"] == 2


def test_sequence_summary_fraud_overlap(clear_sequence_df):
    """Fraud overlap pct is 100 when all flagged transactions are fraud."""
    result = detect_sequences(clear_sequence_df, window_steps=10)
    summary = get_sequence_summary(result)
    assert summary["fraud_overlap_pct"] == 100.0


def test_sequence_summary_no_sequences(no_sequence_df):
    """Summary shows zero sequences when none detected."""
    result = detect_sequences(no_sequence_df, window_steps=10)
    summary = get_sequence_summary(result)
    assert summary["total_sequences_detected"] == 0
    assert summary["total_flagged_transactions"] == 0
