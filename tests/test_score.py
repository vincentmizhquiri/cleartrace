# tests/test_score.py
# ClearTrace — Scoring Engine Tests
# Confirms each feature scores correctly and the composite score
# stays within 0–100 bounds across all transaction types.

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.score import score_transactions, get_score_summary


# --- Fixtures ---

@pytest.fixture
def base_df():
    """Representative filtered dataframe — TRANSFER and CASH_OUT only."""
    return pd.DataFrame({
        "step":             [1,      15,     3,      22,     10],
        "type":             ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount":           [1000.0, 200.0,  5000.0, 800.0,  100.0],
        "nameOrig":         ["C1",   "C2",   "C3",   "C4",   "C5"],
        "oldbalanceOrg":    [1000.0, 200.0,  5000.0, 800.0,  500.0],
        "newbalanceOrig":   [0.0,    0.0,    0.0,    0.0,    400.0],
        "nameDest":         ["C6",   "C7",   "C8",   "C9",   "C10"],
        "oldbalanceDest":   [0.0,    0.0,    0.0,    0.0,    0.0],
        "newbalanceDest":   [1000.0, 200.0,  5000.0, 800.0,  50.0],
        "isFraud":          [1,      1,      1,      1,      0]
    })


@pytest.fixture
def scored_df(base_df):
    """Pre-scored dataframe for use in multiple tests."""
    return score_transactions(base_df)


# --- Output structure tests ---

def test_score_adds_fraud_risk_score_column(scored_df):
    """score_transactions adds a fraud_risk_score column."""
    assert "fraud_risk_score" in scored_df.columns


def test_score_adds_all_feature_columns(scored_df):
    """score_transactions adds all four feature score columns."""
    assert "score_amount" in scored_df.columns
    assert "score_balance_drain" in scored_df.columns
    assert "score_dest_receives_full" in scored_df.columns
    assert "score_time_of_day" in scored_df.columns


def test_score_preserves_row_count(base_df, scored_df):
    """score_transactions does not add or drop rows."""
    assert len(scored_df) == len(base_df)


def test_score_does_not_modify_original(base_df):
    """score_transactions does not mutate the input dataframe."""
    original_columns = set(base_df.columns)
    score_transactions(base_df)
    assert set(base_df.columns) == original_columns


# --- Score bounds tests ---

def test_fraud_risk_score_min_is_zero_or_above(scored_df):
    """No fraud_risk_score is below 0."""
    assert scored_df["fraud_risk_score"].min() >= 0


def test_fraud_risk_score_max_is_100_or_below(scored_df):
    """No fraud_risk_score exceeds 100."""
    assert scored_df["fraud_risk_score"].max() <= 100


def test_feature_scores_are_bounded(scored_df):
    """All individual feature scores are between 0 and 100."""
    for col in ["score_balance_drain", "score_dest_receives_full", "score_time_of_day"]:
        assert scored_df[col].min() >= 0
        assert scored_df[col].max() <= 100


# --- Feature logic tests ---

def test_balance_drain_flags_zero_balance(scored_df):
    """Rows where newbalanceOrig == 0 and oldbalanceOrg > 0 score 100 on balance drain."""
    drained = scored_df[
        (scored_df["newbalanceOrig"] == 0) & (scored_df["oldbalanceOrg"] > 0)
    ]
    assert (drained["score_balance_drain"] == 100).all()


def test_balance_drain_ignores_partial_drain(base_df):
    """Rows where origin account is not fully drained score 0 on balance drain."""
    result = score_transactions(base_df)
    partial = result[result["newbalanceOrig"] > 0]
    assert (partial["score_balance_drain"] == 0).all()


def test_off_hours_steps_score_100(scored_df):
    """Transactions in steps 1–6 or 20–24 score 100 on time_of_day."""
    off_hours = scored_df[scored_df["step"].isin(list(range(1, 7)) + list(range(20, 25)))]
    assert (off_hours["score_time_of_day"] == 100).all()


def test_business_hours_steps_score_0(scored_df):
    """Transactions in steps 7–19 score 0 on time_of_day."""
    business = scored_df[scored_df["step"].isin(range(7, 20))]
    assert (business["score_time_of_day"] == 0).all()


def test_dest_receives_full_amount(scored_df):
    """Rows where dest balance increases by exactly the amount score 100."""
    full_receive = scored_df[
        (scored_df["newbalanceDest"] - scored_df["oldbalanceDest"]).round(2) ==
        scored_df["amount"].round(2)
    ]
    assert (full_receive["score_dest_receives_full"] == 100).all()


# --- Summary tests ---

def test_get_score_summary_returns_dict(scored_df):
    """get_score_summary returns a dictionary."""
    summary = get_score_summary(scored_df)
    assert isinstance(summary, dict)


def test_get_score_summary_has_required_keys(scored_df):
    """get_score_summary contains all required keys."""
    summary = get_score_summary(scored_df)
    assert "mean_score" in summary
    assert "median_score" in summary
    assert "max_score" in summary
    assert "high_risk_count" in summary
    assert "high_risk_pct" in summary


def test_get_score_summary_high_risk_count_is_integer(scored_df):
    """high_risk_count is an integer."""
    summary = get_score_summary(scored_df)
    assert isinstance(summary["high_risk_count"], int)


def test_get_score_summary_high_risk_pct_bounded(scored_df):
    """high_risk_pct is between 0 and 100."""
    summary = get_score_summary(scored_df)
    assert 0 <= summary["high_risk_pct"] <= 100