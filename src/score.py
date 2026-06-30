# src/score.py
# ClearTrace — Behavioral Scoring Engine
# Scores each filtered transaction 0–100 for fraud risk.
# Built from four features the PaySim data already gives us.
# Higher score = higher fraud risk = surfaces higher in the analyst queue.

import pandas as pd

WEIGHT_AMOUNT_PERCENTILE  = 30
WEIGHT_BALANCE_DRAIN      = 35
WEIGHT_DEST_RECEIVES_FULL = 25
WEIGHT_TIME_OF_DAY        = 10


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["score_amount"] = (
        df["amount"].rank(pct=True) * 100
    ).round(2)

    df["score_balance_drain"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int) * 100

    df["score_dest_receives_full"] = (
        (df["newbalanceDest"] - df["oldbalanceDest"]).round(2) == df["amount"].round(2)
    ).astype(int) * 100

    off_hours_steps = set(range(1, 7)) | set(range(20, 25))
    df["score_time_of_day"] = df["step"].isin(off_hours_steps).astype(int) * 100

    df["fraud_risk_score"] = (
        (df["score_amount"]            * WEIGHT_AMOUNT_PERCENTILE / 100) +
        (df["score_balance_drain"]     * WEIGHT_BALANCE_DRAIN / 100) +
        (df["score_dest_receives_full"] * WEIGHT_DEST_RECEIVES_FULL / 100) +
        (df["score_time_of_day"]       * WEIGHT_TIME_OF_DAY / 100)
    ).round(2)

    return df


def get_score_summary(df: pd.DataFrame) -> dict:
    scores = df["fraud_risk_score"]
    return {
        "mean_score":      round(scores.mean(), 2),
        "median_score":    round(scores.median(), 2),
        "max_score":       round(scores.max(), 2),
        "high_risk_count": int((scores >= 70).sum()),
        "high_risk_pct":   round((scores >= 70).mean() * 100, 2)
    }
