# src/queue.py
# ClearTrace -- Ranked Alert Queue
# Combines scoring and sequence detection into a single ranked queue.
# Sequence-flagged transactions surface first regardless of score.
# Within each group, transactions rank by fraud_risk_score descending.

import pandas as pd
from src.filter import filter_transactions
from src.score import score_transactions
from src.sequence import detect_sequences


def build_queue(df: pd.DataFrame, window_steps: int = 10) -> pd.DataFrame:
    """
    Run the full ClearTrace pipeline and return a ranked alert queue.

    Pipeline order:
        1. Filter  -- drop PAYMENT, CASH_IN, DEBIT
        2. Score   -- add fraud_risk_score and feature scores
        3. Detect  -- flag TRANSFER -> CASH_OUT sequences
        4. Rank    -- sequence flags first, then by score descending

    Args:
        df: Raw transaction dataframe from ingest.
        window_steps: Step window for sequence detection.

    Returns:
        Ranked dataframe ready for analyst review.
    """
    filtered  = filter_transactions(df)
    scored    = score_transactions(filtered)
    sequenced = detect_sequences(scored, window_steps=window_steps)

    sequenced["sequence_priority"] = sequenced["is_sequence_flag"].astype(int)

    ranked = sequenced.sort_values(
        by=["sequence_priority", "fraud_risk_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    ranked["queue_rank"] = ranked.index + 1

    return ranked


def get_queue_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the current alert queue state.
    Used by the dashboard header to show queue health at a glance.

    Args:
        df: Ranked queue dataframe from build_queue.

    Returns:
        Dictionary with queue size, sequence count, and top score.
    """
    return {
        "queue_size":             len(df),
        "sequence_flagged_count": int(df["is_sequence_flag"].sum()),
        "high_risk_count":        int((df["fraud_risk_score"] >= 70).sum()),
        "top_score":              round(df["fraud_risk_score"].max(), 2),
        "transfer_count":         int((df["type"] == "TRANSFER").sum()),
        "cashout_count":          int((df["type"] == "CASH_OUT").sum())
    }
