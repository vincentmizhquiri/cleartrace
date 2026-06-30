# src/sequence.py
# ClearTrace -- Sequence Detector
# Watches for the two-step fraud workflow: TRANSFER followed by CASH_OUT
# on the same origin account within a configurable time window.
# This is the steroid feature -- catches the criminal workflow in motion
# before the CASH_OUT completes and funds are gone.

import pandas as pd

DEFAULT_WINDOW_STEPS = 10


def detect_sequences(df: pd.DataFrame, window_steps: int = DEFAULT_WINDOW_STEPS) -> pd.DataFrame:
    """
    Detect TRANSFER -> CASH_OUT sequences on the same origin account.

    For each TRANSFER, looks for a CASH_OUT from the same nameOrig
    within window_steps steps. Flags both transactions as part of a sequence.

    Args:
        df: Filtered and scored dataframe (TRANSFER and CASH_OUT only).
        window_steps: Maximum step gap between TRANSFER and CASH_OUT to flag.

    Returns:
        Dataframe with two new columns:
            is_sequence_flag (bool): True if part of a detected sequence.
            sequence_role (str): 'TRANSFER_ORIGIN', 'CASHOUT_EXIT', or 'NONE'.
    """
    df = df.copy().sort_values("step").reset_index(drop=True)
    df["is_sequence_flag"] = False
    df["sequence_role"] = "NONE"

    transfers = df[df["type"] == "TRANSFER"][["nameOrig", "step"]].copy()
    cashouts  = df[df["type"] == "CASH_OUT"][["nameOrig", "step"]].copy()

    for idx, transfer in transfers.iterrows():
        account = transfer["nameOrig"]
        t_step  = transfer["step"]

        matching_cashouts = cashouts[
            (cashouts["nameOrig"] == account) &
            (cashouts["step"] > t_step) &
            (cashouts["step"] <= t_step + window_steps)
        ]

        if not matching_cashouts.empty:
            df.at[idx, "is_sequence_flag"] = True
            df.at[idx, "sequence_role"]    = "TRANSFER_ORIGIN"

            for co_idx in matching_cashouts.index:
                df.at[co_idx, "is_sequence_flag"] = True
                df.at[co_idx, "sequence_role"]    = "CASHOUT_EXIT"

    return df


def get_sequence_summary(df: pd.DataFrame) -> dict:
    """
    Return summary statistics for detected sequences.
    Used by the dashboard to show sequence detection activity.

    Args:
        df: Dataframe after detect_sequences has been applied.

    Returns:
        Dictionary with sequence counts and fraud overlap rate.
    """
    flagged = df[df["is_sequence_flag"] == True]
    origins = df[df["sequence_role"] == "TRANSFER_ORIGIN"]
    exits   = df[df["sequence_role"] == "CASHOUT_EXIT"]

    fraud_in_flagged = flagged["isFraud"].sum() if "isFraud" in flagged.columns else 0
    total_flagged    = len(flagged)

    return {
        "total_sequences_detected": len(origins),
        "transfer_origin_count":    len(origins),
        "cashout_exit_count":       len(exits),
        "total_flagged_transactions": total_flagged,
        "fraud_in_flagged":         int(fraud_in_flagged),
        "fraud_overlap_pct":        round(
            (fraud_in_flagged / total_flagged * 100) if total_flagged > 0 else 0, 2
        )
    }
