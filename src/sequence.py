# src/sequence.py
# ClearTrace -- Sequence Detector
# Uses vectorized pandas merge for performance on 500k+ rows.

import pandas as pd

DEFAULT_WINDOW_STEPS = 10


def detect_sequences(df: pd.DataFrame, window_steps: int = DEFAULT_WINDOW_STEPS) -> pd.DataFrame:
    df = df.copy().sort_values("step").reset_index(drop=True)
    df["is_sequence_flag"] = False
    df["sequence_role"] = "NONE"

    transfers = df[df["type"] == "TRANSFER"][["nameOrig", "step"]].copy()
    transfers.columns = ["nameOrig", "transfer_step"]

    cashouts = df[df["type"] == "CASH_OUT"][["nameOrig", "step"]].copy()
    cashouts.columns = ["nameOrig", "cashout_step"]

    merged = transfers.merge(cashouts, on="nameOrig", how="inner")

    sequences = merged[
        (merged["cashout_step"] > merged["transfer_step"]) &
        (merged["cashout_step"] <= merged["transfer_step"] + window_steps)
    ]

    transfer_accounts = set(zip(sequences["nameOrig"], sequences["transfer_step"]))
    cashout_accounts  = set(zip(sequences["nameOrig"], sequences["cashout_step"]))

    for idx, row in df.iterrows():
        if row["type"] == "TRANSFER" and (row["nameOrig"], row["step"]) in transfer_accounts:
            df.at[idx, "is_sequence_flag"] = True
            df.at[idx, "sequence_role"]    = "TRANSFER_ORIGIN"
        elif row["type"] == "CASH_OUT" and (row["nameOrig"], row["step"]) in cashout_accounts:
            df.at[idx, "is_sequence_flag"] = True
            df.at[idx, "sequence_role"]    = "CASHOUT_EXIT"

    return df


def get_sequence_summary(df: pd.DataFrame) -> dict:
    flagged  = df[df["is_sequence_flag"] == True]
    origins  = df[df["sequence_role"] == "TRANSFER_ORIGIN"]
    exits    = df[df["sequence_role"] == "CASHOUT_EXIT"]

    fraud_in_flagged = flagged["isFraud"].sum() if "isFraud" in flagged.columns else 0
    total_flagged    = len(flagged)

    return {
        "total_sequences_detected":   len(origins),
        "transfer_origin_count":      len(origins),
        "cashout_exit_count":         len(exits),
        "total_flagged_transactions": total_flagged,
        "fraud_in_flagged":           int(fraud_in_flagged),
        "fraud_overlap_pct":          round(
            (fraud_in_flagged / total_flagged * 100) if total_flagged > 0 else 0, 2
        )
    }
