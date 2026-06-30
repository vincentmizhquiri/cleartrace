# src/dashboard.py
# ClearTrace -- Analyst Dashboard
# Streamlit app that displays the ranked alert queue to a compliance analyst.
# Run with: streamlit run src/dashboard.py

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ingest import load_transactions, get_summary
from src.alert_queue import build_queue, get_queue_summary

# --- Page config ---
st.set_page_config(
    page_title="ClearTrace",
    page_icon="🔍",
    layout="wide"
)

# --- Header ---
st.title("🔍 ClearTrace")
st.caption("Real-time compliance prioritization dashboard — FinTech Fraud Detection")
st.divider()

# --- Load data ---
@st.cache_data
def load_data():
    df  = load_transactions()
    raw = get_summary(df)
    q   = build_queue(df, window_steps=10)
    qs  = get_queue_summary(q)
    return df, raw, q, qs

with st.spinner("Loading PaySim dataset and building alert queue..."):
    try:
        raw_df, raw_summary, queue, queue_summary = load_data()
        load_success = True
    except FileNotFoundError as e:
        st.error(str(e))
        load_success = False

if not load_success:
    st.stop()

# --- Dataset summary row ---
st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions",  f"{raw_summary['total_transactions']:,}")
c2.metric("Confirmed fraud cases", f"{raw_summary['fraud_count']:,}")
c3.metric("Overall fraud rate",  f"{raw_summary['fraud_rate_pct']}%")
c4.metric("Transaction types",   "5 types ingested")
st.divider()

# --- Queue summary row ---
st.subheader("Alert queue")
q1, q2, q3, q4, q5 = st.columns(5)
q1.metric("Queue size",            f"{queue_summary['queue_size']:,}",
          help="TRANSFER + CASH_OUT only. PAYMENT / CASH_IN / DEBIT filtered out.")
q2.metric("Sequence flagged",      f"{queue_summary['sequence_flagged_count']:,}",
          help="TRANSFER -> CASH_OUT two-step workflow detected on same account.")
q3.metric("High risk (score 70+)", f"{queue_summary['high_risk_count']:,}")
q4.metric("TRANSFER rows",         f"{queue_summary['transfer_count']:,}")
q5.metric("CASH_OUT rows",         f"{queue_summary['cashout_count']:,}")
st.divider()

# --- Filters ---
st.subheader("Filter queue")
col_a, col_b, col_c = st.columns(3)

with col_a:
    type_filter = st.selectbox(
        "Transaction type",
        options=["All", "TRANSFER", "CASH_OUT"]
    )

with col_b:
    sequence_filter = st.selectbox(
        "Sequence flag",
        options=["All", "Sequence flagged only", "Non-sequence only"]
    )

with col_c:
    min_score = st.slider(
        "Minimum fraud risk score",
        min_value=0, max_value=100, value=0, step=5
    )

# --- Apply filters ---
filtered_queue = queue.copy()

if type_filter != "All":
    filtered_queue = filtered_queue[filtered_queue["type"] == type_filter]

if sequence_filter == "Sequence flagged only":
    filtered_queue = filtered_queue[filtered_queue["is_sequence_flag"] == True]
elif sequence_filter == "Non-sequence only":
    filtered_queue = filtered_queue[filtered_queue["is_sequence_flag"] == False]

filtered_queue = filtered_queue[filtered_queue["fraud_risk_score"] >= min_score]

st.caption(f"Showing {len(filtered_queue):,} of {queue_summary['queue_size']:,} alerts")
st.divider()

# --- Queue table ---
st.subheader("Ranked alert queue")

display_cols = [
    "queue_rank", "type", "sequence_role", "fraud_risk_score",
    "amount", "nameOrig", "nameDest",
    "score_balance_drain", "score_dest_receives_full",
    "score_amount", "score_time_of_day",
    "oldbalanceOrg", "newbalanceOrig", "isFraud"
]

display_df = filtered_queue[display_cols].copy()
display_df.columns = [
    "Rank", "Type", "Sequence Role", "Risk Score",
    "Amount", "Origin Account", "Dest Account",
    "Balance Drain", "Full Transfer", "Amount Pct", "Off Hours",
    "Orig Balance Before", "Orig Balance After", "Is Fraud"
]

def highlight_row(row):
    if row["Sequence Role"] in ["TRANSFER_ORIGIN", "CASHOUT_EXIT"]:
        return ["background-color: #E1F5EE"] * len(row)
    elif row["Risk Score"] >= 70:
        return ["background-color: #FAEEDA"] * len(row)
    else:
        return [""] * len(row)

st.dataframe(
    display_df.head(1000),
    use_container_width=True,
    height=500
)

st.divider()

# --- SCR insight panel ---
st.subheader("Data insight")
st.info(
    "**Setup:** Compliance teams apply uniform surveillance across all 5 transaction types "
    "in 6.3M mobile money transactions — treating a routine PAYMENT the same as a TRANSFER "
    "with a 0.77% fraud rate.\n\n"
    "**Conflict:** 100% of fraud concentrates in TRANSFER and CASH_OUT only. "
    "PAYMENT, CASH_IN, and DEBIT produce zero fraud cases — not low fraud, zero.\n\n"
    "**Resolution:** ClearTrace filters to TRANSFER and CASH_OUT exclusively, ranks by "
    "per-transaction fraud rate, and detects the two-step TRANSFER -> CASH_OUT sequence "
    "before funds are liquidated."
)
