from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Brickwise", page_icon="🏠", layout="wide")

st.title("Brickwise Property Decision Support")
st.caption("Initial interactive prototype")

asking_price = st.number_input("Asking price", min_value=1_000, value=500_000, step=5_000)
estimated_value = st.number_input(
    "Initial estimated value", min_value=1_000, value=525_000, step=5_000
)
monthly_cash_flow = st.number_input("Estimated monthly cash flow", value=250, step=50)

valuation_margin_pct = (estimated_value - asking_price) / asking_price * 100
if valuation_margin_pct >= 3 and monthly_cash_flow >= 0:
    recommendation = "BUY"
elif valuation_margin_pct < -10 or monthly_cash_flow < -250:
    recommendation = "AVOID"
else:
    recommendation = "WATCH"

left, middle, right = st.columns(3)
left.metric("Recommendation", recommendation)
middle.metric("Valuation margin", f"{valuation_margin_pct:+.1f}%")
right.metric("Monthly cash flow", f"${monthly_cash_flow:,.0f}")

st.info("The valuation model and scenario engine will replace these preliminary rules.")
