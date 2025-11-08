# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment → Stock (Demo)", layout="wide")
st.title("Sentiment Analysis for Stock Prediction — Demo")

# Sidebar: inputs
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
start_date = st.sidebar.date_input("Start date", datetime.date.today() - datetime.timedelta(days=90))
end_date = st.sidebar.date_input("End date", datetime.date.today())
use_sample = st.sidebar.checkbox("Use sample headlines (demo)", True)

analyzer = SentimentIntensityAnalyzer()

st.markdown("## Provide headlines")
st.markdown("You can paste headlines (one per line) or upload a CSV with a `date` and `headline` column.")

uploaded = st.file_uploader("Upload CSV (date,headline)", type=["csv"])
headlines_df = None

if uploaded:
    df = pd.read_csv(uploaded)
    if "headline" not in df.columns:
        st.error("CSV must have a `headline` column.")
    else:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            df["date"] = pd.NaT
        headlines_df = df[["date","headline"]]
elif use_sample:
    sample = [
        ("2025-09-10","Company reports strong quarterly revenue growth"),
        ("2025-09-12","CEO resigns unexpectedly"),
        ("2025-09-20","New product line announced, analysts optimistic"),
        ("2025-10-01","Regulatory fine imposed over minor compliance lapse"),
    ]
    headlines_df = pd.DataFrame(sample, columns=["date","headline"])
    headlines_df["date"] = pd.to_datetime(headlines_df["date"]).dt.date

# Text area fallback
st.markdown("Or paste headlines (one per line):")
pasted = st.text_area("Headlines (one per line)", height=120)
if pasted.strip():
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    df2 = pd.DataFrame({"date":[None]*len(lines),"headline":lines})
    headlines_df = df2

if headlines_df is None or headlines_df.empty:
    st.warning("No headlines provided yet. Use sample or upload/paste headlines.")
else:
    # Sentiment scoring
    def score_text(s):
        return analyzer.polarity_scores(s)["compound"]
    headlines_df["score"] = headlines_df["headline"].apply(score_text)
    st.subheader("Headlines + Sentiment")
    st.dataframe(headlines_df)

    # Aggregate daily sentiment
    agg = headlines_df.copy()
    agg["date"] = agg["date"].fillna(pd.to_datetime(start_date).date())
    agg_by_date = agg.groupby("date", as_index=False)["score"].mean()
    agg_by_date = agg_by_date.sort_values("date")

    # Fetch stock prices for ticker
    st.subheader(f"Stock prices for {ticker}")
    try:
        hist = yf.download(ticker, start=start_date - datetime.timedelta(days=7), end=end_date + datetime.timedelta(days=1), progress=False)
        if hist.empty:
            st.error("No price data found for ticker.")
        else:
            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
            prices = hist[["Date","Close"]].rename(columns={"Date":"date","Close":"close"})
            st.write(prices.tail(5))

            # Merge sentiment with prices
            merged = pd.merge(prices, agg_by_date, on="date", how="left").sort_values("date")
            merged["score"] = merged["score"].fillna(method="ffill").fillna(0)

            # Plot
            fig, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(merged["date"], merged["close"], label="Close price")
            ax1.set_ylabel("Close price")
            ax1.tick_params(axis='y')
            ax2 = ax1.twinx()
            ax2.bar(merged["date"], merged["score"], alpha=0.3, label="Avg sentiment (compound)")
            ax2.set_ylabel("Sentiment (compound)")
            fig.autofmt_xdate()
            st.pyplot(fig)

            st.markdown("### Simple correlation (sentiment vs next-day return)")
            merged["next_close"] = merged["close"].shift(-1)
            merged["next_return"] = (merged["next_close"] - merged["close"]) / merged["close"]
            corr = merged[["score","next_return"]].corr().iloc[0,1]
            st.write(f"Pearson corr (score vs next-day return): **{corr:.3f}**")
            st.dataframe(merged.head(10))
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
