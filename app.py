# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sentiment → Stock (Graphs)", layout="wide")
st.title("📈 Sentiment vs Stock — Interactive Visuals (Render-ready)")

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
start_date = st.sidebar.date_input("Start date", datetime.date.today() - datetime.timedelta(days=180))
end_date = st.sidebar.date_input("End date", datetime.date.today())
use_sample = st.sidebar.checkbox("Use sample headlines (demo)", True)

# -------------------------
# Helper functions for robust merge
# -------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has MultiIndex columns, flatten them into single-level strings joined by '_'.
    Otherwise return df unchanged.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        new_cols = []
        for col in df.columns:
            # col is a tuple
            parts = [str(c) for c in col if (c is not None and str(c) != "")]
            new_cols.append("_".join(parts) if parts else "")
        df.columns = new_cols
    return df

def ensure_date_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Make sure 'col' exists as a column, not just an index level.
    Normalize to Python date objects where possible.
    """
    df = df.copy()
    # If date is index and named, reset it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        if col not in df.columns and df.index.name is not None:
            # reset index to bring it into columns
            df = df.reset_index()
    # Normalize date column to datetime.date if present
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        except Exception:
            df[col] = df[col].astype(str)
    return df

# -------------------------
# Upload / parse headlines
# -------------------------
st.markdown("## Provide headlines")
st.markdown("Upload CSV with columns `date,headline` OR paste headlines one per line in `YYYY-MM-DD|headline` format.")

uploaded = st.file_uploader("Upload CSV (date,headline)", type=["csv"])
headlines_df = None

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        # try common alternate column names and standardize to 'headline'
        if "headline" not in df.columns:
            for alt in ["Headline", "title", "Title", "text", "Text"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "headline"})
                    break
        if "headline" not in df.columns:
            st.error("CSV must have a `headline` column (or rename a column to 'headline').")
        else:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            else:
                df["date"] = pd.NaT
            headlines_df = df[["date", "headline"]].copy()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# sample fallback
if headlines_df is None and use_sample:
    sample = [
        ("2025-09-10", "Apple reports strong quarterly revenue growth"),
        ("2025-09-12", "CEO resigns unexpectedly"),
        ("2025-09-20", "New product line announced, analysts optimistic"),
        ("2025-10-01", "Regulatory fine imposed over minor compliance lapse"),
        ("2025-10-15", "Rumors of strategic acquisition boost investor sentiment"),
    ]
    headlines_df = pd.DataFrame(sample, columns=["date", "headline"])
    headlines_df["date"] = pd.to_datetime(headlines_df["date"]).dt.date

# paste area (overrides sample if provided)
pasted = st.text_area("Or paste headlines (YYYY-MM-DD|headline per line)", height=140)
if pasted.strip():
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    parsed = []
    for line in lines:
        if "|" in line:
            d, text = line.split("|", 1)
            try:
                d_parsed = pd.to_datetime(d, errors="coerce").date()
                d_val = d_parsed if pd.notna(d_parsed) else None
            except Exception:
                d_val = None
            parsed.append({"date": d_val, "headline": text.strip()})
        else:
            parsed.append({"date": None, "headline": line})
    headlines_df = pd.DataFrame(parsed)
    if "date" in headlines_df.columns:
        headlines_df["date"] = headlines_df["date"].apply(lambda x: pd.to_datetime(x, errors="coerce").date() if pd.notna(x) else None)

if headlines_df is None or headlines_df.empty:
    st.warning("No headlines provided yet. Use sample, upload a CSV, or paste headlines.")
    st.stop()

# -------------------------
# Score headlines (VADER)
# -------------------------
analyzer = SentimentIntensityAnalyzer()

def score_text(s: str) -> float:
    try:
        return analyzer.polarity_scores(str(s))["compound"]
    except Exception:
        return 0.0

headlines_df = headlines_df.copy()
headlines_df["score"] = headlines_df["headline"].apply(score_text)
# Fill missing dates with start_date so headlines without dates map somewhere
headlines_df["date"] = headlines_df["date"].apply(lambda x: x if (x is not None and not pd.isna(x)) else start_date)

# Aggregate daily sentiment
daily_sentiment = headlines_df.groupby("date", as_index=False).agg(
    avg_score=("score", "mean"),
    count=("score", "size")
).sort_values("date")

# -------------------------
# Fetch price data
# -------------------------
try:
    hist = yf.download(ticker, start=start_date - datetime.timedelta(days=7), end=end_date + datetime.timedelta(days=1), progress=False)
    if hist.empty:
        st.error("No price data found for ticker. Check ticker symbol or date range.")
        st.stop()
    hist = hist.reset_index()
    # ensure plain column name 'date'
    hist["date"] = pd.to_datetime(hist["Date"]).dt.date
    prices = hist[["date", "Close", "Volume"]].rename(columns={"Close": "close"})
except Exception as e:
    st.error(f"Error fetching prices: {e}")
    st.stop()

# -------------------------
# Flatten and normalize before merge (fixes MultiIndex MergeError)
# -------------------------
prices = flatten_columns(prices)
daily_sentiment = flatten_columns(daily_sentiment)

prices = ensure_date_column(prices, col="date")
daily_sentiment = ensure_date_column(daily_sentiment, col="date")

# double-check both have 'date' column and are same dtype
if "date" not in prices.columns or "date" not in daily_sentiment.columns:
    st.error("Required 'date' column missing after normalization. Inspect input CSV and price data.")
    st.write("prices.columns:", list(prices.columns))
    st.write("daily_sentiment.columns:", list(daily_sentiment.columns))
    st.stop()

# Merge safely now
try:
    merged = pd.merge(prices, daily_sentiment, on="date", how="left").sort_values("date")
except Exception as e:
    st.error("Error merging price and sentiment data:")
    st.write(str(e))
    st.write("prices.columns:", list(prices.columns))
    st.write("prices.index.names:", prices.index.names)
    st.write("daily_sentiment.columns:", list(daily_sentiment.columns))
    st.write("daily_sentiment.index.names:", daily_sentiment.index.names)
    st.stop()

# Fill sentiment gaps, compute returns
merged["avg_score"] = merged.get("avg_score").fillna(method="ffill").fillna(0.0)
if "count" in merged.columns:
    merged["count"] = merged["count"].fillna(0).astype(int)
else:
    merged["count"] = 0

merged["return"] = merged["close"].pct_change()
merged["next_close"] = merged["close"].shift(-1)
merged["next_return"] = (merged["next_close"] - merged["close"]) / merged["close"]

# -------------------------
# Display headlines and charts
# -------------------------
st.markdown("### Headlines (sample)")
st.dataframe(headlines_df.head(10))

# Price vs Sentiment - Plotly with two y-axes
st.markdown("## Price vs Sentiment")
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged["date"], y=merged["close"], name="Close Price", mode="lines", yaxis="y"))
fig.add_trace(go.Bar(x=merged["date"], y=merged["avg_score"], name="Daily Avg Sentiment (compound)", yaxis="y2", opacity=0.5))

# add rolling sentiment
rolling_window = 7
merged["sent_roll"] = merged["avg_score"].rolling(window=rolling_window, min_periods=1).mean()
fig.add_trace(go.Scatter(x=merged["date"], y=merged["sent_roll"], name=f"{rolling_window}-day rolling sentiment", yaxis="y2", mode="lines", line=dict(dash="dash")))

fig.update_layout(
    xaxis=dict(title="Date"),
    yaxis=dict(title="Close Price", side="left"),
    yaxis2=dict(title="Sentiment (compound)", overlaying="y", side="right", range=[-1,1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Scatter: sentiment vs next-day return with regression
st.markdown("## Sentiment vs Next-day Return (scatter + regression)")
scatter_df = merged.dropna(subset=["avg_score", "next_return"]).copy()
if scatter_df.empty:
    st.write("Not enough data to compute next-day returns (need at least 2 price days).")
else:
    X = scatter_df["avg_score"].values.reshape(-1, 1)
    y = scatter_df["next_return"].values.reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    x_line = np.linspace(scatter_df["avg_score"].min(), scatter_df["avg_score"].max(), 50)
    y_line = intercept + slope * x_line

    sc_fig = px.scatter(scatter_df, x="avg_score", y="next_return", hover_data=["date", "close"], labels={"avg_score": "Avg daily sentiment", "next_return": "Next-day return"})
    sc_fig.add_traces(go.Line(x=x_line, y=y_line, name=f"LR fit (slope {slope:.4f})"))
    sc_fig.update_layout(legend=dict(orientation="h"))
    st.plotly_chart(sc_fig, use_container_width=True)

    scatter_df["sent_bucket"] = pd.qcut(scatter_df["avg_score"].rank(method="first"), q=3, labels=["low", "mid", "high"])
    stats = scatter_df.groupby("sent_bucket").agg(mean_next_return=("next_return", "mean"), median_next_return=("next_return", "median"), count=("next_return", "size")).reset_index()
    st.markdown("### Next-day return by sentiment bucket")
    st.dataframe(stats)

# Correlation heatmap
st.markdown("## Correlation heatmap")
corr_cols = ["avg_score", "return", "next_return", "close", "Volume"]
corr_df = merged[[c for c in corr_cols if c in merged.columns]].copy().dropna()
if not corr_df.empty:
    corr = corr_df.corr()
    hm = px.imshow(corr, text_auto=True, aspect="auto", labels=dict(x="Metric", y="Metric", color="Correlation"))
    st.plotly_chart(hm, use_container_width=True)
else:
    st.write("Not enough data to compute correlations.")

# Merged data table and quick metrics
st.markdown("## Merged data sample (prices + sentiment)")
st.dataframe(merged.tail(60).reset_index(drop=True))

st.markdown("## Quick metrics")
corr_val = merged[["avg_score", "next_return"]].corr().iloc[0, 1]
st.write(f"Pearson correlation (avg_score vs next-day return): **{corr_val:.4f}**")
st.write(f"Average daily sentiment (mean): **{merged['avg_score'].mean():.4f}**, sample days used: **{len(merged)}**")

st.markdown("---")
st.markdown("**Notes:** VADER is rule-based and gives a compound score between -1 and 1. This demo is for exploration — not investment advice.")
