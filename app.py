import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for server-side rendering
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io
import base64

from flask import Flask, request, render_template_string

# --- NLTK Setup ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- Initialize Flask App ---
app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# --- Helper Functions ---

def fetch_stock_data(ticker):
    """Fetches 1-year historical price data."""
    stock = yf.Ticker(ticker)
    df_price = stock.history(period="1y", interval="1d")
    if df_price.empty:
        raise ValueError(f"Could not fetch stock data for {ticker}. Is the symbol correct?")
    df_price.reset_index(inplace=True)
    df_price = df_price[['Date', 'Close']]
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    return df_price

def fetch_headlines(ticker):
    """Scrapes news headlines from Finviz with robust error handling."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    
    # User-Agent to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error connecting to Finviz: {e}")

    soup = BeautifulSoup(response.content, "html.parser")
    news_table = soup.find('table', class_='fullview-news-outer')
    
    data = []
    if not news_table:
        raise ValueError(f"Could not find news table for {ticker}. Finviz layout may have changed or ticker is invalid.")
        
    for row in news_table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) == 2:
            timestamp = cols[0].text.strip()
            headline_tag = cols[1].a
            if headline_tag:
                headline = headline_tag.text.strip()
                
                # Handle Finviz date format
                if " " in timestamp:
                    parts = timestamp.split(" ", 1)
                    if len(parts) == 2:
                        date_str, time_str = parts[0], parts[1]
                    else:
                        date_str, time_str = None, timestamp
                else:
                    date_str, time_str = None, timestamp
                    
                data.append({"Date": date_str, "Time": time_str, "Headline": headline})

    if not data:
        raise ValueError(f"No headlines found for {ticker}.")

    df_news = pd.DataFrame(data)
    
    # FIX: Modern pandas forward fill syntax replacing legacy .fillna(method='ffill')
    df_news['Date'] = df_news['Date'].ffill()
    
    # Convert dates and handle errors (creates NaT for bad dates)
    try:
        df_news['Date'] = pd.to_datetime(df_news['Date'], format='%b-%d-%y').dt.date
    except ValueError:
        df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce').dt.date
    
    # Remove rows where Date is NaT (Not a Time)
    df_news = df_news.dropna(subset=['Date'])
        
    return df_news

def analyze_and_merge(df_price, df_news):
    """Performs sentiment analysis and merges with price data."""
    # Calculate Compound Sentiment
    df_news['Sentiment'] = df_news['Headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Group by Date
    df_daily_sentiment = df_news.groupby('Date').agg({'Sentiment': 'mean'}).reset_index()
    
    # Merge with Price
    df_merged = pd.merge(df_price, df_daily_sentiment, on='Date', how='left')
    df_merged['Sentiment'] = df_merged['Sentiment'].fillna(0)
    
    # Calculate Next Day Return
    df_merged['NextDay_PctChange'] = df_merged['Close'].pct_change().shift(-1)
    
    # Calculate correlation
    data_for_corr = df_merged.dropna()
    if not data_for_corr.empty:
        correlation = data_for_corr['Sentiment'].corr(data_for_corr['NextDay_PctChange'])
    else:
        correlation = 0
        
    return df_merged, correlation, df_news[['Date', 'Time', 'Headline', 'Sentiment']]

def run_regression(df_merged):
    """Runs a simple linear regression and returns R2 score."""
    data = df_merged.dropna()
    if len(data) < 2:
        return 0 
        
    X = data[['Sentiment']]
    y = data['NextDay_PctChange']
    
    if len(data) < 10:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def create_plots(df_merged, ticker):
    """Creates plots and returns them as base64 encoded strings."""
    plots_base64 = []

    # --- Plot 1: Correlation Scatter Plot ---
    try:
        data_for_plot = df_merged.dropna()
        if not data_for_plot.empty:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(data_for_plot['Sentiment'], data_for_plot['NextDay_PctChange'], alpha=0.6, color='#1a73e8')
            ax1.set_title(f"Sentiment vs. Next-Day % Change ({ticker})")
            ax1.set_xlabel("Daily Sentiment Score (-1 to +1)")
            ax1.set_ylabel("Next-Day % Price Change")
            ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax1.axvline(0, color='grey', linewidth=0.8, linestyle='--')
            ax1.grid(True, alpha=0.3)
            
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", bbox_inches='tight')
            buf1.seek(0)
            plots_base64.append(base64.b64encode(buf1.read()).decode('ascii'))
            plt.close(fig1)
        else:
            plots_base64.append(None) 
    except Exception as e:
        print(f"Error creating plot 1: {e}")
        plots_base64.append(None)

    # --- Plot 2: Time Series Plot ---
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Stock Price ($)', color=color)
        ax2.plot(df_merged['Date'], df_merged['Close'], color=color, linewidth=2, label='Price')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax3 = ax2.twinx()  
        color = 'tab:orange'
        ax3.set_ylabel('Sentiment Score', color=color)
        ax3.plot(df_merged['Date'], df_merged['Sentiment'], color=color, linestyle='--', alpha=0.5, label='Sentiment')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_ylim(-1, 1) 
        
        plt.title(f"{ticker} — Price vs. Sentiment Trend")
        fig2.tight_layout()
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches='tight')
        buf2.seek(0)
        plots_base64.append(base64.b64encode(buf2.read()).decode('ascii'))
        plt.close(fig2)
    except Exception as e:
        print(f"Error creating plot 2: {e}")
        plots_base64.append(None)

    return plots_base64

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Sentiment Analyzer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        header { padding: 30px; border-bottom: 1px solid #e0e0e0; text-align: center; }
        header h1 { margin: 0; color: #1a73e8; }
        header p { margin: 5px 0 0; font-size: 1.1em; color: #5f6368; }
        main { padding: 30px; }
        form { display: flex; gap: 10px; margin-bottom: 30px; }
        input[type="text"] { flex-grow: 1; padding: 12px; font-size: 1em; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 12px 20px; font-size: 1em; background-color: #1a73e8; color: white; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
        button:hover { background-color: #185abc; }
        .results { border-top: 1px solid #e0e0e0; padding-top: 20px; }
        h2 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 5px; margin-top: 40px; }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; background: #f9f9f9; padding: 20px; border-radius: 4px; margin-bottom: 20px; }
        .stat p { margin: 0; font-size: 1.1em; color: #333; }
        .stat p strong { font-size: 1.4em; color: #185abc; display: block; margin-top: 4px; }
        .plot { text-align: center; margin-bottom: 30px; }
        .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }
        .error { background: #fdecea; color: #a94442; padding: 15px; border: 1px solid #f5c6cb; border-radius: 4px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-size: 0.9em; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }
        th { background-color: #f4f7f6; font-weight: 600; }
        tr:nth-child(even) { background-color: #fdfdfd; }
        tr:hover { background-color: #f1f1f1; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Stock News Sentiment Analyzer</h1>
            <p>Enter a stock ticker (e.g., AAPL, NVDA, TSLA) to analyze market sentiment.</p>
        </header>
        <main>
            <form action="/" method="POST">
                <input type="text" name="ticker" placeholder="Enter Ticker Symbol (e.g. AAPL)" required>
                <button type="submit">Analyze</button>
            </form>

            {% if error %}
                <div class="error">
                    <strong>Error:</strong> {{ error }}
                </div>
            {% endif %}

            {% if results %}
                <div class="results">
                    <h2>Analysis for {{ results.ticker }}</h2>
                    
                    <div class="stats">
                        <div class="stat">
                            <p>Correlation (Sentiment vs. Price)
                                <strong>{{ "%.4f"|format(results.correlation) }}</strong>
                            </p>
                        </div>
                        <div class="stat">
                            <p>Prediction Score (R²)
                                <strong>{{ "%.4f"|format(results.r2) }}</strong>
                            </p>
                        </div>
                    </div>

                    <div class="plot">
                        <h3>Price vs. Sentiment Trend</h3>
                        {% if results.plot_timeseries %}
                            <img src="data:image/png;base64,{{ results.plot_timeseries }}">
                        {% else %}
                            <p>Time series plot could not be generated.</p>
                        {% endif %}
                    </div>
                    
                    <div class="plot">
                        <h3>Correlation Analysis</h3>
                        {% if results.plot_correlation %}
                            <img src="data:image/png;base64,{{ results.plot_correlation }}">
                        {% else %}
                            <p>Correlation plot could not be generated.</p>
                        {% endif %}
                    </div>

                    <h2>Recent Headlines</h2>
                    <div style="overflow-x: auto;">
                        {{ results.headlines_table | safe }}
                    </div>
                </div>
            {% endif %}
        </main>
    </div>
</body>
</html>
"""

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper().strip()
        if not ticker:
            return render_template_string(HTML_TEMPLATE, error="Ticker symbol cannot be empty.")
            
        try:
            df_price = fetch_stock_data(ticker)
            df_news = fetch_headlines(ticker)
            df_merged, correlation, df_headlines_sentiment = analyze_and_merge(df_price, df_news)
            r2 = run_regression(df_merged)
            plot_corr, plot_ts = create_plots(df_merged, ticker)
            
            results = {
                'ticker': ticker,
                'correlation': correlation,
                'r2': r2,
                'plot_correlation': plot_corr,
                'plot_timeseries': plot_ts,
                'headlines_table': df_headlines_sentiment.head(15).to_html(classes='table', index=False, float_format='{:.4f}'.format)
            }
            return render_template_string(HTML_TEMPLATE, results=results)

        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=str(e))

    return render_template_string(HTML_TEMPLATE, results=None, error=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
