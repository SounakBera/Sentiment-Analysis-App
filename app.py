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

# --- HTML Template (Enhanced UI with Tailwind CSS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-50">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PulseMarket // Sentiment Analyzer</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        
        /* Custom scrollbar for data tables */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f5f9; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

        /* Override the pandas default HTML table rendering to blend beautifully with Tailwind */
        .table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.875rem;
        }
        .table th {
            background-color: #f8fafc;
            color: #475569;
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 2px solid #e2e8f0;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        .table td {
            padding: 12px 16px;
            border-bottom: 1px solid #f1f5f9;
            color: #334155;
        }
        .table tr:hover {
            background-color: #f8fafc;
        }
    </style>
</head>
<body class="flex flex-col min-h-screen text-slate-800">

    <nav class="bg-white border-b border-slate-200 sticky top-0 z-50 backdrop-blur-md bg-white/90">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16 items-center">
                <div class="flex items-center gap-3">
                    <div class="bg-blue-600 text-white p-2 rounded-xl shadow-md shadow-blue-200">
                        <i class="fa-solid fa-chart-line text-lg"></i>
                    </div>
                    <span class="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">PulseMarket</span>
                </div>
                <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Engine Active
                </div>
            </div>
        </div>
    </nav>

    <main class="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10">
        
        <div class="bg-white rounded-2xl border border-slate-200 p-6 md:p-8 shadow-sm mb-8 transition-all hover:shadow-md">
            <div class="max-w-2xl">
                <h1 class="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight mb-2">Predictive Sentiment Terminal</h1>
                <p class="text-slate-500 text-sm md:text-base mb-6">Scrapes real-time financial headlines, evaluates aggregate psychological movement using VADER lexicons, and cross-references data against shifting equity targets.</p>
            </div>
            
            <form action="/" method="POST" class="flex flex-col sm:flex-row gap-3 max-w-xl">
                <div class="relative flex-grow">
                    <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-400">
                        <i class="fa-solid fa-magnifying-glass"></i>
                    </div>
                    <input type="text" name="ticker" placeholder="Enter stock symbol (e.g., AAPL, NVDA, TSLA)" required
                           class="w-full pl-11 pr-4 py-3.5 bg-slate-50 border border-slate-200 rounded-xl font-medium placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 focus:bg-white transition-all text-sm uppercase tracking-wider">
                </div>
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 active:scale-95 text-white px-6 py-3.5 rounded-xl font-semibold text-sm transition-all shadow-lg shadow-blue-100 flex items-center justify-center gap-2 cursor-pointer">
                    <span>Analyze Assets</span>
                    <i class="fa-solid fa-arrow-right text-xs"></i>
                </button>
            </form>
        </div>

        {% if error %}
            <div class="bg-red-50 border border-red-200 text-red-800 rounded-xl p-4 mb-8 flex items-start gap-3">
                <i class="fa-solid fa-circle-exclamation text-lg mt-0.5 text-red-500"></i>
                <div>
                    <h4 class="font-semibold text-sm">Execution Interrupted</h4>
                    <p class="text-xs text-red-600 mt-1">{{ error }}</p>
                </div>
            </div>
        {% endif %}

        {% if results %}
            <div class="space-y-8">
                
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-widest text-blue-600 bg-blue-50 px-2.5 py-1 rounded-md">Analysis Target</span>
                        <h2 class="text-3xl font-extrabold text-slate-900 mt-1">${{ results.ticker }} <span class="text-lg font-normal text-slate-400">Equity Dossier</span></h2>
                    </div>
                    <div class="text-xs text-slate-400 sm:text-right">
                        <p><i class="fa-regular fa-clock mr-1"></i> Temporal Domain: 1 Year Historical Window</p>
                    </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between">
                        <div>
                            <p class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Directional Correlation</p>
                            <h3 class="text-3xl font-bold text-slate-800 tracking-tight">{{ "%.4f"|format(results.correlation) }}</h3>
                            <p class="text-xs text-slate-500 mt-1">Sentiment vs Next-Day Value Swings</p>
                        </div>
                        <div class="w-12 h-12 rounded-xl flex items-center justify-center text-xl {{ 'bg-emerald-50 text-emerald-600' if results.correlation >= 0 else 'bg-rose-50 text-rose-600' }}">
                            <i class="fa-solid {{ 'fa-arrow-trend-up' if results.correlation >= 0 else 'fa-arrow-trend-down' }}"></i>
                        </div>
                    </div>

                    <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between">
                        <div>
                            <p class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Predictive Capacity Score (R²)</p>
                            <h3 class="text-3xl font-bold text-slate-800 tracking-tight">{{ "%.4f"|format(results.r2) }}</h3>
                            <p class="text-xs text-slate-500 mt-1">Variance explained via Linear Regression modeling</p>
                        </div>
                        <div class="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-xl flex items-center justify-center text-xl">
                            <i class="fa-solid fa-circle-nodes"></i>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-white p-5 md:p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                        <div class="flex items-center gap-2 mb-4">
                            <i class="fa-solid fa-chart-area text-slate-400 text-sm"></i>
                            <h4 class="font-bold text-sm tracking-tight text-slate-700">Price & Sentiment Synchronization</h4>
                        </div>
                        <div class="bg-slate-50 border border-slate-100 rounded-xl p-2 flex items-center justify-center flex-grow min-h-[300px]">
                            {% if results.plot_timeseries %}
                                <img src="data:image/png;base64,{{ results.plot_timeseries }}" class="max-w-full h-auto object-contain rounded-lg">
                            {% else %}
                                <span class="text-xs text-slate-400 font-medium">Visualization asset generation fault encountered.</span>
                            {% endif %}
                        </div>
                    </div>
                    
                    <div class="bg-white p-5 md:p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                        <div class="flex items-center gap-2 mb-4">
                            <i class="fa-solid fa-chart-line text-slate-400 text-sm"></i>
                            <h4 class="font-bold text-sm tracking-tight text-slate-700">Regression Distribution Mapping</h4>
                        </div>
                        <div class="bg-slate-50 border border-slate-100 rounded-xl p-2 flex items-center justify-center flex-grow min-h-[300px]">
                            {% if results.plot_correlation %}
                                <img src="data:image/png;base64,{{ results.plot_correlation }}" class="max-w-full h-auto object-contain rounded-lg">
                            {% else %}
                                <span class="text-xs text-slate-400 font-medium">Visualization asset generation fault encountered.</span>
                            {% endif %}
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                    <div class="p-5 md:p-6 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
                        <div class="flex items-center gap-2">
                            <i class="fa-regular fa-newspaper text-slate-400"></i>
                            <h4 class="font-bold text-sm tracking-tight text-slate-700">Scraped Media Narrative Stream (VADER Vectorized)</h4>
                        </div>
                        <span class="text-xs font-semibold text-slate-500 bg-white border border-slate-200 px-2.5 py-1 rounded-md">Top 15 Headers</span>
                    </div>
                    <div class="overflow-x-auto">
                        {{ results.headlines_table | safe }}
                    </div>
                </div>
                
            </div>
        {% endif %}
    </main>

    <footer class="bg-white border-t border-slate-200 mt-auto py-6">
        <div class="max-w-7xl mx-auto px-4 text-center text-xs text-slate-400 font-medium flex flex-col sm:flex-row justify-between items-center gap-2">
            <p>&copy; 2026 PulseMarket Terminal. Deployed via Gunicorn engine environments.</p>
            <div class="flex gap-4 text-slate-400">
                <span class="hover:text-slate-600 transition-colors cursor-pointer">Security Protocol Documentation</span>
                <span class="hover:text-slate-600 transition-colors cursor-pointer">REST Core Sandbox API</span>
            </div>
        </div>
    </footer>
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
