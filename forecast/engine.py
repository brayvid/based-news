import os
import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pytz
import pandas_market_calendars as mcal
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas.tseries.offsets import BQuarterEnd

# --- CONFIGURATION ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
MARKET_TIMEZONE = "America/New_York"
MARKET_CALENDAR = "XNYS"

# The 11 S&P 500 Select Sector SPDRs
SECTORS = [
    "XLC", # Communication Services
    "XLY", # Consumer Discretionary
    "XLP", # Consumer Staples
    "XLE", # Energy
    "XLF", # Financials
    "XLV", # Health Care
    "XLI", # Industrials
    "XLB", # Materials
    "XLRE", # Real Estate
    "XLK", # Technology
    "XLU"  # Utilities
]

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Silence warnings
warnings.filterwarnings('ignore')

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def get_last_market_close_time():
    nyse = mcal.get_calendar(MARKET_CALENDAR)
    now_et = datetime.now(pytz.timezone(MARKET_TIMEZONE))
    schedule = nyse.schedule(start_date=now_et.date() - timedelta(days=10), end_date=now_et.date())
    valid_schedule = schedule[schedule['market_close'] <= now_et]
    
    if valid_schedule.empty:
        return now_et.astimezone(pytz.utc) - timedelta(days=1)

    return valid_schedule.iloc[-1]['market_close'].to_pydatetime().astimezone(pytz.utc)

def get_news_and_split(last_market_close_utc):
    print(f"1. Loading news... Cutoff: {last_market_close_utc.strftime('%Y-%m-%d %H:%M')}")
    conn = get_db_connection()
    try:
        # Using raw cursor to avoid pandas/sqlalchemy warning if preferred, 
        # but read_sql is convenient. We suppressed warnings above.
        df = pd.read_sql("SELECT id, topic, title, pub_date FROM articles ORDER BY pub_date ASC", conn)
    finally:
        conn.close()

    if df.empty: return pd.DataFrame(), pd.DataFrame()

    df['pub_date'] = pd.to_datetime(df['pub_date'], utc=True)
    
    training_df = df[df['pub_date'] <= last_market_close_utc]
    prediction_df = df[df['pub_date'] > last_market_close_utc]
    
    print(f"   - Historical Articles: {len(training_df)}")
    print(f"   - New Articles: {len(prediction_df)}")
    
    return training_df, prediction_df

def get_sentiment_score(texts):
    sia = SentimentIntensityAnalyzer()
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
    if not texts: return 0.0
    scores = [sia.polarity_scores(str(t))['compound'] for t in texts]
    return np.mean(scores)

def process_and_group_news(df, is_for_training=True):
    if df.empty: return pd.DataFrame()
    
    df['topic'] = df['topic'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['News_Text'] = "[" + df['topic'] + "] " + df['title']
    
    if is_for_training:
        df['Date'] = df['pub_date'].dt.tz_convert(MARKET_TIMEZONE).dt.date
        grouped = df.groupby('Date').agg(
            News_Text=('News_Text', ' '.join),
            Sentiment_Score=('title', get_sentiment_score),
            article_ids=('id', list),
            articles=('title', list),
            topics=('topic', list)
        ).reset_index()
        return grouped.sort_values('Date')
    else:
        return pd.DataFrame({
            'Date': [datetime.now(pytz.timezone(MARKET_TIMEZONE)).date()], 
            'News_Text': [' '.join(df['News_Text'])],
            'Sentiment_Score': [get_sentiment_score(df['title'].tolist())],
            'article_ids': [list(df['id'])],
            'articles': [list(df['title'])],
            'topics': [list(df['topic'])]
        })

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def download_market_data(min_date):
    print("2. Downloading market data for all sectors...")
    ny_tz = pytz.timezone(MARKET_TIMEZONE)
    now_ny = datetime.now(ny_tz)
    
    start_date = min_date - timedelta(days=3650) # 10 years history
    end_date = (now_ny + timedelta(days=5)).date()
    
    # Download everything at once
    tickers_to_download = SECTORS + ["^VIX", "^TNX"]
    try:
        data = yf.download(tickers_to_download, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"   Error downloading: {e}")
        return pd.DataFrame()
        
    return data

def process_ticker_data(ticker, raw_data, news_df):
    """
    Extracts specific ticker data from the bulk download, calculates features, 
    and merges with news.
    """
    try:
        # Handle MultiIndex from yfinance
        df = raw_data[ticker].copy()
        
        # Get Macro Context
        vix = raw_data['^VIX'][['Close']].rename(columns={'Close': 'VIX_Close'})
        tnx = raw_data['^TNX'][['Close']].rename(columns={'Close': 'TNX_Close'})
    except KeyError:
        return pd.DataFrame() # Ticker missing

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']) # Ensure datetime64
    vix = vix.reset_index()
    vix['Date'] = pd.to_datetime(vix['Date'])
    tnx = tnx.reset_index()
    tnx['Date'] = pd.to_datetime(tnx['Date'])
    
    # Merge Macro
    df = df.merge(vix, on='Date', how='left').merge(tnx, on='Date', how='left')
    df[['VIX_Close', 'TNX_Close']] = df[['VIX_Close', 'TNX_Close']].ffill()

    # Features
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    df['VIX_Change'] = df['VIX_Close'].pct_change()
    df['TNX_Change'] = df['TNX_Close'].pct_change()

    # --- REGRESSION TARGET: % Return to Quarter End ---
    df['Next_QE_Date'] = df['Date'] + BQuarterEnd(startingMonth=3)
    
    price_lookup = df[['Date', 'Close']].sort_values('Date').rename(columns={'Close': 'QE_Close'})
    df = df.sort_values('Date')
    
    df = pd.merge_asof(
        df, 
        price_lookup, 
        left_on='Next_QE_Date', 
        right_on='Date', 
        direction='backward', 
        suffixes=('', '_Future')
    )
    
    df['Days_To_QE'] = (df['Next_QE_Date'] - df['Date']).dt.days
    
    # TARGET: Percentage Return from Today to Quarter End
    df['Target_Return'] = (df['QE_Close'] - df['Close']) / df['Close']
    
    df = df.dropna(subset=['SMA_200', 'RSI', 'Target_Return'])
    
    # Merge with News
    df['Date'] = df['Date'].dt.date
    full_df = pd.merge(news_df, df, on='Date', how='inner')
    
    return full_df

def train_and_predict_sector(ticker, raw_data, train_news, pred_news, vectorizer=None):
    """
    Trains a model specifically for this sector.
    Returns: Predicted Return (float), Model Importance (dict)
    """
    df = process_ticker_data(ticker, raw_data, train_news)
    
    if df.empty or len(df) < 50:
        return -999, None # Not enough data

    # Prepare Features
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500, min_df=2)
        X_text = vectorizer.fit_transform(df['News_Text'])
    else:
        X_text = vectorizer.transform(df['News_Text'])
        
    feature_cols = ['VIX_Change', 'TNX_Change', 'RSI', 'Dist_SMA_50', 'Dist_SMA_200', 'Sentiment_Score', 'Days_To_QE']
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[feature_cols])
    
    X = hstack([X_text, X_num])
    y = df['Target_Return'] # Regression Target

    # Train Regressor (Not Classifier)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42)
    model.fit(X, y)

    # --- PREDICT ---
    # Prepare Prediction Vector
    latest_market_row = process_ticker_data(ticker, raw_data, train_news).iloc[-1]
    
    # Calculate current days to QE for the prediction date
    pred_date = pd.to_datetime(pred_news.iloc[0]['Date'])
    qe_date = pred_date + BQuarterEnd(startingMonth=3)
    days_left = (qe_date - pred_date).days
    
    # Build Input
    X_text_pred = vectorizer.transform(pred_news['News_Text'])
    
    input_features = pd.DataFrame([latest_market_row[feature_cols[:-1]].values], columns=feature_cols[:-1])
    input_features['Sentiment_Score'] = pred_news.iloc[0]['Sentiment_Score']
    input_features['Days_To_QE'] = days_left
    # Reorder
    input_features = input_features[feature_cols]
    
    X_num_pred = scaler.transform(input_features)
    X_pred = hstack([X_text_pred, X_num_pred])
    
    predicted_return = model.predict(X_pred)[0]
    
    return predicted_return, vectorizer, qe_date.date()

def save_top_pick(news_date, top_sector, pred_return, qe_date):
    # We will save the #1 ranked sector as the prediction in the DB
    direction = "UP" if pred_return > 0 else "DOWN"
    # Map return to a pseudo-confidence (0.0 to 1.0) for schema compatibility
    # e.g., 5% return = 0.6 confidence, 10% = 0.8
    confidence = min(0.5 + (abs(pred_return) * 5), 0.99)
    
    print("5. Saving #1 Pick to DB...")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (news_date, ticker, direction, confidence, evidence_1)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (news_date, ticker) DO UPDATE SET
                    direction = EXCLUDED.direction, confidence = EXCLUDED.confidence, updated_at = NOW();
            """, (news_date, top_sector, direction, float(confidence), f"Projected Return: {pred_return:.2%} by {qe_date}"))
        conn.commit()
        print(f"   ✅ Saved Top Pick ({top_sector}) to DB.")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

def run_engine():
    cutoff = get_last_market_close_time()
    train_raw, pred_raw = get_news_and_split(cutoff)

    if pred_raw.empty:
        print("No new news.")
        return

    train_grouped = process_and_group_news(train_raw, True)
    pred_grouped = process_and_group_news(pred_raw, False)
    
    min_date = pd.to_datetime(train_grouped['Date'].min()).date()
    raw_market_data = download_market_data(min_date)

    print("3. Training Sector Models & Ranking...")
    
    rankings = []
    global_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2, ngram_range=(1,2))
    global_vectorizer.fit(train_grouped['News_Text'])
    
    target_qe_date = None

    for ticker in SECTORS:
        pred_ret, _, qe_date = train_and_predict_sector(ticker, raw_market_data, train_grouped, pred_grouped, global_vectorizer)
        
        if pred_ret != -999:
            rankings.append({
                'Sector': ticker,
                'Predicted_Return': pred_ret
            })
            target_qe_date = qe_date
            print(f"   - {ticker}: {pred_ret:.2%}")
        else:
            print(f"   - {ticker}: Insufficient Data")

    # Sort Rankings
    rankings.sort(key=lambda x: x['Predicted_Return'], reverse=True)
    
    print("\n" + "="*60)
    print(f"      SECTOR RANKINGS (Forecast to {target_qe_date})")
    print("="*60)
    print(f"      {'Rank':<5} {'Ticker':<10} {'Proj. Return':<15}")
    print("-" * 40)
    
    for i, item in enumerate(rankings):
        print(f"      #{i+1:<4} {item['Sector']:<10} {item['Predicted_Return']:+.2%}")
        
    print("="*60)

    # Save the winner
    if rankings:
        winner = rankings[0]
        save_top_pick(pred_grouped.iloc[0]['Date'], winner['Sector'], winner['Predicted_Return'], target_qe_date)

if __name__ == "__main__":
    run_engine()