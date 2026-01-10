import os
import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pytz
import pandas_market_calendars as mcal
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
MARKET_TIMEZONE = "America/New_York"
MARKET_CALENDAR = "XNYS"
SECTOR_LOOKAHEAD_DAYS = 20
CRASH_LOOKAHEAD_DAYS = 10
CRASH_THRESHOLD = -0.05
MARKET_TICKER = "SPY"
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

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
    if hasattr(texts, 'tolist'): texts = texts.tolist()
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
            Sentiment_Score=('title', get_sentiment_score)
        ).reset_index()
        return grouped.sort_values('Date')
    else:
        return pd.DataFrame({
            'Date': [datetime.now(pytz.timezone(MARKET_TIMEZONE)).date()], 
            'News_Text': [' '.join(df['News_Text'])],
            'Sentiment_Score': [get_sentiment_score(df['title'].tolist())]
        })

def download_market_data(min_date):
    print("2. Downloading market data...")
    ny_tz = pytz.timezone(MARKET_TIMEZONE)
    now_ny = datetime.now(ny_tz)
    start_date = min_date - timedelta(days=365*5)
    end_date = (now_ny + timedelta(days=5)).date()
    
    tickers = SECTORS + [MARKET_TICKER, "^VIX", "^TNX"]
    print(f"   - Fetching {len(tickers)} tickers from {start_date} to {end_date}")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
    
    # --- DIAGNOSTIC BLOCK ---
    print("\n   [DIAGNOSTIC] Market Data Shape:", data.shape)
    if isinstance(data.columns, pd.MultiIndex):
        downloaded_tickers = data.columns.levels[0].tolist()
        print(f"   [DIAGNOSTIC] Tickers found in data: {downloaded_tickers}")
        missing = [t for t in tickers if t not in downloaded_tickers]
        if missing: print(f"   [DIAGNOSTIC] ⚠️ MISSING TICKERS: {missing}")
    else:
        print("   [DIAGNOSTIC] ⚠️ Data is not MultiIndex. Check yfinance version.")
    # ------------------------
    
    return data

def process_ticker_data(ticker, raw_data, news_df, lookahead_days, is_crash_mode=False):
    # Retrieve DataFrame
    if ticker in raw_data:
        df = raw_data[ticker].copy()
    elif ticker in raw_data.columns.get_level_values(0):
        df = raw_data[ticker].copy()
    else:
        print(f"     [DEBUG] {ticker}: Not in Market Data.")
        return pd.DataFrame()

    # Macro
    try:
        vix = raw_data['^VIX'][['Close']].rename(columns={'Close': 'VIX_Close'}) if '^VIX' in raw_data else pd.DataFrame()
        tnx = raw_data['^TNX'][['Close']].rename(columns={'Close': 'TNX_Close'}) if '^TNX' in raw_data else pd.DataFrame()
    except:
        vix, tnx = pd.DataFrame(), pd.DataFrame()
    
    if vix.empty or tnx.empty:
        # Try MultiIndex access fallback
        try:
            vix = raw_data['^VIX'][['Close']].rename(columns={'Close': 'VIX_Close'})
            tnx = raw_data['^TNX'][['Close']].rename(columns={'Close': 'TNX_Close'})
        except:
            print(f"     [DEBUG] {ticker}: Macro data missing.")
            return pd.DataFrame()

    df = df.reset_index()
    vix = vix.reset_index()
    tnx = tnx.reset_index()
    
    # Normalize Dates
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    tnx['Date'] = pd.to_datetime(tnx['Date']).dt.tz_localize(None)
    
    df = df.merge(vix, on='Date', how='left').merge(tnx, on='Date', how='left').ffill()

    # Indicators
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).where(lambda x: x > 0, 0).rolling(14).mean() / (-df['Close'].diff(1).where(lambda x: x < 0, 0).rolling(14).mean()))))
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['VIX_Change'] = df['VIX_Close'].pct_change()
    
    # Target
    if is_crash_mode:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=lookahead_days)
        df['Future_Low'] = df['Low'].rolling(window=indexer).min()
        df['Target'] = ((df['Future_Low'] - df['Close']) / df['Close'] < CRASH_THRESHOLD).astype(int)
    else:
        df['Target'] = (df['Close'].shift(-lookahead_days) - df['Close']) / df['Close']

    df = df.dropna(subset=['RSI', 'Target', 'Dist_SMA_50'])
    
    # Merge
    df['Date_Only'] = df['Date'].dt.date
    full_df = pd.merge(news_df, df, left_on='Date', right_on='Date_Only', how='inner')
    
    if full_df.empty:
        # DIAGNOSTIC PRINT FOR FIRST FAILURE
        print(f"     [DEBUG] {ticker}: Merge result is EMPTY.")
        print(f"             News Range: {news_df['Date'].min()} to {news_df['Date'].max()}")
        print(f"             Mkt  Range: {df['Date_Only'].min()} to {df['Date_Only'].max()}")
        return pd.DataFrame()
        
    return full_df

def run_engine():
    cutoff = get_last_market_close_time()
    train_raw, pred_raw = get_news_and_split(cutoff)
    if pred_raw.empty: return

    train_grouped = process_and_group_news(train_raw, True)
    pred_grouped = process_and_group_news(pred_raw, False)
    
    # Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectorizer.fit(train_grouped['News_Text'])
    
    # Download
    min_date = pd.to_datetime(train_grouped['Date'].min()).date()
    raw_market_data = download_market_data(min_date)

    # 1. CRASH
    print("\n3. Training Crash Model...")
    df_spy = process_ticker_data(MARKET_TICKER, raw_market_data, train_grouped, CRASH_LOOKAHEAD_DAYS, True)
    if not df_spy.empty:
        X = hstack([vectorizer.transform(df_spy['News_Text']), StandardScaler().fit_transform(df_spy[['VIX_Close', 'VIX_Change', 'RSI', 'Dist_SMA_50', 'Sentiment_Score']])])
        y = df_spy['Target']
        if len(np.unique(y)) > 1:
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X, y)
            
            # Predict
            latest = process_ticker_data(MARKET_TICKER, raw_market_data, train_grouped, CRASH_LOOKAHEAD_DAYS, True).iloc[-1]
            latest_features = pd.DataFrame([latest[['VIX_Close', 'VIX_Change', 'RSI', 'Dist_SMA_50']].values], columns=['VIX_Close', 'VIX_Change', 'RSI', 'Dist_SMA_50'])
            latest_features['Sentiment_Score'] = pred_grouped.iloc[0]['Sentiment_Score']
            
            X_pred = hstack([vectorizer.transform(pred_grouped['News_Text']), StandardScaler().fit(df_spy[['VIX_Close', 'VIX_Change', 'RSI', 'Dist_SMA_50', 'Sentiment_Score']]).transform(latest_features)])
            
            crash_prob = clf.predict_proba(X_pred)[0][list(clf.classes_).index(1)]
            print(f"   > Crash Probability: {crash_prob:.1%}")
            
            if crash_prob > 0.5:
                save_prediction(pred_grouped.iloc[0]['Date'], "SPY", "DOWN", crash_prob, "Crash Predicted")
        else:
            print("   > History contains no crashes. Market assumed stable.")
    else:
        print("   > Not enough SPY data.")

    # 2. SECTORS
    print("\n4. Ranking Sectors...")
    rankings = []
    for ticker in SECTORS:
        df = process_ticker_data(ticker, raw_market_data, train_grouped, SECTOR_LOOKAHEAD_DAYS, False)
        if df.empty: continue
        
        # Simple Train/Predict
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[['VIX_Change', 'RSI', 'Dist_SMA_50', 'Sentiment_Score']])
        X = hstack([vectorizer.transform(df['News_Text']), X_num])
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['Target'])
        
        # Predict
        latest = df.iloc[-1]
        feats = pd.DataFrame([latest[['VIX_Change', 'RSI', 'Dist_SMA_50']].values], columns=['VIX_Change', 'RSI', 'Dist_SMA_50'])
        feats['Sentiment_Score'] = pred_grouped.iloc[0]['Sentiment_Score']
        X_p = hstack([vectorizer.transform(pred_grouped['News_Text']), scaler.transform(feats)])
        
        pred = model.predict(X_p)[0]
        rankings.append({'Sector': ticker, 'Return': pred})
        print(f"   - {ticker}: {pred:.2%}")

    if rankings:
        rankings.sort(key=lambda x: x['Return'], reverse=True)
        top = rankings[0]
        print(f"\n   WINNER: {top['Sector']} ({top['Return']:.2%})")
        save_prediction(pred_grouped.iloc[0]['Date'], top['Sector'], "UP" if top['Return']>0 else "DOWN", 0.8, "Top Sector")
    else:
        print("\n   ! No rankings generated.")

def save_prediction(news_date, ticker, direction, confidence, note):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (news_date, ticker, direction, confidence, evidence_1)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (news_date, ticker) DO UPDATE SET
                    direction = EXCLUDED.direction, confidence = EXCLUDED.confidence, updated_at = NOW();
            """, (news_date, ticker, direction, float(confidence), note))
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    run_engine()