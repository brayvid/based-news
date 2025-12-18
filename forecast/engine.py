import os
import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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
TICKER = "SPY"
MARKET_TIMEZONE = "America/New_York"
MARKET_CALENDAR = "XNYS" # Updated to correct ISO code

# Download VADER lexicon if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Silence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

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
        # Use simple query (Pandas read_sql warning is harmless here)
        df = pd.read_sql("SELECT id, topic, title, pub_date FROM articles ORDER BY pub_date ASC", conn)
    finally:
        conn.close()

    if df.empty: return pd.DataFrame(), pd.DataFrame()

    df['pub_date'] = pd.to_datetime(df['pub_date'], utc=True)
    
    # Split
    training_df = df[df['pub_date'] <= last_market_close_utc]
    prediction_df = df[df['pub_date'] > last_market_close_utc]
    
    print(f"   - Historical Articles: {len(training_df)}")
    print(f"   - New Articles (to predict): {len(prediction_df)}")
    
    return training_df, prediction_df

def get_sentiment_score(texts):
    """Calculates average compound sentiment for a list (or Series) of headlines."""
    sia = SentimentIntensityAnalyzer()
    
    # FIX: Convert Pandas Series to list
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
        
    if not texts: return 0.0
    
    # Calculate scores ensuring inputs are strings
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
            # This calls the fixed function above
            Sentiment_Score=('title', get_sentiment_score),
            article_ids=('id', list),
            articles=('title', list),
            topics=('topic', list)
        ).reset_index()
        return grouped.sort_values('Date')
    else:
        # Prediction grouping
        return pd.DataFrame({
            'Date': [datetime.now(pytz.timezone(MARKET_TIMEZONE)).date()], 
            'News_Text': [' '.join(df['News_Text'])],
            # Manually convert to list for single-row dataframe creation
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

def add_market_context(df):
    print("2. Downloading market context...")
    if df.empty: return pd.DataFrame()
    
    ny_tz = pytz.timezone(MARKET_TIMEZONE)
    now_ny = datetime.now(ny_tz)
    
    start_date = pd.to_datetime(df['Date'].min()).date() - timedelta(days=730)
    end_date = (now_ny + timedelta(days=2)).date()
    
    tickers = [TICKER, "^VIX", "^TNX"]
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"   Error: {e}")
        return pd.DataFrame()

    if data.empty: return pd.DataFrame()

    try:
        spy = data[TICKER].copy()
        vix = data['^VIX'][['Close']].rename(columns={'Close': 'VIX_Close'})
        tnx = data['^TNX'][['Close']].rename(columns={'Close': 'TNX_Close'})
    except KeyError:
        spy = data.xs(TICKER, level=0, axis=1).copy()
        vix = data.xs('^VIX', level=0, axis=1)[['Close']].rename(columns={'Close': 'VIX_Close'})
        tnx = data.xs('^TNX', level=0, axis=1)[['Close']].rename(columns={'Close': 'TNX_Close'})

    market = spy.reset_index()
    market['Date'] = pd.to_datetime(market['Date']).dt.date
    vix = vix.reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.date
    tnx = tnx.reset_index()
    tnx['Date'] = pd.to_datetime(tnx['Date']).dt.date
    
    market = market.merge(vix, on='Date', how='left').merge(tnx, on='Date', how='left')
    market[['VIX_Close', 'TNX_Close']] = market[['VIX_Close', 'TNX_Close']].ffill()

    # --- INDICATORS ---
    market['RSI'] = calculate_rsi(market['Close'])
    market['SMA_50'] = market['Close'].rolling(window=50).mean()
    market['Dist_SMA_50'] = (market['Close'] - market['SMA_50']) / market['SMA_50']
    
    # Stationary features
    market['VIX_Change'] = market['VIX_Close'].pct_change()
    market['TNX_Change'] = market['TNX_Close'].pct_change()
    market['Vol_Change'] = market['Volume'].pct_change(fill_method=None)

    market = market.dropna(subset=['SMA_50', 'RSI', 'VIX_Change'])

    # Targets
    market['Future_Close'] = market['Close'].shift(-5)
    market['Target_1W'] = np.where(market['Future_Close'] > market['Close'], 1, 0)
    
    full_df = pd.merge(df, market, on='Date', how='inner')
    return full_df

def save_prediction(news_date, direction, confidence, evidence):
    print("5. Saving to DB...")
    evidence_headlines = [e['headline'] for e in evidence] + [None]*3
    evidence_ids = [e['id'] for e in evidence] + [None]*3
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (news_date, ticker, direction, confidence, evidence_1, evidence_1_id, evidence_2, evidence_2_id, evidence_3, evidence_3_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (news_date, ticker) DO UPDATE SET
                    direction = EXCLUDED.direction, confidence = EXCLUDED.confidence, updated_at = NOW();
            """, (news_date, TICKER, direction, float(confidence), evidence_headlines[0], evidence_ids[0], evidence_headlines[1], evidence_ids[1], evidence_headlines[2], evidence_ids[2]))
        conn.commit()
        print("   ✅ Saved.")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

def run_engine():
    cutoff = get_last_market_close_time()
    train_raw, pred_raw = get_news_and_split(cutoff)

    if pred_raw.empty:
        print("No new news. Exiting.")
        return

    train_grouped = process_and_group_news(train_raw, True)
    pred_grouped = process_and_group_news(pred_raw, False)

    labeled_data = add_market_context(train_grouped)
    training_df = labeled_data.dropna(subset=['Future_Close']).copy()
    
    if training_df.empty:
        print("Insufficient training data.")
        return

    print(f"3. Training Model (Samples: {len(training_df)})...")
    
    # Text Features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2, ngram_range=(1,2))
    X_text_train = vectorizer.fit_transform(training_df['News_Text'])
    
    # Numerical Features
    feature_cols = ['VIX_Change', 'TNX_Change', 'RSI', 'Dist_SMA_50', 'Vol_Change', 'Sentiment_Score']
    training_df[feature_cols] = training_df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(training_df[feature_cols])
    
    X_train = hstack([X_text_train, X_num_train])
    y_train = training_df['Target_1W']

    model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=10, 
        min_samples_leaf=4,
        class_weight='balanced_subsample', 
        random_state=42
    )
    model.fit(X_train, y_train)

    print("4. Predicting...")
    latest_market = labeled_data.iloc[-1]
    
    X_text_pred = vectorizer.transform(pred_grouped['News_Text'])
    
    latest_nums_df = pd.DataFrame([latest_market[feature_cols[:-1]].values], columns=feature_cols[:-1])
    latest_nums_df['Sentiment_Score'] = pred_grouped.iloc[0]['Sentiment_Score']
    
    X_num_pred = scaler.transform(latest_nums_df[feature_cols])
    X_pred = hstack([X_text_pred, X_num_pred])
    
    pred = model.predict(X_pred)[0]
    conf = model.predict_proba(X_pred)[0]
    
    direction = "UP" if pred == 1 else "DOWN"
    confidence = conf[1] if pred == 1 else conf[0]

    # Evidence
    importances = model.feature_importances_[:X_text_train.shape[1]]
    words = vectorizer.get_feature_names_out()
    word_weights = dict(zip(words, importances))
    
    evidence = []
    ids, titles, topics = pred_grouped.iloc[0][['article_ids', 'articles', 'topics']]
    
    sia = SentimentIntensityAnalyzer()

    for i in range(len(ids)):
        text = (f"[{topics[i]}] {titles[i]}").lower()
        tfidf_score = sum(word_weights.get(w, 0) for w in text.split())
        sent_score = abs(sia.polarity_scores(titles[i])['compound'])
        final_score = tfidf_score + (sent_score * 0.5)
        
        if final_score > 0: 
            evidence.append({'id': ids[i], 'headline': titles[i], 'score': final_score})
    
    top_evidence = sorted(evidence, key=lambda x: x['score'], reverse=True)[:3]

    print("\n" + "="*60)
    print(f"      AI FORECAST: {direction} ({confidence:.1%})")
    print(f"      Sentiment: {latest_nums_df['Sentiment_Score'].values[0]:.2f}")
    print(f"      VIX Chg: {latest_market['VIX_Change']:.1%}, RSI: {latest_market['RSI']:.1f}")
    print("="*60)
    
    save_prediction(pred_grouped.iloc[0]['Date'], direction, confidence, top_evidence)

if __name__ == "__main__":
    run_engine()