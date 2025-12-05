import os
import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- CONFIGURATION ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
TICKER = "SPY"

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(DATABASE_URL)

def get_all_news_data():
    """Loads all articles from the database."""
    print("1. Loading all historical news from Database...")
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT id, topic, title, pub_date FROM articles ORDER BY pub_date ASC", conn)
    finally:
        conn.close()

    if df.empty: return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['pub_date'], utc=True).dt.date
    df['topic'] = df['topic'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['News_Text'] = "[" + df['topic'] + "] " + df['title']
    
    # Group by Date
    daily_df = df.groupby('Date').agg(
        News_Text=('News_Text', ' '.join),
        # Keep a list of original articles for evidence extraction
        articles=('title', list),
        article_ids=('id', list),
        topics=('topic', list)
    ).reset_index()
    
    return daily_df.sort_values('Date')

def add_market_targets(df):
    """Aligns news data with market outcomes."""
    print("2. Downloading market data to create training labels...")
    start_date = pd.to_datetime(df['Date'].min())
    end_date = pd.to_datetime(df['Date'].max())
    
    try:
        market = yf.download(TICKER, start=start_date, end=end_date + timedelta(days=15), progress=False)
        if market.empty: raise ValueError("No data returned from yfinance.")
    except Exception as e:
        print(f"   ERROR: yfinance download failed: {e}")
        return pd.DataFrame()

    if isinstance(market.columns, pd.MultiIndex):
        market.columns = market.columns.get_level_values(0)
    market = market.reset_index()
    market['Date'] = pd.to_datetime(market['Date']).dt.date

    # Calculate 1-Week Target
    market['Future_Close'] = market['Close'].shift(-5)
    market['Target_1W'] = np.where(market['Future_Close'] > market['Close'], 1, 0)

    # Merge and clean
    full_df = pd.merge(df, market[['Date', 'Future_Close', 'Target_1W']], on='Date', how='inner')
    
    return full_df

def save_prediction_to_db(news_date, direction, confidence, evidence):
    """Inserts the prediction results into the 'predictions' table."""
    print("5. Saving prediction to database...")
    evidence_headlines = [item.get('headline') for item in evidence] + [None, None, None]
    evidence_ids = [item.get('id') for item in evidence] + [None, None, None]
    confidence_py_float = float(confidence)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (news_date, ticker, direction, confidence, evidence_1, evidence_1_id, evidence_2, evidence_2_id, evidence_3, evidence_3_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (news_date, TICKER, direction, confidence_py_float, evidence_headlines[0], evidence_ids[0], evidence_headlines[1], evidence_ids[1], evidence_headlines[2], evidence_ids[2])
            )
        conn.commit()
        print("   ✅ Prediction successfully logged.")
    except Exception as e:
        print(f"   ❌ Failed to save prediction: {e}")
        conn.rollback()
    finally:
        conn.close()

def run_engine():
    """Main function to train, predict, and save."""
    
    # --- DATA PREPARATION ---
    all_data = get_all_news_data()
    if all_data.empty:
        print("No news data found. Exiting.")
        return

    # Add market targets to historical data
    labeled_data = add_market_targets(all_data)

    # Separate training data from the prediction data
    # Training data is everything where we know the 1-week outcome
    training_df = labeled_data.dropna(subset=['Future_Close'])
    
    # Prediction data is the most recent day's news
    prediction_df = all_data.iloc[-1]

    if training_df.empty:
        print("Not enough historical data with known outcomes to train. Exiting.")
        return

    # --- TRAINING ---
    print(f"3. Training model on {len(training_df)} historical days...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2)
    X_train = vectorizer.fit_transform(training_df['News_Text'])
    y_train = training_df['Target_1W']

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("   ✅ Model training complete.")

    # --- PREDICTION ---
    print(f"4. Generating prediction for news from {prediction_df['Date']}...")
    X_predict = vectorizer.transform([prediction_df['News_Text']])
    
    pred = model.predict(X_predict)[0]
    probs = model.predict_proba(X_predict)[0]
    
    # Extract Evidence
    importances = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    word_weights = {feature_names[i]: importances[i] for i in range(len(feature_names)) if importances[i] > 0.001}
    
    evidence_scores = []
    # Recreate a small DataFrame for the prediction day to extract evidence
    pred_day_articles = pd.DataFrame({
        'id': prediction_df['article_ids'],
        'topic': prediction_df['topics'],
        'title': prediction_df['articles']
    })

    for index, row in pred_day_articles.iterrows():
        headline_text = ("[" + str(row['topic']) + "] " + str(row['title'])).lower()
        score = sum(word_weights.get(word, 0) for word in headline_text.split())
        if score > 0:
            evidence_scores.append({'id': int(row['id']), 'headline': str(row['title']), 'score': score})
            
    top_evidence = sorted(evidence_scores, key=lambda x: x['score'], reverse=True)[:3]

    # Format Results
    news_date = prediction_df['Date']
    direction = "UP" if pred == 1 else "DOWN"
    confidence = probs[1] if pred == 1 else probs[0]
    
    # --- SAVE & OUTPUT ---
    save_prediction_to_db(news_date, direction, confidence, top_evidence)
    
    print("\n" + "="*60)
    print(f"      AI 1-WEEK MARKET FORECAST ({TICKER})")
    print(f"      Data Date: {news_date}")
    print("="*60)
    # ... (rest of your print output) ...
    print("="*60)

if __name__ == "__main__":
    run_engine()