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

# --- NEW DEPENDENCIES ---
import pytz
import pandas_market_calendars as mcal

# --- CONFIGURATION ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
TICKER = "SPY"
MARKET_TIMEZONE = "America/New_York"
MARKET_CALENDAR = "NYSE"

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(DATABASE_URL)

def get_last_market_close_time():
    """
    Calculates the precise UTC timestamp of the last market close.
    Handles weekends and holidays automatically.
    """
    nyse = mcal.get_calendar(MARKET_CALENDAR)
    now_et = datetime.now(pytz.timezone(MARKET_TIMEZONE))
    
    # Get the schedule for the last 10 days to ensure we catch the last valid trading day
    schedule = nyse.schedule(start_date=now_et.date() - timedelta(days=10), end_date=now_et.date())
    
    # The last market close is the 'market_close' of the last day in the schedule
    # that is strictly before the current time.
    # Note: If running exactly AT close, this effectively treats the current close as valid.
    valid_schedule = schedule[schedule['market_close'] <= now_et]
    
    if valid_schedule.empty:
        # Fallback for extreme edge cases (e.g. fresh year start), usually unlikely
        return now_et.astimezone(pytz.utc) - timedelta(days=1)

    last_valid_day = valid_schedule.iloc[-1]
    last_market_close_et = last_valid_day['market_close']
    
    # Convert to UTC for database comparison
    last_market_close_utc = last_market_close_et.to_pydatetime().astimezone(pytz.utc)
    
    return last_market_close_utc

def get_news_and_split(last_market_close_utc):
    """
    Loads all articles from the DB and splits them into training and prediction sets
    based on the last market close time.
    """
    print(f"1. Loading all news from DB and splitting based on last market close: {last_market_close_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT id, topic, title, pub_date FROM articles ORDER BY pub_date ASC", conn)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Ensure pub_date is timezone-aware (UTC)
    df['pub_date'] = pd.to_datetime(df['pub_date'], utc=True)

    # Split data:
    training_df_raw = df[df['pub_date'] <= last_market_close_utc]
    prediction_df_raw = df[df['pub_date'] > last_market_close_utc]
    
    print(f"   - Found {len(training_df_raw)} articles for training.")
    print(f"   - Found {len(prediction_df_raw)} new articles for prediction.")
    
    return training_df_raw, prediction_df_raw

def process_and_group_news(df, is_for_training=True):
    """
    Processes and aggregates a dataframe of news articles.
    """
    if df.empty:
        return pd.DataFrame()

    df['topic'] = df['topic'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['News_Text'] = "[" + df['topic'] + "] " + df['title']
    
    if is_for_training:
        # For training, group by the US market date
        df['Date'] = df['pub_date'].dt.tz_convert(MARKET_TIMEZONE).dt.date
        grouped_df = df.groupby('Date').agg(
            News_Text=('News_Text', ' '.join),
            articles=('title', list),
            article_ids=('id', list),
            topics=('topic', list)
        ).reset_index()
        return grouped_df.sort_values('Date')
    else:
        # For prediction, aggregate all overnight/new news into a single row
        aggregated_data = {
            'Date': [datetime.now(pytz.timezone(MARKET_TIMEZONE)).date()], 
            'News_Text': [' '.join(df['News_Text'])],
            'articles': [list(df['title'])],
            'article_ids': [list(df['id'])],
            'topics': [list(df['topic'])]
        }
        return pd.DataFrame(aggregated_data)

def add_market_targets(df):
    """
    Aligns news data with market outcomes. 
    Explicitly ensures the YFinance download covers the most recent close,
    even immediately after market close.
    """
    print("2. Downloading market data to create training labels...")
    if df.empty:
        return pd.DataFrame()
    
    # 1. Setup Timezone and Dates
    # We use NY time to ensure 'today' is correct regardless of server time (UTC)
    ny_tz = pytz.timezone(MARKET_TIMEZONE)
    now_ny = datetime.now(ny_tz)
    
    # Start date based on news
    start_date = pd.to_datetime(df['Date'].min()).date()
    
    # End date: Add 2 days to 'now' to ensure the 'end' parameter (which is exclusive)
    # definitely covers today.
    end_date = (now_ny + timedelta(days=2)).date()
    
    try:
        # 2. Bulk Download
        market = yf.download(TICKER, start=start_date, end=end_date, progress=False, interval="1d")
        
        if market.empty: 
            raise ValueError("No data returned from yfinance.")
            
        # Handle MultiIndex columns (common in yfinance v0.2+)
        if isinstance(market.columns, pd.MultiIndex):
            market.columns = market.columns.get_level_values(0)
            
        market = market.reset_index()
        
        # Normalize Date format to match News Data
        # Using .dt.date removes time components/timezones for safe merging
        market['Date'] = pd.to_datetime(market['Date']).dt.date
        
        # 3. LATENCY FIX: Check if today's data is missing after market close
        # If it's a weekday, after 4:15 PM ET, and the last date in data is NOT today:
        last_date_in_data = market['Date'].max()
        is_weekday = now_ny.weekday() < 5 # 0-4 is Mon-Fri
        # Check if we are past 4:15 PM ET (giving 15 mins buffer for data to settle)
        is_post_close = now_ny.time() >= datetime.strptime("16:15", "%H:%M").time()
        
        if is_weekday and is_post_close and last_date_in_data < now_ny.date():
            print(f"   - Bulk download missing today ({now_ny.date()}). Fetching latest update manually...")
            
            # Force fetch the specific day
            ticker_obj = yf.Ticker(TICKER)
            todays_data = ticker_obj.history(period="1d")
            
            if not todays_data.empty:
                todays_data = todays_data.reset_index()
                todays_data['Date'] = pd.to_datetime(todays_data['Date']).dt.date
                
                # Verify the fetched data is actually for today
                if todays_data['Date'].iloc[0] == now_ny.date():
                    # Align columns (keep only what's in market df)
                    common_cols = market.columns.intersection(todays_data.columns)
                    todays_row = todays_data[common_cols]
                    
                    # Append to main dataframe
                    market = pd.concat([market, todays_row], ignore_index=True)
                    # Drop duplicates just in case to be safe
                    market = market.drop_duplicates(subset=['Date'], keep='last')
                    print("   - Successfully appended today's close.")

    except Exception as e:
        print(f"   ERROR: yfinance download failed: {e}")
        return pd.DataFrame()

    # Log the latest market date found
    last_market_date = market['Date'].max()
    print(f"   - Latest market close available: {last_market_date}")

    # Create Targets
    # We want to predict if price goes up in 5 days (1 Week)
    market['Future_Close'] = market['Close'].shift(-5)
    market['Target_1W'] = np.where(market['Future_Close'] > market['Close'], 1, 0)

    # Merge News with Market Data
    # Inner merge ensures we only train on days where we have both News AND Price history
    full_df = pd.merge(df, market[['Date', 'Close', 'Future_Close', 'Target_1W']], on='Date', how='inner')
    
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
                ON CONFLICT (news_date, ticker) DO UPDATE SET
                    direction = EXCLUDED.direction,
                    confidence = EXCLUDED.confidence,
                    evidence_1 = EXCLUDED.evidence_1,
                    evidence_1_id = EXCLUDED.evidence_1_id,
                    evidence_2 = EXCLUDED.evidence_2,
                    evidence_2_id = EXCLUDED.evidence_2_id,
                    evidence_3 = EXCLUDED.evidence_3,
                    evidence_3_id = EXCLUDED.evidence_3_id,
                    updated_at = NOW();
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
    """Main function to train on historical data and predict on news since last close."""
    
    # 1. Determine the exact cutoff time: the last market close.
    last_market_close_utc = get_last_market_close_time()
    
    # 2. Fetch all news and split into raw training/prediction sets.
    training_df_raw, prediction_df_raw = get_news_and_split(last_market_close_utc)

    # Exit if there's no new news to predict on.
    if prediction_df_raw.empty:
        print("No new headlines found since last market close. Nothing to predict. Exiting.")
        return

    # 3. Process and group the dataframes.
    training_data_grouped = process_and_group_news(training_df_raw, is_for_training=True)
    prediction_df_grouped = process_and_group_news(prediction_df_raw, is_for_training=False)

    # 4. Add market targets to historical data
    # Note: This will download prices up to "Now" to ensure we have the absolute latest context
    labeled_data = add_market_targets(training_data_grouped)

    # Filter for valid training data (must have a Future_Close known)
    # Rows from the last 5 days will be dropped here because we don't know the future yet.
    training_df = labeled_data.dropna(subset=['Future_Close'])
    
    # The single row of aggregated recent news
    prediction_row = prediction_df_grouped.iloc[0]

    if training_df.empty:
        print("Not enough historical data with known outcomes (5-day lag) to train. Exiting.")
        return

    # --- TRAINING ---
    print(f"3. Training model on {len(training_df)} historical market days...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2)
    X_train = vectorizer.fit_transform(training_df['News_Text'])
    y_train = training_df['Target_1W']

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("   ✅ Model training complete.")

    # --- PREDICTION ---
    print(f"4. Generating prediction for news published since {last_market_close_utc.strftime('%Y-%m-%d %H:%M')}")
    X_predict = vectorizer.transform([prediction_row['News_Text']])
    
    pred = model.predict(X_predict)[0]
    probs = model.predict_proba(X_predict)[0]
    
    # --- EVIDENCE EXTRACTION ---
    importances = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    word_weights = {feature_names[i]: importances[i] for i in range(len(feature_names)) if importances[i] > 0.001}
    
    evidence_scores = []
    pred_day_articles = pd.DataFrame({
        'id': prediction_row['article_ids'],
        'topic': prediction_row['topics'],
        'title': prediction_row['articles']
    })

    for index, row in pred_day_articles.iterrows():
        headline_text = ("[" + str(row['topic']) + "] " + str(row['title'])).lower()
        score = sum(word_weights.get(word, 0) for word in headline_text.split())
        if score > 0:
            evidence_scores.append({'id': int(row['id']), 'headline': str(row['title']), 'score': score})
            
    top_evidence = sorted(evidence_scores, key=lambda x: x['score'], reverse=True)[:3]

    # Format Results
    news_date = prediction_row['Date']
    direction = "UP" if pred == 1 else "DOWN"
    confidence = probs[1] if pred == 1 else probs[0]
    
    # --- SAVE & OUTPUT ---
    save_prediction_to_db(news_date, direction, confidence, top_evidence)
    
    print("\n" + "="*60)
    print(f"      AI 1-WEEK MARKET FORECAST ({TICKER})")
    print(f"      Prediction Date: {news_date}")
    print(f"      Articles Processed: {len(pred_day_articles)}")
    print("="*60)
    print(f"      Direction:  {direction}")
    print(f"      Confidence: {confidence:.2%}")
    print("\n      Top Evidence Headlines:")
    for i, ev in enumerate(top_evidence):
        print(f"        {i+1}. {ev['headline']} (id: {ev['id']})")
    print("="*60)

if __name__ == "__main__":
    run_engine()