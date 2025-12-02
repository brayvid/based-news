# based-news/web/app.py

import os
import html
from collections import OrderedDict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import psycopg2
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, jsonify, abort, redirect, url_for

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

# Get timezone from environment or use a default
USER_TIMEZONE = os.environ.get("TIMEZONE", "America/New_York")
try:
    ZONE = ZoneInfo(USER_TIMEZONE)
except Exception:
    ZONE = ZoneInfo("America/New_York")

def get_db_connection():
    """Establishes a connection to the database."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def to_user_timezone(dt):
    """Converts a datetime object to the user's local timezone."""
    if dt and dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZONE) if dt else None

def format_digest_data(articles_from_db):
    """Groups articles by topic and formats them for display."""
    digest_data = OrderedDict()
    for article in articles_from_db:
        topic = article[2]
        if topic not in digest_data:
            digest_data[topic] = []
        try:
            pub_dt_user_tz = to_user_timezone(article[5])
            date_str = pub_dt_user_tz.strftime("%a, %d %b %Y %I:%M %p %Z") if pub_dt_user_tz else "Date unavailable"
        except Exception:
            date_str = "Date unavailable"
        digest_data[topic].append({
            "title": article[3], "link": article[4], "date_str": date_str
        })
    return digest_data

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM digests ORDER BY created_at DESC LIMIT 1")
    latest_digest_row = cur.fetchone()
    latest_digest_data = {}
    if latest_digest_row:
        latest_digest_id = latest_digest_row[0]
        cur.execute("SELECT * FROM articles WHERE digest_id = %s ORDER BY display_order ASC", (latest_digest_id,))
        articles = cur.fetchall()
        latest_digest_data = format_digest_data(articles)
    cur.close()
    conn.close()
    return render_template('index.html', latest_digest=latest_digest_data)

@app.route('/forecast') # <-- Renamed route to match your file
def forecast_dashboard():
    """Serves the AI predictions dashboard with a simple daily close price chart."""
    conn = get_db_connection()
    cur = conn.cursor()

    # --- 1. FETCH PREDICTIONS ---
    cur.execute(
        """
        SELECT 
            news_date, direction, confidence,
            evidence_1, evidence_1_id, evidence_2, evidence_2_id, evidence_3, evidence_3_id
        FROM predictions
        ORDER BY prediction_date DESC
        LIMIT 20;
        """
    )
    results = cur.fetchall()
    
    if not results:
        return render_template('forecast.html', predictions=[])

    # --- 2. ONE EFFICIENT & ROBUST DOWNLOAD ---
    master_spy_data = pd.DataFrame()
    try:
        print("INFO: Fetching last 3 months of daily SPY data...")
        master_spy_data = yf.download("SPY", period="3mo", interval="1d", progress=False)
        
        # Add the data cleaning step to handle the multi-level index bug
        if isinstance(master_spy_data.columns, pd.MultiIndex):
            master_spy_data.columns = master_spy_data.columns.get_level_values(0)
        
        if master_spy_data.empty or not pd.api.types.is_numeric_dtype(master_spy_data['Close']):
            print("WARNING: Master SPY data download failed or was invalid.")
            master_spy_data = pd.DataFrame()
    except Exception as e:
        print(f"YFinance master download failed: {e}")

    # --- 3. PROCESS PREDICTIONS & FILTER CHART DATA ---
    predictions_list = []
    for row in results:
        news_date_obj = row[0]
        
        chart_data = {'points': []}
        if not master_spy_data.empty:
            # Filter master data to show prices ON OR BEFORE the prediction's date
            point_in_time_data = master_spy_data[master_spy_data.index.date <= news_date_obj]
            
            if not point_in_time_data.empty:
                points = []
                for index, row_price in point_in_time_data.iterrows():
                    points.append({
                        'x': int(index.value // 1_000_000),
                        'y': float(round(row_price['Close'], 2))
                    })
                chart_data['points'] = points

        prediction = {
            "news_date": news_date_obj.strftime('%Y-%m-%d'),
            "news_date_formatted": news_date_obj.strftime('%B %d, %Y'),
            "direction": str(row[1]),
            "confidence": float(row[2]),
            "evidence": [
                {"headline": str(row[3]) if row[3] else None, "id": int(row[4]) if row[4] else None},
                {"headline": str(row[5]) if row[5] else None, "id": int(row[6]) if row[6] else None},
                {"headline": str(row[7]) if row[7] else None, "id": int(row[8]) if row[8] else None},
            ],
            "chart_data": chart_data
        }
        predictions_list.append(prediction)
    
    cur.close()
    conn.close()

    return render_template('forecast.html', predictions=predictions_list)

@app.route('/manifest')
def get_manifest():
    conn = get_db_connection()
    cur = conn.cursor()
    max_history = int(os.environ.get("MAX_HISTORY_DIGESTS", 12))
    cur.execute("SELECT id, created_at FROM digests ORDER BY created_at DESC LIMIT %s", (max_history,))
    digests = cur.fetchall()
    digests.reverse()
    cur.close()
    conn.close()
    manifest = [{"id": row[0], "timestamp": row[1].isoformat()} for row in digests]
    return jsonify(manifest)

@app.route('/digest/<int:digest_id>')
def get_digest_html(digest_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM articles WHERE digest_id = %s ORDER BY display_order ASC", (digest_id,))
    articles = cur.fetchall()
    cur.close()
    conn.close()
    if not articles: abort(404, description="Digest not found")
    digest_data = format_digest_data(articles)
    html_parts = []
    for topic, articles_list in digest_data.items():
        html_parts.append(f"<h3>{html.escape(topic)}</h3>\n")
        for article in articles_list:
            html_parts.append(f'<p><a href="{html.escape(article["link"])}" target="_blank">{html.escape(article["title"])}</a><br><small>{article["date_str"]}</small></p>\n')
    return "".join(html_parts)

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))