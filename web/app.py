# based-news/web/app.py

import os
import html
from collections import OrderedDict
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import psycopg2
import psycopg2.extras
import yfinance as yf
import pandas as pd
import numpy as np

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

@app.route('/forecast')
def forecast_page():
    """
    Renders the forecast page with predictions and their corresponding
    historical price charts.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # CHANGED: Removed "WHERE ticker = 'SPY'" so we see Sector picks too
    cur.execute("SELECT * FROM predictions ORDER BY news_date DESC LIMIT 10")
    predictions_raw = cur.fetchall()
    cur.close()
    conn.close()

    predictions_list = []
    
    # Current date for chart clamping
    today = datetime.now().date()

    for prediction in predictions_raw:
        pred_dict = dict(prediction)
        ticker = pred_dict['ticker']
        
        # --- DATE LOGIC FIX ---
        # Ensure we are working with a date object
        p_date = pred_dict['news_date']
        if isinstance(p_date, datetime):
            p_date = p_date.date()
        elif isinstance(p_date, str):
            p_date = datetime.strptime(p_date, '%Y-%m-%d').date()

        # Add nicely formatted date string
        pred_dict['news_date_formatted'] = p_date.strftime('%A, %B %d, %Y')

        # Logic: If prediction is in the future (2026), stop chart at TODAY (2025)
        # to prevent yfinance errors.
        if p_date > today:
            chart_end = today
        else:
            chart_end = p_date
        
        # Look back 45 days for context
        chart_start = chart_end - timedelta(days=45) 
        
        try:
            # Fetch data from yfinance
            # progress=False hides the console noise
            market_data = yf.download(
                ticker,
                start=chart_start,
                end=chart_end + timedelta(days=1), # +1 to include the end date
                progress=False,
                auto_adjust=True
            )

            # --- FIX FOR YFINANCE MULTI-INDEX ---
            if not market_data.empty:
                # If columns are MultiIndex (Price, Ticker), drop the ticker level
                if isinstance(market_data.columns, pd.MultiIndex):
                    market_data.columns = market_data.columns.get_level_values(0)

                market_data = market_data.reset_index()
                
                chart_points = []
                for _, row in market_data.iterrows():
                    # Extract timestamp (JS uses milliseconds)
                    ts = int(row['Date'].timestamp() * 1000)
                    
                    # Extract Close price safely (handle scalar vs series)
                    val = row['Close']
                    if hasattr(val, 'iloc'):
                        val = val.iloc[0]
                    
                    if pd.notna(val):
                        chart_points.append({'x': ts, 'y': float(val)})
                
                pred_dict['chart_data'] = {'points': chart_points}
            else:
                 pred_dict['chart_data'] = {'points': []}

        except Exception as e:
            print(f"Error processing yfinance data for {ticker}: {e}")
            pred_dict['chart_data'] = {'points': []}
        
        predictions_list.append(pred_dict)
        
    # NOTE: Ensure you saved the previous HTML I gave you as 'templates/forecast.html'
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