# based-news/web/app.py

import os
import html
from collections import OrderedDict
from datetime import datetime, timedelta
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

TICKER = "SPY"

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
    # Use a DictCursor to get results as dictionaries
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Fetch the last 10 predictions
    cur.execute("SELECT * FROM predictions WHERE ticker = %s ORDER BY news_date DESC LIMIT 10", (TICKER,))
    predictions_raw = cur.fetchall()
    cur.close()
    conn.close()

    predictions_list = []
    for prediction in predictions_raw:
        pred_dict = dict(prediction)
        
        # Define the date range for the historical chart data
        prediction_date = pred_dict['news_date']
        end_date = prediction_date
        start_date = end_date - timedelta(days=35) # Fetch ~1 month of data
        
        try:
            # Fetch data from yfinance
            market_data = yf.download(
                TICKER,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            # --- FIX FOR YFINANCE MULTI-INDEX ISSUE ---
            # If yfinance returns columns like ('Close', 'SPY'), flatten them to just 'Close'
            if isinstance(market_data.columns, pd.MultiIndex):
                market_data.columns = market_data.columns.get_level_values(0)
            # ------------------------------------------

            # Format the data for Chart.js if the download was successful
            if not market_data.empty:
                market_data = market_data.reset_index()
                
                # Chart.js expects UTC timestamps in milliseconds
                timestamps = (market_data['Date'].astype(np.int64) // 10**6).tolist()
                prices = market_data['Close'].tolist()
                
                # Create the points list in the format {x, y}
                chart_points = [{'x': ts, 'y': price} for ts, price in zip(timestamps, prices)]
                
                # Attach the chart data to the prediction dictionary
                pred_dict['chart_data'] = {'points': chart_points}
            else:
                 pred_dict['chart_data'] = {'points': []}

        except Exception as e:
            # Print the specific error to the console for debugging
            print(f"Error processing yfinance data for {prediction_date}: {e}")
            pred_dict['chart_data'] = {'points': []}
        
        # Add a nicely formatted date string for the header
        pred_dict['news_date_formatted'] = prediction_date.strftime('%A, %B %d, %Y')
        
        predictions_list.append(pred_dict)
        
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