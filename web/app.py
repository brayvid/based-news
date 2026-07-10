# based-news/web/app.py

import os
import html
import math
from collections import OrderedDict
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import psycopg2
import psycopg2.extras
import yfinance as yf
import pandas as pd
import numpy as np

from flask import Flask, render_template, jsonify, abort, redirect, url_for, request, send_from_directory

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
        p_date = pred_dict['news_date']
        if isinstance(p_date, datetime):
            p_date = p_date.date()
        elif isinstance(p_date, str):
            p_date = datetime.strptime(p_date, '%Y-%m-%d').date()

        # Format date dynamically without leading zeros
        pred_dict['news_date_formatted'] = p_date.strftime(f'%A, %B {p_date.day}, %Y')

        if p_date > today:
            chart_end = today
        else:
            chart_end = p_date
        
        chart_start = chart_end - timedelta(days=45) 
        
        try:
            market_data = yf.download(
                ticker,
                start=chart_start,
                end=chart_end + timedelta(days=1),
                progress=False,
                auto_adjust=True
            )

            if not market_data.empty:
                if isinstance(market_data.columns, pd.MultiIndex):
                    market_data.columns = market_data.columns.get_level_values(0)

                market_data = market_data.reset_index()
                
                chart_points = []
                for _, row in market_data.iterrows():
                    ts = int(row['Date'].timestamp() * 1000)
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
        # Changed <h3> topic-header to <h2> for sequential accessibility compliance
        html_parts.append(f'<h2 class="topic-header">{html.escape(topic)}</h2>\n')
        for article in articles_list:
            html_parts.append(f'<p><a href="{html.escape(article["link"])}" target="_blank">{html.escape(article["title"])}</a><br><small>{article["date_str"]}</small></p>\n')
    return "".join(html_parts)

@app.route('/api/topic/<path:topic_name>')
def get_topic_articles(topic_name):
    """
    Returns articles for a given topic grouped by day, limiting 
    the returned dataset to the 10 most recent days of updates.
    Returns chronologically ordered days (newest first).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT a.* FROM articles a
            JOIN digests d ON a.digest_id = d.id
            WHERE a.topic = %s
            ORDER BY d.created_at DESC, a.display_order ASC
        """, (topic_name,))
        articles = cur.fetchall()
    except Exception as e:
        print(f"Fallback query executed due to error: {e}")
        cur.execute("""
            SELECT * FROM articles 
            WHERE topic = %s 
            ORDER BY digest_id DESC, display_order ASC
        """, (topic_name,))
        articles = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    # Group records by local timezone day (YYYY-MM-DD)
    days_map = OrderedDict()
    for article in articles:
        try:
            pub_dt_user_tz = to_user_timezone(article[5])
            day_key = pub_dt_user_tz.strftime("%Y-%m-%d") if pub_dt_user_tz else "Date unavailable"
            
            # Big day header format: "Thursday, July 9 2026"
            day_display = pub_dt_user_tz.strftime(f"%A, %B {pub_dt_user_tz.day} %Y") if pub_dt_user_tz else "Date unavailable"
            
            # Small article timestamp format: "Thu, 09 Jul 2026 06:09 PM EDT"
            time_str = pub_dt_user_tz.strftime("%a, %d %b %Y %I:%M %p %Z") if pub_dt_user_tz else ""
        except Exception:
            day_key = "Date unavailable"
            day_display = "Date unavailable"
            time_str = ""

        if day_key not in days_map:
            days_map[day_key] = {
                "day_display": day_display,
                "articles": []
            }

        days_map[day_key]["articles"].append({
            "title": article[3],
            "link": article[4],
            "time_str": time_str
        })

    # Retrieve only the 10 most recent days of updates (ordered newest first)
    days_list = list(days_map.values())[:10]

    return jsonify(days_list)

@app.route('/robots.txt')
def serve_robots():
    """Serves the robots.txt file to web crawlers."""
    return send_from_directory(app.static_folder, 'robots.txt')

@app.route('/llms.txt')
def serve_llms():
    """Serves the llms.txt contextual profile to LLM agents."""
    return send_from_directory(app.static_folder, 'llms.txt', mimetype='text/plain')

@app.route('/sitemap.xml')
def serve_sitemap():
    """Serves a dynamic sitemap.xml to indexing crawlers."""
    sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://news.blakerayvid.com/</loc>
    <changefreq>hourly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://news.blakerayvid.com/forecast</loc>
    <changefreq>daily</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>"""
    return sitemap_xml, 200, {'Content-Type': 'application/xml'}

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))