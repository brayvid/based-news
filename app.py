# based-news/app.py

import os
import html
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
import psycopg2
from flask import Flask, render_template, jsonify, abort

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load database URL from environment variables, provided by Railway
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
        # DB columns: id, digest_id, topic, title, link, pub_date, display_order
        topic = article[2]
        if topic not in digest_data:
            digest_data[topic] = []
        
        try:
            pub_dt_user_tz = to_user_timezone(article[5])
            date_str = pub_dt_user_tz.strftime("%a, %d %b %Y %I:%M %p %Z") if pub_dt_user_tz else "Date unavailable"
        except Exception:
            date_str = "Date unavailable"

        digest_data[topic].append({
            "title": article[3],
            "link": article[4],
            "date_str": date_str
        })
    return digest_data

@app.route('/')
def index():
    """Serves the main page with the latest digest."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Find the ID of the most recent digest
    cur.execute("SELECT id FROM digests ORDER BY created_at DESC LIMIT 1")
    latest_digest_row = cur.fetchone()
    
    latest_digest_data = {}
    if latest_digest_row:
        latest_digest_id = latest_digest_row[0]
        # Fetch all articles for that digest, preserving Gemini's order
        cur.execute(
            "SELECT * FROM articles WHERE digest_id = %s ORDER BY display_order ASC",
            (latest_digest_id,)
        )
        articles = cur.fetchall()
        latest_digest_data = format_digest_data(articles)

    cur.close()
    conn.close()
    
    # Render the main HTML template, passing the latest digest data to it
    return render_template('index.html', latest_digest=latest_digest_data)

@app.route('/manifest')
def get_manifest():
    """Provides a list of historical digests for the frontend slider."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Fetch all digest IDs and timestamps, newest first, up to the configured limit
    max_history = int(os.environ.get("MAX_HISTORY_DIGESTS", 12))
    cur.execute("SELECT id, created_at FROM digests ORDER BY created_at ASC LIMIT %s", (max_history,))
    digests = cur.fetchall()
    cur.close()
    conn.close()
    
    manifest = [{"id": row[0], "timestamp": row[1].isoformat()} for row in digests]
    return jsonify(manifest)

@app.route('/digest/<int:digest_id>')
def get_digest_html(digest_id):
    """Fetches a specific historical digest and returns it as HTML content."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM articles WHERE digest_id = %s ORDER BY display_order ASC",
        (digest_id,)
    )
    articles = cur.fetchall()
    cur.close()
    conn.close()

    if not articles:
        abort(404, description="Digest not found")
        
    digest_data = format_digest_data(articles)
    
    # Generate an HTML string from the data
    html_parts = []
    for topic, articles_list in digest_data.items():
        html_parts.append(f"<h3>{html.escape(topic)}</h3>\n")
        for article in articles_list:
            html_parts.append(
                f'<p>'
                f'<a href="{html.escape(article["link"])}" target="_blank">{html.escape(article["title"])}</a><br>'
                f'<small>{article["date_str"]}</small>'
                f'</p>\n'
            )
    return "".join(html_parts)

if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))