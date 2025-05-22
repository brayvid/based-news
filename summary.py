# Author: Blake Rayvid <https://github.com/brayvid/newsbot>

import os
import sys
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess
import requests
import csv

CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"

# --- Setup ---
BASE_DIR = os.path.dirname(__file__)
LOCKFILE = os.path.join(BASE_DIR, "summary.lock")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "public", "summary.html")
LOGFILE = os.path.join(BASE_DIR, "logs/summary.log")
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

# # --- Locking to prevent concurrent runs ---
# if os.path.exists(LOCKFILE):
#     print("summary.py is already running. Exiting.")
#     sys.exit()
# else:
#     with open(LOCKFILE, "w") as f:
#         f.write("locked")

# --- Logging ---
logging.basicConfig(filename=LOGFILE, level=logging.INFO)
logging.info(f"Summary started at {datetime.now()}")

# --- Load environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Loads key-value config settings from a CSV Google Sheet URL.
def load_config_from_sheet(url):
    config = {}
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        lines = response.text.splitlines()
        reader = csv.reader(lines)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                val = row[1].strip()
                try:
                    if '.' in val:
                        config[key] = float(val)
                    else:
                        config[key] = int(val)
                except ValueError:
                    config[key] = val  # fallback to string
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {url}: {e}")
        return None

# Load config before main
CONFIG = load_config_from_sheet(CONFIG_CSV_URL)
if CONFIG is None:
    logging.critical("Fatal: Unable to load CONFIG from sheet. Exiting.")
    sys.exit(1)  # Stop immediately if config fails

# Load specified timezone to report in
USER_TIMEZONE = CONFIG.get("TIMEZONE", "America/New_York")
try:
    ZONE = ZoneInfo(USER_TIMEZONE)
except Exception:
    logging.warning(f"Invalid TIMEZONE '{USER_TIMEZONE}' in config. Falling back to 'America/New_York'")
    ZONE = ZoneInfo("America/New_York")

# --- Load history ---
try:
    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)
except Exception as e:
    logging.critical(f"Failed to load history.json: {e}")
    os.remove(LOCKFILE)
    sys.exit(1)

# --- Format history into plain text ---
def format_history(data):
    parts = []
    for topic, articles in data.items():
        parts.append(f"### {topic.title()}")
        for a in articles:
            parts.append(f"- {a['title']} ({a['pubDate']})")
    return "\n".join(parts)

# --- Gemini query ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")

question = (
    "Give a brief report with short paragraphs in roughly 100 words on how the world has been doing lately based on the attached headlines. Use simple language, cite figures, and be specific with people, places, things, etc. Do not use bullet points. State the timeframe being discussed. Don't state that it's a report, simply present the findings. Then at the end, in 50 words, using all available clues in the headlines, predict what should in all likelihood occur in the near future, and less likely but still entirely possible events, and give a sense of the ramifications."
)

try:
    logging.info("Sending prompt to Gemini...")
    prompt = f"{question}\n\n{format_history(history_data)}"
    result = model.generate_content(prompt)
    answer = result.text.strip()
    logging.info("Gemini returned a response.")

except Exception as e:
    logging.error(f"Gemini request failed: {e}")
    os.remove(LOCKFILE)
    sys.exit(1)

# --- Format HTML output ---
formatted = answer.replace('\n', '<br>')
timestamp = datetime.now(ZONE).strftime("%A, %d %B %Y %I:%M %p %Z")
html_output = f"""<html>
  <body>
    <p>{formatted}</p>
    <div class='timestamp' id='summary-last-updated' style='display: none;'>Last updated: {timestamp}</div>
  </body>
</html>"""

# --- Write to file ---
try:
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(html_output)
    logging.info(f"Summary written to {SUMMARY_FILE}")
except Exception as e:
    logging.error(f"Failed to write summary.html: {e}")

# Git commit and push
try:
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_USER = os.getenv("GITHUB_USER", "your-username")
    REPO = "based-news"

    remote_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO}.git"

    subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, cwd=BASE_DIR)
    subprocess.run(["git", "pull", "origin", "main"], check=True, cwd=BASE_DIR)
    subprocess.run(["git", "add", "public/summary.html"], check=True, cwd=BASE_DIR)
    subprocess.run(["git", "commit", "-m", "Auto-update digest and history"], check=True, cwd=BASE_DIR)
    subprocess.run(["git", "push"], check=True, cwd=BASE_DIR)
    logging.info("Digest and history committed and pushed to GitHub.")
except Exception as e:
    logging.error(f"Git commit/push failed: {e}")

# # --- Cleanup ---
# if os.path.exists(LOCKFILE):
#     os.remove(LOCKFILE)
#     logging.info("Lockfile released.")
