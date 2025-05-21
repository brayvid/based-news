# Author: Blake Rayvid <https://github.com/brayvid/newsbot>

import os
import sys

# Set number of threads for various libraries to 1 if parallelism is not permitted on your system
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Define paths and URLs for local files and remote configuration.
BASE_DIR = os.path.dirname(__file__)
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"
TOPICS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=0&single=true&output=csv"
KEYWORDS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=314441026&single=true&output=csv"
OVERRIDES_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=1760236101&single=true&output=csv"

# Prevent concurrent runs using a lockfile
LOCKFILE = os.path.join(BASE_DIR, "based_news.lock")
if os.path.exists(LOCKFILE):
    print("Script is already running. Exiting.")
    sys.exit()
else:
    with open(LOCKFILE, 'w') as f:
        f.write("locked")

# Import all required libraries
import csv
import html
import logging
# import shutil
import json
import re
import ast
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import requests
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
from nltk.stem import PorterStemmer, WordNetLemmatizer
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess

# Initialize logging immediately to capture all runtime info
log_path = os.path.join(BASE_DIR, "logs/based_news.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO)
logging.info(f"Script started at {datetime.now()}")

# Initialize NLP tools and load environment variables from .env file.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
load_dotenv()

# Download nltk resources
from nltk.data import find
import nltk
nltk.data.path.append("~/nltk_data")

def ensure_nltk_data():
    for resource in ['wordnet', 'omw-1.4']:
        try:
            find(f'corpora/{resource}')
        except LookupError:
            try:
                nltk.download(resource)
            except Exception as e:
                print(f"Failed to download {resource}: {e}")

ensure_nltk_data()

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

# Now it's safe to use .get()
MAX_ARTICLE_HOURS = int(CONFIG.get("MAX_ARTICLE_HOURS", 6))
MAX_TOPICS = int(CONFIG.get("MAX_TOPICS", 7))
MAX_ARTICLES_PER_TOPIC = int(CONFIG.get("MAX_ARTICLES_PER_TOPIC", 1))
DEMOTE_FACTOR = float(CONFIG.get("DEMOTE_FACTOR", 0.5))
MATCH_THRESHOLD = 0.4   # For deduplication

# Load specified timezone to report in
USER_TIMEZONE = CONFIG.get("TIMEZONE", "America/New_York")
try:
    ZONE = ZoneInfo(USER_TIMEZONE)
except Exception:
    logging.warning(f"Invalid TIMEZONE '{USER_TIMEZONE}' in config. Falling back to 'America/New_York'")
    ZONE = ZoneInfo("America/New_York")

# Load user-defined topic and keyword importance scores from Google Sheets.
def load_csv_weights(url):
    weights = {}
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        lines = response.text.splitlines()
        reader = csv.reader(lines)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    weights[row[0].strip()] = int(row[1])
                except ValueError:
                    continue
        return weights
    except Exception as e:
        logging.error(f"Failed to load weights from {url}: {e}")
        return None
    
# Load user overrides (banned or demoted terms) from Google Sheets.
def load_overrides(url):
    overrides = {}
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        reader = csv.reader(response.text.splitlines())
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                overrides[row[0].strip().lower()] = row[1].strip().lower()
        return overrides
    except Exception as e:
        logging.error(f"Failed to load overrides: {e}")
        return None

TOPIC_WEIGHTS = load_csv_weights(TOPICS_CSV_URL)
KEYWORD_WEIGHTS = load_csv_weights(KEYWORDS_CSV_URL)
OVERRIDES = load_overrides(OVERRIDES_CSV_URL)

if None in (TOPIC_WEIGHTS, KEYWORD_WEIGHTS, OVERRIDES):
    logging.critical("Fatal: Failed to load topics, keywords, or overrides. Exiting.")
    sys.exit(1)

# Lowercases, stems, and lemmatizes words to produce normalized text for matching.
def normalize(text):
    words = text.lower().split()
    stemmed = [stemmer.stem(w) for w in words]
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]
    return " ".join(lemmatized)

# Checks if a normalized article title is already in history.json
def is_in_history(article_title, history):
    norm_title_tokens = set(normalize(article_title).split())

    for articles in history.values():
        for a in articles:
            past_tokens = set(normalize(a["title"]).split())
            if not past_tokens:
                continue
            overlap = norm_title_tokens.intersection(past_tokens)
            similarity = len(overlap) / len(norm_title_tokens)
            if similarity >= MATCH_THRESHOLD:
                return True

    return False

# Converts datetime to user timezone
def to_user_timezone(dt):
    return dt.astimezone(ZONE)

# Fetch recent RSS news articles for a given topic from Google News.
def fetch_articles_for_topic(topic, max_articles=10):
    """
    Fetch recent RSS news articles for a given topic from Google News RSS.
    Limits results to articles published in the last MAX_ARTICLE_HOURS hours
    and returns up to `max_articles` fresh articles.
    """
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(topic)}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        time_cutoff = datetime.now(ZONE) - timedelta(hours=MAX_ARTICLE_HOURS)
        articles = []

        for item in root.findall("./channel/item"):
            title = item.findtext("title") or "No title"
            link = item.findtext("link")
            pubDate = item.findtext("pubDate")

            try:
                pub_dt = parsedate_to_datetime(pubDate).astimezone(ZONE)
            except Exception:
                continue  # skip if publication date is malformed

            if pub_dt <= time_cutoff:
                continue  # too old

            articles.append({
                "title": title,
                "link": link,
                "pubDate": pubDate
            })

            if len(articles) >= max_articles:
                break  # respect per-topic cap

        return articles

    except Exception as e:
        logging.warning(f"Failed to fetch articles for {topic}: {e}")
        return []

# Construct a string summarizing user preferences from weights and overrides.
def build_user_preferences(topics, keywords, overrides):
    """
    Build a structured string representing the user's topic/keyword preferences
    and overrides, preserving importance scores.
    """
    preferences = []

    if topics:
        preferences.append("User topics (ranked 1-5 in importance):")
        for topic, score in sorted(topics.items(), key=lambda x: -x[1]):
            preferences.append(f"- {topic}: {score}")

    if keywords:
        preferences.append("\nHeadline keywords (ranked 1-5 in importance):")
        for keyword, score in sorted(keywords.items(), key=lambda x: -x[1]):
            preferences.append(f"- {keyword}: {score}")

    banned = [k for k, v in overrides.items() if v == "ban"]
    demoted = [k for k, v in overrides.items() if v == "demote"]

    if banned:
        preferences.append("\nBanned terms (must not appear in topics or headlines):")
        for term in banned:
            preferences.append(f"- {term}")

    if demoted:
        preferences.append(f"\nDemoted terms (consider headlines with these terms {DEMOTE_FACTOR} times as important to the user, all else equal):")
        for term in demoted:
            preferences.append(f"- {term}")

    return "\n".join(preferences)

# Clean and safely parse a possibly malformed JSON string.
def safe_parse_json(raw: str) -> dict:
    """
    Robustly parses malformed JSON returned by Gemini.
    Handles:
    - Markdown code fences
    - Smart/single quotes
    - Trailing commas
    - Unbalanced braces/brackets
    - Python-style dicts
    - Fallback to regex extraction
    """

    # First try untouched JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON load failed: {e}")  # Only then begin cleaning

    def strip_wrappers(text):
        text = text.strip()
        # Remove Markdown-style code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text

    def balance_braces(text):
        # Naively add missing closing brackets/braces
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            text = text.rstrip('}')[:-(close_braces - open_braces)]

        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            text = text.rstrip(']')[:-(close_brackets - open_brackets)]

        return text

    def fix_common_json_errors(text):
        # Replace smart quotes
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # Fix trailing commas before closing brackets/braces
        text = re.sub(r",\s*([\]}])", r"\1", text)

        # only replace single-quoted dict keys/values
        text = re.sub(r"(?<=[:\s])'([^']*?)'", r'"\1"', text)

        # Replace closing } for arrays with ]
        text = re.sub(r'(\[[^\[\]]*?)\s*\}', r'\1]', text)

        # Strip control characters, BOMs, directional marks
        text = re.sub(r"[\x00-\x1f\x7f\u202a-\u202e\u2060\uFEFF]", "", text)

        # Balance brackets/braces
        text = balance_braces(text)

        return text

    logging.debug("Pre-cleaned raw input (first 1000 chars):\n%s", raw[:1000])

    raw = strip_wrappers(raw)
    cleaned = fix_common_json_errors(raw)

    # 1. Try strict JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e_json:
        logging.warning(f"json.loads() failed: {e_json}")

    # 2. Try Python-style dict parsing
    try:
        return ast.literal_eval(cleaned)
    except Exception as e_ast:
        logging.warning(f"ast.literal_eval() failed: {e_ast}")

    # 3. Regex fallback to salvage dict
    try:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            candidate = match.group(0)
            candidate = fix_common_json_errors(candidate)
            return json.loads(candidate)
    except Exception as e_fallback:
        logging.warning(f"Regex-based fallback failed: {e_fallback}")

    logging.error("All parsing methods failed. Raw content:\n" + raw[:2000])
    return {}

# Returns True if any banned term is found in the normalized text
def contains_banned_keyword(text, banned_terms):
    norm_text = normalize(text)
    return any(banned in norm_text for banned in banned_terms)

# Call Gemini to select top topics/headlines among those retrieved based on user preferences and constraints.
def prioritize_with_gemini(headlines_to_send: dict, user_preferences: str, gemini_api_key: str) -> dict:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    prompt = (
        "You are choosing the most relevant news topics and headlines to include in an email digest for a user based on their specific preferences.\n"
        f"Given a dictionary of headlines retrieved for specific topics, and the user's preferences, select up to {MAX_TOPICS} of the most important topics to include in the digest.\n"
        f"For each selected topic, return the top {MAX_ARTICLES_PER_TOPIC} most important headlines to include in the digest.\n"
        "Respond only with valid JSON. Ensure you respond **WITH VALID JSON ONLY** like:\n"
        "{ \"Technology\": [\"Headline A\", \"Headline B\"], \"Climate\": [\"Headline C\"] }\n\n"
        "Ensure you **do not return multiple copies of the same or similar headlines** that are covering roughly the same thing, even if they are in different topics.\n"
        "Avoid all local news, for example any headlines containing a regional town or county name. Focus on U.S. News.\n"
        "Respect the user's importance preferences for topics and keywords, with 1 the lowest and 5 the highest.\n"
        f"Reject any headlines containing terms flagged 'banned', and demote headlines with terms flagged 'demote' by a multiplier of {DEMOTE_FACTOR}.\n"
        "Reject all advertisements and mentions of specific products or services unless it is newsworthy criticism."
        "There should be a healthy diversity of subjects covered overall in your article recommendations. Do not focus on one theme.\n"
        "Prefer to recommend content-rich and informative headlines over any clickbait or filler or questions or lists.\n"
        f"Again, there can only be up to {MAX_TOPICS} topics in your response, and up to {MAX_ARTICLES_PER_TOPIC} articles per topic.\n"
        f"User Preferences:\n{user_preferences}\n\n"
        f"Topics and Headlines:\n{json.dumps(dict(sorted(headlines_to_send.items())), indent=2)}\n"
    )

    # print(prompt)
    response = model.generate_content([prompt])
    raw = getattr(response, "text", None)

    # Handle Markdown-wrapped JSON like ```json\n{...}\n```
    if raw and raw.strip().startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())  # remove opening ```
        raw = re.sub(r"\s*```$", "", raw.strip())          # remove closing ```

    try:
        return safe_parse_json(raw)
    except Exception as e:
        logging.warning(f"Gemini returned invalid JSON. Skipping digest.")
        return {}


def main():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            logging.warning("history.json is empty or invalid. Starting with an empty history.")
            history = {}
    else:
        history = {}

    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.error("Missing GEMINI_API_KEY. Exiting.")
            return

        user_preferences = build_user_preferences(TOPIC_WEIGHTS, KEYWORD_WEIGHTS, OVERRIDES)

        headlines_to_send = {}
        full_articles = {}
        for topic in TOPIC_WEIGHTS:
            articles = fetch_articles_for_topic(topic, 10)
            if articles:
                banned_terms = [k for k, v in OVERRIDES.items() if v == "ban"]
                allowed_articles = [
                    a for a in articles
                    if not is_in_history(a["title"], history) and not contains_banned_keyword(a["title"], banned_terms)
                ]
                if allowed_articles:
                    headlines_to_send[topic] = [a["title"] for a in allowed_articles]
                    full_articles[topic] = allowed_articles

        if not headlines_to_send:
            logging.info("No headlines available for LLM input.")
            return

        total_headlines = sum(len(v) for v in headlines_to_send.values())
        logging.info(f"Sending {total_headlines} headlines across {len(headlines_to_send)} topics to Gemini.")

        digest_titles = prioritize_with_gemini(headlines_to_send, user_preferences, gemini_api_key)

        # Deduplicate
        seen_normalized_titles = set()
        digest = {}
        for topic, titles in digest_titles.items():
            selected = []
            for title in titles:
                norm = normalize(title)
                if norm in seen_normalized_titles:
                    continue
                seen_normalized_titles.add(norm)
                for a in full_articles.get(topic, []):
                    if normalize(a["title"]) == norm:
                        selected.append(a)
                        break
            if selected:
                digest[topic] = selected

        if not digest:
            logging.info("Gemini returned no digest-worthy content. Updating timestamp only.")

            digest_path = os.path.join(BASE_DIR, "public", "digest.html")
            os.makedirs(os.path.dirname(digest_path), exist_ok=True)

            last_updated = datetime.now(ZONE).strftime("%A, %d %B %Y %I:%M %p %Z")
            footer_html = (
                f"<div id='last-updated' style='display: none;'>"
                f"Last updated: {last_updated}"
                f"</div>\n"
            )

            try:
                with open(digest_path, "r+", encoding="utf-8") as f:
                    content = f.read()
                    # Replace old timestamp or append if missing
                    new_content = re.sub(
                        r"<div id='last-updated'.*?</div>",
                        footer_html.strip(),
                        content,
                        flags=re.DOTALL
                    )
                    if new_content == content:
                        new_content += "\n" + footer_html
                    f.seek(0)
                    f.write(new_content)
                    f.truncate()
            except FileNotFoundError:
                with open(digest_path, "w", encoding="utf-8") as f:
                    f.write(footer_html)

            return


        # Build HTML digest file
        digest_path = os.path.join(BASE_DIR, "public", "digest.html")
        os.makedirs(os.path.dirname(digest_path), exist_ok=True)

        with open(digest_path, "w", encoding="utf-8") as f:
            for topic, articles in digest.items():
                f.write(f"<h3>{html.escape(topic)}</h3>\n")
                for article in articles:
                    pub_dt = to_user_timezone(parsedate_to_datetime(article["pubDate"]))
                    f.write(
                        f'<p>'
                        f'<a href="{article["link"]}" target="_blank">{html.escape(article["title"])}</a><br>'
                        f'<small>📅 {pub_dt.strftime("%a, %d %b %Y %I:%M %p %Z")}</small>'
                        f'</p>\n'
                    )
        # Add last updated footer
        last_updated = datetime.now(ZONE).strftime("%A, %d %B %Y %I:%M %p %Z")
        footer_html = (
            f"<div class='timestamp' id='last-updated' style='display: none;'>"
            f"Last updated: {last_updated}"
            f"</div>\n"
        )

        with open(digest_path, "a", encoding="utf-8") as f:
            f.write(footer_html)

        logging.info("Digest HTML written to public/digest.html")

        # Update history
        for topic, articles in digest.items():
            key = topic.replace(" ", "_").lower()
            if key not in history:
                history[key] = []
            existing_titles = {normalize(a["title"]) for a in history[key]}
            for article in articles:
                if normalize(article["title"]) not in existing_titles:
                    history[key].append({
                        "title": article["title"],
                        "pubDate": article["pubDate"]
                    })
            history[key] = history[key][-40:]

        # Clean old history
        two_weeks_ago = datetime.now(ZONE) - timedelta(days=15)
        for topic in list(history.keys()):
            history[topic] = [
                a for a in history[topic]
                if parsedate_to_datetime(a["pubDate"]).astimezone(ZONE) >= two_weeks_ago
            ]

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        # Git commit and push
        try:
            GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
            GITHUB_USER = os.getenv("GITHUB_USER", "your-username")
            REPO = "based-news"

            remote_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO}.git"

            subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, cwd=BASE_DIR)
            subprocess.run(["git", "pull", "origin", "main"], check=True, cwd=BASE_DIR)
            subprocess.run(["git", "add", "public/digest.html", "history.json"], check=True, cwd=BASE_DIR)
            subprocess.run(["git", "commit", "-m", "Auto-update digest and history"], check=True, cwd=BASE_DIR)
            subprocess.run(["git", "push"], check=True, cwd=BASE_DIR)
            logging.info("Digest and history committed and pushed to GitHub.")
        except Exception as e:
            logging.error(f"Git commit/push failed: {e}")

    finally:
        # nltk_path = os.path.expanduser("~/nltk_data")
        # if os.path.exists(nltk_path):
        #     try:
        #         shutil.rmtree(nltk_path)
        #     except Exception as e:
        #         logging.warning(f"Failed to delete ~/nltk_data: {e}")

        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
        logging.info(f"Lockfile released at {datetime.now()}")


if __name__ == "__main__":
    main()
