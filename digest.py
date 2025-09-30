# Author: Blake Rayvid <https://github.com/brayvid/based-news>

import os
import sys

# Set number of threads for various libraries to 1 if parallelism is not permitted on your system
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Define paths and URLs for local files and remote configuration.
# Robust BASE_DIR definition
try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:  # __file__ is not defined, e.g., in interactive shell
    BASE_DIR = os.getcwd()

HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
DIGEST_STATE_FILE = os.path.join(BASE_DIR, "content.json")
# New paths for digest history
DIGESTS_DIR = os.path.join(BASE_DIR, "public", "digests")
DIGEST_MANIFEST_FILE = os.path.join(BASE_DIR, "public", "digest-manifest.json")
# Path for the file containing only the latest digest's HTML content
LATEST_DIGEST_HTML_FILE = os.path.join(BASE_DIR, "public", "digest.html")


CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"
TOPICS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=0&single=true&output=csv"
KEYWORDS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=314441026&single=true&output=csv"
OVERRIDES_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=1760236101&single=true&output=csv"

# Import all required libraries
import csv
import html
import logging
import json
import re
import ast
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from collections import defaultdict
import requests
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
from nltk.stem import PorterStemmer, WordNetLemmatizer
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import subprocess
from proto.marshal.collections.repeated import RepeatedComposite
from proto.marshal.collections.maps import MapComposite

# Initialize logging immediately to capture all runtime info
log_path = os.path.join(BASE_DIR, "logs/digest.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Script started at {datetime.now()}")

# Initialize NLP tools and load environment variables from .env file.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
load_dotenv()

# Download nltk resources
from nltk.data import find
import nltk

if os.getenv('CI'):
    nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
else:
    nltk.data.path.append(os.path.expanduser("~/nltk_data"))


def ensure_nltk_data():
    for resource in ['wordnet', 'omw-1.4']:
        try:
            find(f'corpora/{resource}')
            logging.info(f"NLTK resource '{resource}' found.")
        except LookupError:
            logging.info(f"NLTK resource '{resource}' not found. Attempting download to {nltk.data.path[-1]}...")
            try:
                nltk.download(resource, download_dir=nltk.data.path[-1])
                logging.info(f"Successfully downloaded NLTK resource '{resource}'.")
            except Exception as e:
                logging.error(f"Failed to download NLTK resource '{resource}': {e}")
                print(f"Failed to download {resource}: {e}")

ensure_nltk_data()

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
                    # Improved check for float conversion
                    if '.' in val and val.replace('.', '', 1).isdigit():
                        config[key] = float(val)
                    else:
                        config[key] = int(val)
                except ValueError:
                    if val.lower() == 'true':
                        config[key] = True
                    elif val.lower() == 'false':
                        config[key] = False
                    else:
                        config[key] = val
        logging.info(f"Config loaded successfully from {url}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {url}: {e}")
        return None

CONFIG = load_config_from_sheet(CONFIG_CSV_URL)
if CONFIG is None:
    logging.critical("Fatal: Unable to load CONFIG from sheet. Exiting.")
    sys.exit(1)

MAX_ARTICLE_HOURS = int(CONFIG.get("MAX_ARTICLE_HOURS", 6))
MAX_TOPICS = int(CONFIG.get("MAX_TOPICS", 10))
MAX_ARTICLES_PER_TOPIC = int(CONFIG.get("MAX_ARTICLES_PER_TOPIC", 1))
MAX_HISTORY_DIGESTS = int(CONFIG.get("MAX_HISTORY_DIGESTS", 12))
DEMOTE_FACTOR = float(CONFIG.get("DEMOTE_FACTOR", 0.5))
MATCH_THRESHOLD = float(CONFIG.get("DEDUPLICATION_MATCH_THRESHOLD", 0.4))
GEMINI_MODEL_NAME = CONFIG.get("GEMINI_MODEL_NAME", "gemini-2.5-lite") # Corrected model name
STALE_TOPIC_THRESHOLD_HOURS = int(CONFIG.get("STALE_TOPIC_THRESHOLD_HOURS", 72)) # Not used in this "snapshot" model

USER_TIMEZONE = CONFIG.get("TIMEZONE", "America/New_York")
try:
    ZONE = ZoneInfo(USER_TIMEZONE)
except Exception:
    logging.warning(f"Invalid TIMEZONE '{USER_TIMEZONE}' in config. Falling back to 'America/New_York'")
    ZONE = ZoneInfo("America/New_York")

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
                    logging.warning(f"Skipping invalid weight in {url}: {row}")
                    continue
        logging.info(f"Weights loaded successfully from {url}")
        return weights
    except Exception as e:
        logging.error(f"Failed to load weights from {url}: {e}")
        return None

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
        logging.info(f"Overrides loaded successfully from {url}")
        return overrides
    except Exception as e:
        logging.error(f"Failed to load overrides from {url}: {e}")
        return None

TOPIC_WEIGHTS = load_csv_weights(TOPICS_CSV_URL)
KEYWORD_WEIGHTS = load_csv_weights(KEYWORDS_CSV_URL)
OVERRIDES = load_overrides(OVERRIDES_CSV_URL)

if None in (TOPIC_WEIGHTS, KEYWORD_WEIGHTS, OVERRIDES):
    logging.critical("Fatal: Failed to load topics, keywords, or overrides. Exiting.")
    sys.exit(1)

def normalize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    # Combining stemming and lemmatization is often redundant; lemmatization is generally preferred.
    # Sticking to lemmatization for cleaner base words.
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

def is_high_confidence_duplicate(norm_title_tokens: set, history_token_sets: list, threshold: float) -> bool:
    """
    Checks if a set of tokens for a new title is a high-confidence duplicate
    of any token set in a pre-normalized list of history token sets using Jaccard similarity.
    """
    if not norm_title_tokens:
        return False

    for past_tokens in history_token_sets:
        if not past_tokens:
            continue
        
        intersection_len = len(norm_title_tokens.intersection(past_tokens))
        union_len = len(norm_title_tokens.union(past_tokens))
        if union_len == 0: continue

        similarity = intersection_len / union_len
        if similarity >= threshold:
            return True
    return False

def to_user_timezone(dt):
    return dt.astimezone(ZONE)

def fetch_articles_for_topic(topic, max_articles=10):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(topic)}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        time_cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(hours=MAX_ARTICLE_HOURS)
        articles = []
        for item in root.findall("./channel/item"):
            title_element = item.find("title")
            title = title_element.text if title_element is not None and title_element.text else "No title"
            link_element = item.find("link")
            link = link_element.text if link_element is not None and link_element.text else None
            pubDate_element = item.find("pubDate")
            pubDate = pubDate_element.text if pubDate_element is not None and pubDate_element.text else None
            if not link or not pubDate:
                logging.warning(f"Skipping article with missing link or pubDate for topic '{topic}': Title '{title}'")
                continue
            try:
                pub_dt_utc = parsedate_to_datetime(pubDate).astimezone(ZoneInfo("UTC"))
            except Exception as e:
                logging.warning(f"Could not parse pubDate '{pubDate}' for article '{title}': {e}")
                continue
            if pub_dt_utc <= time_cutoff:
                continue
            articles.append({"title": title.strip(), "link": link, "pubDate": pubDate})
            if len(articles) >= max_articles:
                break
        return articles
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request failed for topic {topic} articles: {e}")
        return []
    except ET.ParseError as e:
        logging.warning(f"Failed to parse XML for topic {topic}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching articles for {topic}: {e}")
        return []

def load_recent_digest_history(digests_dir, manifest_file, max_digests):
    """
    Loads headlines from the most recent historical digests by reading the HTML files.
    This provides the LLM with a true representation of what the user has already seen.
    """
    history_headlines = []
    if not os.path.exists(manifest_file):
        logging.info("Digest manifest not found. Starting with no digest history.")
        return []

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Could not read or parse manifest file for history, continuing without. Error: {e}")
        return []

    digests_to_load = manifest[:max_digests]
    logging.info(f"Loading history from the {len(digests_to_load)} most recent digests.")

    for entry in digests_to_load:
        digest_file_path = os.path.join(os.path.dirname(manifest_file), entry["file"])
        if os.path.exists(digest_file_path):
            try:
                with open(digest_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    found_headlines = re.findall(r'<a href="[^"]+" target="_blank">([^<]+)</a>', content)
                    history_headlines.extend(html.unescape(h.strip()) for h in found_headlines)
            except IOError as e:
                logging.warning(f"Could not read historical digest file {digest_file_path}: {e}")

    unique_history = list(dict.fromkeys(history_headlines))
    logging.info(f"Loaded {len(unique_history)} unique headlines from recent digest history.")
    return unique_history

def safe_parse_json(raw_json_string: str) -> dict:
    if not raw_json_string:
        logging.warning("safe_parse_json received empty string.")
        return {}
    text = raw_json_string.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        logging.warning("JSON string is empty after stripping wrappers.")
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If json.loads fails, try to fix common errors and re-parse.
        # This is a common failure mode for LLMs.
        text = re.sub(r",\s*([\]}])", r"\1", text) # Remove trailing commas
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed after cleaning: {e}. Raw content (first 500 chars): {raw_json_string[:500]}")
            return {}

# --- Tooling for Gemini ---

digest_tool_schema = {
    "type": "object",
    "properties": {
        "selected_digest_entries": {
            "type": "array",
            "description": (
                f"A list of selected news topics. Each entry should be an object "
                f"with 'topic_name' (string) and 'headlines' (list of strings). "
                f"Select up to {MAX_TOPICS} topics, and for each topic, up to "
                f"{MAX_ARTICLES_PER_TOPIC} relevant headlines."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string", "description": "The name of the news topic."},
                    "headlines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"A list of up to {MAX_ARTICLES_PER_TOPIC} important headlines."
                    }
                },
                "required": ["topic_name", "headlines"]
            }
        }
    },
    "required": ["selected_digest_entries"]
}

SELECT_DIGEST_ARTICLES_TOOL = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="format_digest_selection",
            description=(
                f"Formats selected news topics and headlines. Select up to {MAX_TOPICS} topics "
                f"and up to {MAX_ARTICLES_PER_TOPIC} headlines per topic. The output is a list "
                "of objects, each containing a 'topic_name' and a 'headlines' list."
            ),
            parameters=digest_tool_schema,
        )
    ]
)

def contains_banned_keyword(text, banned_terms):
    if not text: return False
    # No need to normalize here if banned_terms are pre-normalized
    return any(banned_term in text.lower() for banned_term in banned_terms if banned_term)

def pre_filter_and_deduplicate_headlines(
    headlines_to_send: dict,
    digest_history_headlines: list,
    banned_terms: list
) -> dict:
    """
    Filters headlines based on history and banned terms, and performs aggressive
    deduplication before sending them to the AI. This is a deterministic pre-filter.
    """
    # Create a set of normalized headlines from the recent digest history for fast lookups.
    history_set = {normalize(h) for h in digest_history_headlines}
    
    seen_normalized_headlines = set()
    pre_filtered_headlines = defaultdict(list)
    
    # Iterate through all candidate articles
    for topic, headlines in headlines_to_send.items():
        for headline in headlines:
            # 1. Banned Term Filtering (done first as it's a hard rule)
            # Assumes banned_terms are already normalized (lower-cased)
            if contains_banned_keyword(headline, banned_terms):
                logging.debug(f"Pre-filter BAN: '{headline}'")
                continue

            normalized_headline = normalize(headline)
            if not normalized_headline:
                continue

            # 2. History Filtering
            if normalized_headline in history_set:
                logging.debug(f"Pre-filter HISTORY: '{headline}'")
                continue
                
            # 3. Aggressive Deduplication within this run
            if normalized_headline in seen_normalized_headlines:
                logging.debug(f"Pre-filter DEDUPE: '{headline}'")
                continue
            
            # If the headline passes all deterministic checks, add it to the list for the LLM
            seen_normalized_headlines.add(normalized_headline)
            pre_filtered_headlines[topic].append(headline)
            
    # Remove any topics that have become empty after filtering
    return {k: v for k, v in pre_filtered_headlines.items() if v}

def prioritize_with_gemini(
    headlines_to_send: dict,
    digest_history: list,
    gemini_api_key: str,
    topic_weights: dict,
    keyword_weights: dict,
    overrides: dict
) -> dict:
    # This function now assumes a smaller, pre-filtered input for best performance.
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME, # Uses the name from your config
        tools=[SELECT_DIGEST_ARTICLES_TOOL]
    )

    pref_data = {
        "topic_weights": topic_weights,
        "keyword_weights": keyword_weights,
        "demoted_terms": [k for k, v in overrides.items() if v == "demote"]
    }
    user_preferences_json = json.dumps(pref_data, indent=2)
    
    # Using the concise prompt that works well with pre-filtered data
    prompt = (
        "You are an expert news curator. A pre-filter has removed duplicates and banned topics. "
        "Your task is to perform the final nuanced selection and sorting on this high-relevance candidate list.\n\n"
        f"User Preferences:\n```json\n{user_preferences_json}\n```\n"
        f"Candidate Headlines:\n```json\n{json.dumps(dict(sorted(headlines_to_send.items())), indent=2)}\n```\n\n"
        "--- INSTRUCTIONS ---\n"
        "1.  **Filter for Quality:** Remove purely local news, investment advice, low-quality content, and strongly deprioritize headlines with 'demoted_terms'.\n"
        "2.  **Select & Sort:** Choose the best articles, respecting the limits. Topics MUST be sorted by importance, NOT alphabetically. Find the best overall headline; its topic is #1. Find the best headline from the remaining topics; its topic is #2, and so on.\n"
        "3.  **Output:** Call the `format_digest_selection` tool with your final, sorted list."
    )

    try:
        logging.info("Sending smaller, pre-filtered request to Gemini.")
        response = model.generate_content(
            [prompt],
            # This tool_config is crucial for reliability
            tool_config={"function_calling_config": {"mode": "ANY", "allowed_function_names": ["format_digest_selection"]}}
        )

        function_call_part = None
        if response.parts:
            for part in response.parts:
                if part.function_call:
                    function_call_part = part.function_call
                    break

        if function_call_part:
            args = function_call_part.args
            transformed_output = {}
            if "selected_digest_entries" in args:
                for entry in args["selected_digest_entries"]:
                    topic_name = entry.get("topic_name")
                    # This is the special Protobuf object
                    headlines_proto = entry.get("headlines", [])
                    
                    # --- THIS IS THE FIX ---
                    # Explicitly convert the special object to a standard Python list of strings.
                    headlines_list = [str(h) for h in headlines_proto]
                    # --- END OF FIX ---

                    if topic_name and headlines_list:
                        # Assign the clean Python list to the output
                        transformed_output[topic_name] = headlines_list

            logging.info(f"Successfully processed tool call from Gemini. Returning {len(transformed_output)} topics.")
            return transformed_output
        else:
            logging.warning("Gemini returned no usable function call. Check prompt feedback if available.")
            if response.prompt_feedback:
                logging.warning(f"Prompt Feedback: {response.prompt_feedback}")
            return {}

    except Exception as e:
        logging.error(f"Error during Gemini API call or processing response: {e}", exc_info=True)
        return {}
    
def select_top_candidate_topics(headlines_by_topic: dict, topic_weights: dict, max_topics_to_consider: int) -> dict:
    """
    Scores topics based on user weights and headline volume, returning a reduced set for the LLM.
    """
    topic_scores = {}
    for topic, headlines in headlines_by_topic.items():
        if headlines:
            score = topic_weights.get(topic, 1) * len(headlines)
            topic_scores[topic] = score

    sorted_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)
    top_topics = sorted_topics[:max_topics_to_consider]
    
    return {topic: headlines_by_topic[topic] for topic in top_topics}
     
def generate_digest_html_content(digest_data, current_zone):
    """Generates just the HTML string for a digest, without writing to a file."""
    html_parts = []
    for topic, articles in digest_data.items():
        html_parts.append(f"<h3>{html.escape(topic)}</h3>\n")
        for article in articles:
            try:
                pub_dt_orig = parsedate_to_datetime(article["pubDate"])
                pub_dt_user_tz = to_user_timezone(pub_dt_orig)
                date_str = pub_dt_user_tz.strftime("%a, %d %b %Y %I:%M %p %Z")
            except Exception as e:
                logging.warning(f"Could not parse date for article '{article['title']}': {article['pubDate']} - {e}")
                date_str = "Date unavailable"

            html_parts.append(
                f'<p>'
                f'<a href="{html.escape(article["link"])}" target="_blank">{html.escape(article["title"])}</a><br>'
                f'<small>{date_str}</small>'
                f'</p>\n'
            )
    return "".join(html_parts)

def update_digest_history(new_digest_html_content, current_zone):
    """Saves a new historical digest and updates the manifest."""
    # 1. Generate filename and path for the new historical digest
    now_utc = datetime.now(ZoneInfo("UTC"))
    ts_filename = now_utc.strftime("%Y%m%d%H%M%S")
    new_digest_filename = f"{ts_filename}.html"
    new_digest_path = os.path.join(DIGESTS_DIR, new_digest_filename)

    # 2. Write the new historical digest HTML file
    os.makedirs(os.path.dirname(new_digest_path), exist_ok=True)
    with open(new_digest_path, "w", encoding="utf-8") as f:
        f.write(new_digest_html_content)
    logging.info(f"Wrote historical digest to {new_digest_path}")

    # 3. Read the existing manifest
    try:
        if os.path.exists(DIGEST_MANIFEST_FILE):
            with open(DIGEST_MANIFEST_FILE, "r") as f:
                manifest = json.load(f)
        else:
            manifest = []
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Could not read or parse manifest file, starting fresh. Error: {e}")
        manifest = []

    # 4. Add new entry to manifest (newest first)
    new_entry = {
        "timestamp": now_utc.isoformat(),
        "file": f"digests/{new_digest_filename}"
    }
    manifest.insert(0, new_entry)

    # 5. Prune old digests and remove files
    if len(manifest) > MAX_HISTORY_DIGESTS:
        old_entries = manifest[MAX_HISTORY_DIGESTS:]
        manifest = manifest[:MAX_HISTORY_DIGESTS]
        for entry in old_entries:
            old_file_path = os.path.join(BASE_DIR, "public", entry["file"])
            if os.path.exists(old_file_path):
                try:
                    os.remove(old_file_path)
                    logging.info(f"Removed old digest file: {old_file_path}")
                except OSError as e:
                    logging.error(f"Error removing old digest file {old_file_path}: {e}")

    # 6. Write updated manifest
    with open(DIGEST_MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Updated digest manifest file: {DIGEST_MANIFEST_FILE}")


def update_history_file(newly_selected_articles_by_topic, current_history, history_file_path, current_zone):
    if not newly_selected_articles_by_topic or not isinstance(newly_selected_articles_by_topic, dict):
        logging.info("No newly selected articles provided to update_history_file, or invalid format. Proceeding to prune existing history.")

    for topic, articles in newly_selected_articles_by_topic.items():
        history_key = topic.lower().replace(" ", "_")
        if history_key not in current_history:
            current_history[history_key] = []

        existing_norm_titles_in_topic_history = {normalize(a.get("title","")) for a in current_history[history_key]}

        for article in articles:
            norm_title = normalize(article["title"])
            if norm_title not in existing_norm_titles_in_topic_history:
                current_history[history_key].append({
                    "title": article["title"],
                    "pubDate": article["pubDate"]
                })
                existing_norm_titles_in_topic_history.add(norm_title)

    history_retention_days = int(CONFIG.get("HISTORY_RETENTION_DAYS", 7))
    time_limit_utc = datetime.now(ZoneInfo("UTC")) - timedelta(days=history_retention_days)

    for topic_key in list(current_history.keys()):
        updated_topic_articles_in_history = []
        for article_entry in current_history[topic_key]:
            try:
                pub_dt_str = article_entry.get("pubDate")
                if not pub_dt_str:
                    logging.warning(f"History entry for topic '{topic_key}' title '{article_entry.get('title')}' missing pubDate. Keeping.")
                    updated_topic_articles_in_history.append(article_entry)
                    continue

                pub_dt_orig = parsedate_to_datetime(pub_dt_str)
                pub_dt_utc = pub_dt_orig.astimezone(ZoneInfo("UTC")) if pub_dt_orig.tzinfo else pub_dt_orig.replace(tzinfo=ZoneInfo("UTC"))

                if pub_dt_utc >= time_limit_utc:
                    updated_topic_articles_in_history.append(article_entry)
            except Exception as e:
                logging.warning(f"Could not parse pubDate '{article_entry.get('pubDate')}' for history cleaning of article '{article_entry.get('title')}': {e}. Keeping entry.")
                updated_topic_articles_in_history.append(article_entry)

        if updated_topic_articles_in_history:
            current_history[topic_key] = updated_topic_articles_in_history
        else:
            logging.info(f"Removing empty topic_key '{topic_key}' from history after pruning.")
            del current_history[topic_key]

    try:
        with open(history_file_path, "w", encoding="utf-8") as f:
            json.dump(current_history, f, indent=2)
        logging.info(f"History file updated at {history_file_path}")
    except IOError as e:
        logging.error(f"Failed to write updated history file: {e}")

def perform_git_operations(base_dir, current_zone, config_obj):
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        github_repository_owner_slash_repo = os.getenv("GITHUB_REPOSITORY")

        if not github_token or not github_repository_owner_slash_repo:
            logging.error("GITHUB_TOKEN or GITHUB_REPOSITORY not set. Cannot push to GitHub.")
            return

        remote_url = f"https://oauth2:{github_token}@github.com/{github_repository_owner_slash_repo}.git"

        commit_author_name = os.getenv("GITHUB_USER", config_obj.get("GIT_USER_NAME", "Automated Digest Bot"))
        commit_author_email = os.getenv("GITHUB_EMAIL", config_obj.get("GIT_USER_EMAIL", "bot@example.com"))

        logging.info(f"Using Git Commit Author Name: '{commit_author_name}', Email: '{commit_author_email}'")

        subprocess.run(["git", "config", "user.name", commit_author_name], check=True, cwd=base_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", commit_author_email], check=True, cwd=base_dir, capture_output=True)
        try:
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, cwd=base_dir, capture_output=True)
        except subprocess.CalledProcessError:
            logging.info("Failed to set-url (maybe remote 'origin' doesn't exist). Attempting to add.")
            subprocess.run(["git", "remote", "add", "origin", remote_url], check=True, cwd=base_dir, capture_output=True)

        branch_result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True, cwd=base_dir)
        current_branch = branch_result.stdout.strip()
        if not current_branch or current_branch == "HEAD":
            logging.warning(f"Could not reliably determine current branch (got '{current_branch}'). Defaulting to 'main'.")
            current_branch = "main"
            try:
                subprocess.run(["git", "checkout", current_branch], check=True, cwd=base_dir, capture_output=True)
            except subprocess.CalledProcessError as e_checkout:
                err_msg = e_checkout.stderr.decode(errors='ignore') if e_checkout.stderr else e_checkout.stdout.decode(errors='ignore')
                logging.error(f"Failed to checkout branch '{current_branch}': {err_msg}. Proceeding with caution.")

        logging.info("Attempting to stash local changes before pull.")
        stash_result = subprocess.run(["git", "stash", "push", "-u", "-m", "WIP_Stash_By_Script"], capture_output=True, text=True, cwd=base_dir)
        stashed_changes = "No local changes to save" not in stash_result.stdout and stash_result.returncode == 0

        if stashed_changes:
            logging.info(f"Stashed local changes. Output: {stash_result.stdout.strip()}")
        elif stash_result.returncode != 0 :
             logging.warning(f"git stash push failed. Stdout: {stash_result.stdout.strip()}, Stderr: {stash_result.stderr.strip()}")
        else:
            logging.info("No local changes to stash.")

        logging.info(f"Attempting to pull with rebase from origin/{current_branch}...")
        pull_rebase_cmd = ["git", "pull", "--rebase", "origin", current_branch]
        pull_result = subprocess.run(pull_rebase_cmd, capture_output=True, text=True, cwd=base_dir)

        if pull_result.returncode != 0:
            logging.warning(f"'git pull --rebase' failed. Stdout: {pull_result.stdout.strip()}. Stderr: {pull_result.stderr.strip()}")
            if "CONFLICT" in pull_result.stdout or "CONFLICT" in pull_result.stderr:
                 logging.error("Rebase conflict detected during pull. Aborting rebase.")
                 subprocess.run(["git", "rebase", "--abort"], cwd=base_dir, capture_output=True)
                 if stashed_changes:
                     logging.info("Attempting to pop stashed changes after rebase abort.")
                     pop_after_abort_result = subprocess.run(["git", "stash", "pop"], cwd=base_dir, capture_output=True, text=True)
                     if pop_after_abort_result.returncode != 0:
                         logging.error(f"Failed to pop stash after rebase abort. Stderr: {pop_after_abort_result.stderr.strip()}")
                 logging.warning("Skipping push this cycle due to rebase conflict.")
                 return
        else:
            logging.info(f"'git pull --rebase' successful. Stdout: {pull_result.stdout.strip()}")

        if stashed_changes:
            logging.info("Attempting to pop stashed changes.")
            pop_result = subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, cwd=base_dir)
            if pop_result.returncode != 0:
                logging.error(f"git stash pop failed! This might indicate conflicts. Stderr: {pop_result.stderr.strip()}")
                logging.warning("Proceeding to add/commit script changes, but manual conflict resolution for stash might be needed later.")
            else:
                logging.info("Stashed changes popped successfully.")

        # --- CORRECTED GIT ADD LOGIC ---
        files_for_git_add = []
        history_file_abs = os.path.join(base_dir, "history.json")
        digest_state_file_abs = os.path.join(base_dir, "content.json")
        digests_dir_abs = os.path.join(base_dir, "public/digests")
        digest_manifest_abs = os.path.join(base_dir, "public/digest-manifest.json")
        latest_digest_abs = os.path.join(base_dir, "public/digest.html")
        # --- ADD THIS LINE ---
        index_html_abs = os.path.join(base_dir, "public/index.html") 


        if os.path.exists(history_file_abs): files_for_git_add.append(os.path.relpath(history_file_abs, base_dir))
        if os.path.exists(digest_state_file_abs): files_for_git_add.append(os.path.relpath(digest_state_file_abs, base_dir))
        if os.path.exists(digests_dir_abs): files_for_git_add.append(os.path.relpath(digests_dir_abs, base_dir))
        if os.path.exists(digest_manifest_abs): files_for_git_add.append(os.path.relpath(digest_manifest_abs, base_dir))
        if os.path.exists(latest_digest_abs): files_for_git_add.append(os.path.relpath(latest_digest_abs, base_dir))
        # --- AND THIS LINE ---
        if os.path.exists(index_html_abs): files_for_git_add.append(os.path.relpath(index_html_abs, base_dir))

        if files_for_git_add:
            logging.info(f"Staging script generated/modified files: {files_for_git_add}")
            add_process = subprocess.run(["git", "add"] + files_for_git_add,
                                         capture_output=True, text=True, cwd=base_dir)
            if add_process.returncode != 0:
                logging.error(f"git add command for script files failed. RC: {add_process.returncode}, Stdout: {add_process.stdout.strip()}, Stderr: {add_process.stderr.strip()}")
            else:
                logging.info(f"git add successful for: {files_for_git_add}")
        else:
            logging.info("No specific script-generated files found/modified to add.")
        # --- END CORRECTED GIT ADD LOGIC ---

        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=base_dir)
        if not status_result.stdout.strip():
            logging.info("No changes to commit after all operations. Local branch likely matches remote or no script changes.")
            return

        commit_message = f"Auto-update digest content - {datetime.now(current_zone).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        commit_cmd = ["git", "commit", "-m", commit_message]
        commit_result = subprocess.run(commit_cmd, capture_output=True, text=True, cwd=base_dir)

        if commit_result.returncode != 0:
            if "nothing to commit" in commit_result.stdout.lower() or \
               "no changes added to commit" in commit_result.stdout.lower() or \
               "your branch is up to date" in commit_result.stdout.lower():
                logging.info(f"Commit attempt reported no new changes. Stdout: {commit_result.stdout.strip()}")
                try:
                    local_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=base_dir).stdout.strip()
                    remote_head_cmd_out = subprocess.run(["git", "ls-remote", "origin", f"refs/heads/{current_branch}"], capture_output=True, text=True, cwd=base_dir)
                    if remote_head_cmd_out.returncode == 0 and remote_head_cmd_out.stdout.strip():
                        remote_head = remote_head_cmd_out.stdout.split()[0].strip()
                        if local_head == remote_head:
                            logging.info(f"Local {current_branch} is same as origin/{current_branch}. No push needed.")
                            return
                    logging.info("Local commit differs from remote or remote check failed. Will attempt push.")
                except Exception as e_rev:
                    logging.warning(f"Could not compare local/remote revisions: {e_rev}. Will attempt push.")
            else:
                logging.error(f"git commit command failed. RC: {commit_result.returncode}, Stdout: {commit_result.stdout.strip()}, Stderr: {commit_result.stderr.strip()}")
        else:
            logging.info(f"Commit successful: {commit_result.stdout.strip()}")

        logging.info(f"Pushing changes to origin/{current_branch}...")
        push_cmd = ["git", "push", "origin", current_branch]
        push_result = subprocess.run(push_cmd, check=True, cwd=base_dir, capture_output=True)
        logging.info(f"Content committed and pushed to GitHub on branch '{current_branch}'. Push output: {push_result.stdout.decode(errors='ignore').strip() if push_result.stdout else ''}")

    except subprocess.CalledProcessError as e:
        output_str = e.output.decode(errors='ignore') if hasattr(e, 'output') and e.output else ""
        stderr_str = e.stderr.decode(errors='ignore') if hasattr(e, 'stderr') and e.stderr else ""
        logging.error(f"Git operation failed: {e}. Command: '{e.cmd}'. Output: {output_str}. Stderr: {stderr_str}")
    except Exception as e:
        logging.error(f"General error during Git operations: {e}", exc_info=True)
        
def main():
    try:
        # --- STAGE 1: LOAD HISTORY & FETCH ARTICLES ---
        logging.info("--- STAGE 1: Loading History and Fetching Articles ---")
        
        # Load the user-facing history from recent HTML digests. This is the context passed to the LLM.
        recent_digest_headlines_for_llm = load_recent_digest_history(DIGESTS_DIR, DIGEST_MANIFEST_FILE, MAX_HISTORY_DIGESTS)

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.critical("Missing GEMINI_API_KEY environment variable. Exiting.")
            sys.exit(1)

        all_fetched_headlines = defaultdict(list)
        full_articles_map_this_run = {}
        articles_to_fetch_per_topic = int(CONFIG.get("ARTICLES_TO_FETCH_PER_TOPIC", 10))

        logging.info("Fetching all candidate articles...")
        for topic_name in TOPIC_WEIGHTS:
            fetched_articles = fetch_articles_for_topic(topic_name, articles_to_fetch_per_topic)
            if fetched_articles:
                for art in fetched_articles:
                    all_fetched_headlines[topic_name].append(art["title"])
                    norm_title_key = normalize(art["title"])
                    if norm_title_key not in full_articles_map_this_run:
                         full_articles_map_this_run[norm_title_key] = art
        
        logging.info(f"Fetched a total of {len(full_articles_map_this_run)} unique articles across {len(all_fetched_headlines)} topics.")

        # --- STAGE 2: AGGRESSIVE PRE-FILTERING IN PYTHON ---
        logging.info("--- STAGE 2: Pre-filtering Candidates ---")
        
        banned_terms = [k for k, v in OVERRIDES.items() if v == 'ban']
        
        # Stage 2a: Basic Filtering (History, Bans, Duplicates)
        stage1_filtered = pre_filter_and_deduplicate_headlines(
            all_fetched_headlines, recent_digest_headlines_for_llm, banned_terms
        )
        stage1_count = sum(len(h) for h in stage1_filtered.values())
        logging.info(f"Stage 1 filter (history/bans/dupes) reduced candidates to {stage1_count} headlines.")

        # Stage 2b: Topic Selection Filter (reduces the payload for the AI)
        max_topics_for_gemini = int(CONFIG.get("MAX_TOPICS_FOR_GEMINI", 40))
        final_candidates_for_gemini = select_top_candidate_topics(
            stage1_filtered, TOPIC_WEIGHTS, max_topics_for_gemini
        )
        final_candidates_count = sum(len(h) for h in final_candidates_for_gemini.values())
        logging.info(f"Stage 2 filter (topic selection) reduced candidates to {final_candidates_count} headlines across {len(final_candidates_for_gemini)} topics.")

        # --- STAGE 3: CALL GEMINI FOR PRIORITIZATION ---
        logging.info("--- STAGE 3: Calling Gemini with a refined candidate list ---")
        
        gemini_processed_content = {}
        if not final_candidates_for_gemini:
            logging.info("No headlines remained after all pre-filtering. No call to Gemini needed.")
        else:
            selected_content_raw_from_llm = prioritize_with_gemini(
                headlines_to_send=final_candidates_for_gemini,
                digest_history=recent_digest_headlines_for_llm,
                gemini_api_key=gemini_api_key,
                topic_weights=TOPIC_WEIGHTS,
                keyword_weights=KEYWORD_WEIGHTS,
                overrides=OVERRIDES
            )

            # --- STAGE 4: MAP LLM RESPONSE BACK TO FULL ARTICLE DATA ---
            logging.info("--- STAGE 4: Mapping Gemini response to full article data ---")
            if not selected_content_raw_from_llm:
                logging.warning("Gemini returned no content or invalid format.")
            else:
                logging.info(f"Processing {len(selected_content_raw_from_llm)} topics returned by Gemini.")
                seen_normalized_titles_in_llm_output = set()

                for topic_from_llm, titles_from_llm in selected_content_raw_from_llm.items():
                    current_topic_articles_for_digest = []
                    # titles_from_llm is now a standard Python list, so slicing is safe
                    for title_from_llm in titles_from_llm[:MAX_ARTICLES_PER_TOPIC]:
                        norm_llm_title = normalize(title_from_llm)
                        if not norm_llm_title or norm_llm_title in seen_normalized_titles_in_llm_output:
                            continue

                        article_data = full_articles_map_this_run.get(norm_llm_title)
                        if article_data:
                            current_topic_articles_for_digest.append(article_data)
                            seen_normalized_titles_in_llm_output.add(norm_llm_title)
                        else:
                            # Fallback logic for slightly modified titles
                            found_fallback = False
                            for stored_norm_title, stored_article_data in full_articles_map_this_run.items():
                                if norm_llm_title in stored_norm_title or stored_norm_title in norm_llm_title:
                                    if stored_norm_title not in seen_normalized_titles_in_llm_output:
                                        current_topic_articles_for_digest.append(stored_article_data)
                                        seen_normalized_titles_in_llm_output.add(stored_norm_title)
                                        found_fallback = True
                                        break
                            if not found_fallback:
                                logging.warning(f"Could not map LLM title '{title_from_llm}' back to a fetched article.")
                    
                    if current_topic_articles_for_digest:
                        gemini_processed_content[topic_from_llm] = current_topic_articles_for_digest

        final_digest_for_display_and_state = gemini_processed_content

        # --- STAGE 5: GENERATE OUTPUTS AND UPDATE STATE ---
        logging.info("--- STAGE 5: Generating outputs and updating state files ---")
        if final_digest_for_display_and_state:
            logging.info(f"Final digest contains {len(final_digest_for_display_and_state)} topics, ordered by Gemini.")
            
            new_digest_html = generate_digest_html_content(final_digest_for_display_and_state, ZONE)

            with open(LATEST_DIGEST_HTML_FILE, "w", encoding="utf-8") as f:
                f.write(new_digest_html)
            logging.info(f"Latest digest content written to {LATEST_DIGEST_HTML_FILE}")

            try:
                template_path = os.path.join(BASE_DIR, "public", "index.template.html")
                final_index_path = os.path.join(BASE_DIR, "public", "index.html")
                with open(template_path, "r", encoding="utf-8") as f:
                    index_content = f.read().replace("<!--LATEST_DIGEST_CONTENT_PLACEHOLDER-->", new_digest_html)
                with open(final_index_path, "w", encoding="utf-8") as f:
                    f.write(index_content)
                logging.info(f"Generated final index.html with embedded digest.")
            except Exception as e:
                logging.error(f"Failed to embed latest digest into index.html from template: {e}")

            update_digest_history(new_digest_html, ZONE)

            try:
                now_utc_iso = datetime.now(ZoneInfo("UTC")).isoformat()
                content_json_to_save = {
                    topic: {"articles": articles, "last_updated_ts": now_utc_iso}
                    for topic, articles in final_digest_for_display_and_state.items()
                }
                with open(DIGEST_STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(content_json_to_save, f, indent=2)
                logging.info(f"Snapshot of current digest saved to {DIGEST_STATE_FILE}")
            except IOError as e:
                logging.error(f"Failed to write digest state file {DIGEST_STATE_FILE}: {e}")
        else:
            logging.info("No topics selected by Gemini this run. Output files are not modified.")

        # Load the full history log for updating, to preserve older data not included in the LLM context
        full_history = {}
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    full_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                logging.warning("Could not read full history for update, will create a new one.")

        update_history_file(final_digest_for_display_and_state, full_history, HISTORY_FILE, ZONE)

        if CONFIG.get("ENABLE_GIT_PUSH", False):
            perform_git_operations(BASE_DIR, ZONE, CONFIG)
        else:
            logging.info("Git push is disabled in config. Skipping.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
    finally:
        logging.info(f"Script finished at {datetime.now(ZONE)}")
                  
if __name__ == "__main__":
    main()