# Author: Blake Rayvid <https://github.com/brayvid/based-news>
# Version: Corrected to emulate the successful logic of Script B

import os
import sys
import csv
import logging
import json
import re
import ast
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import requests
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from proto.marshal.collections.repeated import RepeatedComposite
from proto.marshal.collections.maps import MapComposite
import psycopg2
from psycopg2 import extras
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.data import find
import nltk

# --- SETUP: LOAD ENVIRONMENT VARIABLES FIRST ---
load_dotenv()

# Robust BASE_DIR definition
try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATABASE_URL = os.environ.get('DATABASE_URL')
CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"
TOPICS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=0&single=true&output=csv"
KEYWORDS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=314441026&single=true&output=csv"
OVERRIDES_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=1760236101&single=true&output=csv"

# --- Logging and NLP Setup ---
log_path = os.path.join(BASE_DIR, "logs/digest.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Worker script started at {datetime.now()}")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def ensure_nltk_data():
    if os.getenv('CI'):
        nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
    else:
        nltk.data.path.append(os.path.expanduser("~/nltk_data"))
    for resource in ['wordnet', 'omw-1.4']:
        try: find(f'corpora/{resource}')
        except LookupError: nltk.download(resource, download_dir=nltk.data.path[-1], quiet=True)
ensure_nltk_data()

# --- Configuration Loading ---
def load_config_from_sheet(url):
    try:
        response = requests.get(url, timeout=15); response.raise_for_status()
        config = {}
        reader = csv.reader(response.text.splitlines()); next(reader, None)
        for row in reader:
            if len(row) >= 2:
                key, val = row[0].strip(), row[1].strip()
                try: config[key] = int(val) if '.' not in val else float(val)
                except ValueError: config[key] = {'true': True, 'false': False}.get(val.lower(), val)
        return config
    except Exception as e: logging.error(f"Failed to load config from {url}: {e}"); return None
CONFIG = load_config_from_sheet(CONFIG_CSV_URL)
if CONFIG is None: logging.critical("Fatal: Unable to load CONFIG. Exiting."); sys.exit(1)

MAX_ARTICLE_HOURS = int(CONFIG.get("MAX_ARTICLE_HOURS", 6))
MAX_TOPICS = int(CONFIG.get("MAX_TOPICS", 10))
MAX_ARTICLES_PER_TOPIC = int(CONFIG.get("MAX_ARTICLES_PER_TOPIC", 1))
MAX_HISTORY_DIGESTS = int(CONFIG.get("MAX_HISTORY_DIGESTS", 12))
GEMINI_MODEL_NAME = CONFIG.get("DIGEST_GEMINI_MODEL_NAME", "gemini-2.5-flash")
ZONE = ZoneInfo(CONFIG.get("TIMEZONE", "America/New_York"))

def load_csv_data(url, is_overrides=False):
    try:
        response = requests.get(url, timeout=15); response.raise_for_status()
        data = {}
        reader = csv.reader(response.text.splitlines()); next(reader, None)
        for row in reader:
            if len(row) >= 2:
                key, val = row[0].strip(), row[1].strip()
                if is_overrides: data[key.lower()] = val.lower()
                else:
                    try: data[key] = int(val)
                    except ValueError: continue
        return data
    except Exception as e: logging.error(f"Failed to load data from {url}: {e}"); return None
TOPIC_WEIGHTS = load_csv_data(TOPICS_CSV_URL)
KEYWORD_WEIGHTS = load_csv_data(KEYWORDS_CSV_URL)
OVERRIDES = load_csv_data(OVERRIDES_CSV_URL, is_overrides=True)
if None in (TOPIC_WEIGHTS, KEYWORD_WEIGHTS, OVERRIDES):
    logging.critical("Fatal: Failed to load topics, keywords, or overrides. Exiting."); sys.exit(1)

# --- Helper Functions ---
def normalize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    lemmatized = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words]
    return " ".join(lemmatized)

def fetch_articles_for_topic(topic, max_articles=10):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(topic)}&hl=en-US&gl=US&ceid=US:en"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        time_cutoff_utc = datetime.now(ZoneInfo("UTC")) - timedelta(hours=MAX_ARTICLE_HOURS)
        articles = []

        for item in root.findall("./channel/item"):
            try:  # --- ADDED: Inner try-except block for each article ---
                title_element = item.find("title")
                title = title_element.text.strip() if title_element is not None and title_element.text else None
                
                link_element = item.find("link")
                link = link_element.text if link_element is not None else None

                pubDate_element = item.find("pubDate")
                pubDate_text = pubDate_element.text if pubDate_element is not None else None

                if not all([title, link, pubDate_text]):
                    # Silently skip if essential data is missing, or log a warning if you prefer
                    continue

                pub_dt_naive = parsedate_to_datetime(pubDate_text)
                if pub_dt_naive.tzinfo is None:
                    pub_dt_utc = pub_dt_naive.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    pub_dt_utc = pub_dt_naive.astimezone(ZoneInfo("UTC"))

                if pub_dt_utc <= time_cutoff_utc:
                    continue

                articles.append({
                    "title": title,
                    "link": link,
                    "pubDate": pubDate_text
                })

                if len(articles) >= max_articles:
                    break
            except Exception as e:
                # If one article fails to parse, log it and continue with the next one.
                # This prevents the whole topic from failing.
                logging.warning(f"Skipping one article in topic '{topic}' due to parse error: {e}")
                continue

        if articles:
            logging.info(f"Successfully fetched {len(articles)} recent articles for topic '{topic}'.")
        return articles
        
    except Exception as e:
        logging.error(f"Major error fetching articles for topic '{topic}': {e}", exc_info=True)
        return []
     
def contains_banned_keyword(text, banned_terms):
    return any(term in normalize(text) for term in banned_terms)

# --- Database Functions ---
def get_db_connection():
    if not DATABASE_URL: logging.critical("DATABASE_URL not set."); sys.exit(1)
    return psycopg2.connect(DATABASE_URL)

def load_history_from_db(max_digests):
    """Loads a flat list of recent headlines from the DB to use as context for the LLM."""
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT id FROM digests ORDER BY created_at DESC LIMIT %s", (max_digests,))
        digest_ids = [row[0] for row in cur.fetchall()]
        if not digest_ids: return []
        MAX_HISTORY_HEADLINES_FOR_LLM = int(CONFIG.get("MAX_HISTORY_HEADLINES_FOR_LLM", 150))
        cur.execute("SELECT title FROM articles WHERE digest_id = ANY(%s) ORDER BY pub_date DESC", (digest_ids,))
        headlines = [row[0] for row in cur.fetchmany(MAX_HISTORY_HEADLINES_FOR_LLM)]
        cur.close(); conn.close()
        logging.info(f"Loaded {len(headlines)} headlines from DB for LLM context.")
        return headlines
    except Exception as e: logging.error(f"Could not load history from DB: {e}"); return []

def save_digest_to_db(digest_content):
    conn = None
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("INSERT INTO digests (created_at) VALUES (NOW()) RETURNING id")
        digest_id = cur.fetchone()[0]
        articles_to_insert = [
            (digest_id, topic, art["title"], art["link"], parsedate_to_datetime(art["pubDate"]).astimezone(ZoneInfo("UTC")), i)
            for topic, articles in digest_content.items() for i, art in enumerate(articles)
        ]
        extras.execute_values(cur, "INSERT INTO articles (digest_id, topic, title, link, pub_date, display_order) VALUES %s", articles_to_insert)
        cur.execute("DELETE FROM digests WHERE id IN (SELECT id FROM digests ORDER BY created_at DESC OFFSET %s)", (MAX_HISTORY_DIGESTS,))
        conn.commit()
        logging.info(f"Saved digest {digest_id} with {len(articles_to_insert)} articles to DB.")
    except Exception as e:
        if conn: conn.rollback()
        logging.error(f"Failed to save digest to DB: {e}", exc_info=True)
    finally:
        if conn: conn.close()

# --- START: TRANSPLANTED BLOCK FROM SCRIPT B (THE CORE FIX & IMPROVEMENT) ---
digest_tool_schema = { "type": "object", "properties": { "selected_digest_entries": { "type": "array", "items": { "type": "object", "properties": { "topic_name": {"type": "string"}, "headlines": {"type": "array", "items": {"type": "string"}}}, "required": ["topic_name", "headlines"]}}}, "required": ["selected_digest_entries"]}
SELECT_DIGEST_ARTICLES_TOOL = Tool(function_declarations=[FunctionDeclaration(name="format_digest_selection", description="Formats the selected news.", parameters=digest_tool_schema)])

def prioritize_with_gemini(
    headlines_to_send: dict,
    digest_history: list,
    gemini_api_key: str,
    topic_weights: dict,
    keyword_weights: dict,
    overrides: dict
) -> dict:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, tools=[SELECT_DIGEST_ARTICLES_TOOL])

    pref_data = {"topic_weights": topic_weights, "keyword_weights": keyword_weights, "banned_terms": [k for k, v in overrides.items() if v == "ban"], "demoted_terms": [k for k, v in overrides.items() if v == "demote"]}
    
    prompt = (
        "You are an Advanced News Synthesis Engine. Your function is to act as an expert, hyper-critical news curator. Your single most important mission is to produce a high-signal, non-redundant, and deeply relevant news digest for a user. You must be ruthless in eliminating noise, repetition, and low-quality content.\n\n"
        f"### Inputs Provided\n1.  **User Preferences:**\n```json\n{json.dumps(pref_data, indent=2)}\n```\n"
        f"2.  **Candidate Headlines:**\n```json\n{json.dumps(dict(sorted(headlines_to_send.items())), indent=2)}\n```\n"
        f"3.  **Digest History:**\n```json\n{json.dumps(digest_history, indent=2)}\n```\n\n"
        "### Core Processing Pipeline (Follow these steps sequentially)\n\n"
        "**Step 1: Cross-Topic Semantic Clustering & Deduplication (CRITICAL FIRST STEP)**\nFirst, analyze ALL `Candidate Headlines`. Your primary task is to identify and group all headlines from ALL topics that cover the same core news event. From each cluster, select ONLY ONE headline—the one that is the most comprehensive, recent, objective, and authoritative. Discard all other headlines in that cluster immediately.\n\n"
        "**Step 2: History-Based Filtering**\nNow, take your deduplicated list of 'champion' headlines. Compare each one against the `Digest History`. If any of your champion headlines reports on the exact same event that has already been sent, DISCARD it.\n\n"
        "**Step 3: Rigorous Relevance & Quality Filtering**\nFor the remaining, unique, and new headlines, apply the following strict filtering criteria with full force:\n"
        f"*   **Output Limits:** Adhere strictly to a maximum of **{MAX_TOPICS} topics** and **{MAX_ARTICLES_PER_TOPIC} headlines** per topic.\n"
        "*   **Content Quality & Style (CRITICAL):** AGGRESSIVELY AVOID AND REJECT headlines that are: Sensationalist, celebrity gossip, clickbait, primarily opinion/op-ed pieces, or resemble 'hot stock tips'. Focus on content-rich, factual, objective reporting.\n\n"
        "**Step 4: Final Selection and Ordering**\nFrom the fully filtered and vetted pool of headlines, make your final selection. Order topics and headlines from most to least significant based on a blend of user preference and objective news importance.\n\n"
        "### Final Output\nBased on this rigorous process, provide your final, curated selection using the 'format_digest_selection' tool."
    )

    logging.info("Sending request to Gemini for prioritization with history.")
    try:
        response = model.generate_content([prompt], tool_config={"function_calling_config": "any"})
        function_call_part = next((part.function_call for part in response.candidates[0].content.parts if hasattr(part, 'function_call')), None)

        if function_call_part and function_call_part.name == "format_digest_selection":
            args = function_call_part.args
            logging.info(f"Gemini used tool 'format_digest_selection' with args (type: {type(args)}).")
            transformed_output = {}
            if isinstance(args, (MapComposite, dict)):
                entries = args.get("selected_digest_entries", [])
                if isinstance(entries, (RepeatedComposite, list)):
                    for entry in entries:
                        if isinstance(entry, (MapComposite, dict)):
                            topic = entry.get("topic_name")
                            headlines = [str(h) for h in entry.get("headlines", [])]
                            if topic and headlines: transformed_output[topic.strip()] = headlines
            logging.info(f"Transformed output from Gemini tool call: {transformed_output}")
            return transformed_output
        else:
            logging.warning("Gemini did not use the expected tool."); return {}
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}", exc_info=True); return {}


def main():
    logging.info("--- Main execution starting ---")
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.critical("Missing GEMINI_API_KEY. Exiting.")
            sys.exit(1)

        # 1. Load history from DB for LLM context.
        recent_headlines_for_llm = load_history_from_db(MAX_HISTORY_DIGESTS)

        # 2. Fetch articles and apply a correctly built banned keyword filter.
        headlines_to_send_to_llm = {}
        full_articles_map_this_run = {}
        
        # --- THE DEFINITIVE FIX IS HERE ---
        # First, we normalize ALL potential banned terms.
        all_normalized_terms = [normalize(k) for k, v in OVERRIDES.items() if v == "ban"]
        # THEN, we create the final list, filtering out any that became empty strings.
        normalized_banned_terms = [term for term in all_normalized_terms if term]

        articles_to_fetch_per_topic = int(CONFIG.get("ARTICLES_TO_FETCH_PER_TOPIC", 10))

        logging.info(f"Fetching articles. Valid banned terms being used: {normalized_banned_terms}")
        for topic in TOPIC_WEIGHTS:
            fetched_articles = fetch_articles_for_topic(topic, articles_to_fetch_per_topic)
            if not fetched_articles:
                continue

            current_topic_headlines = []
            for art in fetched_articles:
                if contains_banned_keyword(art['title'], normalized_banned_terms):
                    logging.debug(f"Skipping (banned keyword): {art['title']}")
                    continue
                
                current_topic_headlines.append(art['title'])
                norm_title = normalize(art['title'])
                if norm_title not in full_articles_map_this_run:
                    full_articles_map_this_run[norm_title] = art
            
            if current_topic_headlines:
                headlines_to_send_to_llm[topic] = current_topic_headlines

        if not headlines_to_send_to_llm:
            logging.warning("Still no headlines after fetching and filtering. This may be due to a slow news day or a very broad banned term. Check MAX_ARTICLE_HOURS.")
            return
        
        total_candidates = sum(len(v) for v in headlines_to_send_to_llm.values())
        logging.info(f"SUCCESS: Sending {total_candidates} candidate headlines across {len(headlines_to_send_to_llm)} topics to Gemini.")

        # 3. Call Gemini.
        selected_raw = prioritize_with_gemini(
            headlines_to_send=headlines_to_send_to_llm,
            digest_history=recent_headlines_for_llm,
            gemini_api_key=gemini_api_key,
            topic_weights=TOPIC_WEIGHTS,
            keyword_weights=KEYWORD_WEIGHTS,
            overrides=OVERRIDES
        )
        if not selected_raw:
            logging.warning("Gemini returned no content. Exiting."); return

        # 4. Map and save results.
        final_digest = {}
        seen_titles = set()
        for topic, titles in selected_raw.items():
            articles_for_topic = []
            for title in titles[:MAX_ARTICLES_PER_TOPIC]:
                norm_title = normalize(title)
                if norm_title in seen_titles: continue
                
                article_data = full_articles_map_this_run.get(norm_title)
                if article_data:
                    articles_for_topic.append(article_data)
                    seen_titles.add(norm_title)
                else:
                    logging.warning(f"Could not map LLM title '{title}' back to a fetched article.")
            
            if articles_for_topic:
                final_digest[topic] = articles_for_topic
        
        if final_digest:
            save_digest_to_db(final_digest)
        else:
            logging.info("No topics in final digest after processing. Nothing to save.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
    finally:
        logging.info(f"--- Worker script finished ---")

if __name__ == "__main__":
    main()