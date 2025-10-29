# Author: Blake Rayvid <https://github.com/brayvid/based-news>

import os
import sys
import csv
import logging
import json
import re
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
MAX_CANDIDATES_FOR_LLM = int(CONFIG.get("MAX_CANDIDATES_FOR_LLM", 500))

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

def create_title_fingerprint(title: str) -> str:
    """Creates a deterministic, normalized fingerprint from an article title."""
    base_title = title.rsplit(' - ', 1)[0]
    return normalize(base_title)

def calculate_local_score(article_title, topic, topic_weights, keyword_weights):
    """Calculates a simple score for an article based on its topic and keywords."""
    score = topic_weights.get(topic, 0)
    normalized_title = normalize(article_title)
    for keyword, weight in keyword_weights.items():
        if keyword in normalized_title:
            score += weight
    return score

def fetch_articles_for_topic(topic, max_articles=5):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(topic)}&hl=en-US&gl=US&ceid=US:en"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        time_cutoff_utc = datetime.now(ZoneInfo("UTC")) - timedelta(hours=MAX_ARTICLE_HOURS)
        articles = []
        for item in root.findall("./channel/item"):
            try:
                title_element, link_element, pubDate_element = item.find("title"), item.find("link"), item.find("pubDate")
                title = title_element.text.strip() if title_element is not None and title_element.text else None
                link = link_element.text if link_element is not None else None
                pubDate_text = pubDate_element.text if pubDate_element is not None else None
                if not all([title, link, pubDate_text]): continue
                
                pub_dt_naive = parsedate_to_datetime(pubDate_text)
                pub_dt_utc = pub_dt_naive.astimezone(ZoneInfo("UTC")) if pub_dt_naive.tzinfo else pub_dt_naive.replace(tzinfo=ZoneInfo("UTC"))
                
                if pub_dt_utc <= time_cutoff_utc: continue
                
                articles.append({"title": title, "link": link, "pubDate": pubDate_text})
                if len(articles) >= max_articles: break
            except Exception as e:
                logging.warning(f"Skipping one article in topic '{topic}' due to parse error: {e}")
                continue
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

def load_recent_titles_for_dedup(hours_to_check: int) -> list[str]:
    """Loads raw titles of all articles from digests created in the last N hours."""
    titles = []
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            """
            SELECT a.title
            FROM articles a
            JOIN digests d ON a.digest_id = d.id
            WHERE d.created_at >= NOW() - INTERVAL '%s hours'
            """,
            (hours_to_check,)
        )
        titles = [row[0] for row in cur.fetchall()]
        cur.close(); conn.close()
        logging.info(f"Loaded {len(titles)} raw titles from DB to build deduplication set.")
        return titles
    except Exception as e:
        logging.error(f"Could not load recent titles from DB: {e}")
        return titles

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

        ### MODIFIED FOR GLOBAL IMPORTANCE SORTING ###
        # The 'digest_content' dict is now ordered by importance. We use enumerate
        # to get the topic's rank and save it as the display_order for its articles.
        articles_to_insert = []
        for topic_rank, (topic, articles) in enumerate(digest_content.items()):
            for art in articles:
                articles_to_insert.append(
                    (
                        digest_id, 
                        topic, 
                        art["title"], 
                        art["link"], 
                        parsedate_to_datetime(art["pubDate"]).astimezone(ZoneInfo("UTC")), 
                        topic_rank # Use the topic's rank as the global display order
                    )
                )
        ### END MODIFICATION ###

        extras.execute_values(cur, "INSERT INTO articles (digest_id, topic, title, link, pub_date, display_order) VALUES %s", articles_to_insert)
        conn.commit()
        logging.info(f"Saved digest {digest_id} with {len(articles_to_insert)} articles to DB, preserving importance order.")
    except Exception as e:
        if conn: conn.rollback()
        logging.error(f"Failed to save digest to DB: {e}", exc_info=True)
    finally:
        if conn: conn.close()

# --- Gemini Interaction ---

# NEW SCHEMA: Now includes a mandatory 'importance_rank' field.
digest_tool_schema = {
    "type": "object",
    "properties": {
        "selected_digest_entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string"},
                    "selected_article_ids": {"type": "array", "items": {"type": "string"}},
                    "importance_rank": {
                        "type": "integer",
                        "description": "A numerical rank for the topic's importance (1 is the most important, 2 is second most important, etc.). This field is mandatory."
                    }
                },
                "required": ["topic_name", "selected_article_ids", "importance_rank"]
            }
        }
    },
    "required": ["selected_digest_entries"]
}
SELECT_DIGEST_ARTICLES_TOOL = Tool(function_declarations=[FunctionDeclaration(name="format_digest_selection", description="Formats the selected news articles using their unique IDs and assigns an importance rank to each topic.", parameters=digest_tool_schema)])

def prioritize_with_gemini(candidates_to_send: list[dict], digest_history: list, gemini_api_key: str, topic_weights: dict, keyword_weights: dict, overrides: dict) -> dict:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, tools=[SELECT_DIGEST_ARTICLES_TOOL])
    pref_data = {"topic_weights": topic_weights, "keyword_weights": keyword_weights, "banned_terms": [k for k, v in overrides.items() if v == "ban"], "demoted_terms": [k for k, v in overrides.items() if v == "demote"]}
    
    # PROMPT UPDATED with instructions to prioritize informative headlines
    prompt = (
        "You are an Advanced News Synthesis Engine. Your function is to act as an expert, hyper-critical news curator. Your single most important mission is to produce a high-signal, non-redundant, and deeply relevant news digest for a user. You must be ruthless in eliminating noise, repetition, and low-quality content.\n\n"
        f"### Inputs Provided\n1.  **User Preferences:**\n```json\n{json.dumps(pref_data, indent=2)}\n```\n"
        f"2.  **Candidate Articles:** Each article has a unique `id` for you to reference.\n```json\n{json.dumps(candidates_to_send, indent=2)}\n```\n"
        f"3.  **Digest History:**\n```json\n{json.dumps(digest_history, indent=2)}\n```\n\n"
        "### Core Processing Pipeline (Follow these steps sequentially)\n\n"
        "**Step 1: Cross-Topic Semantic Clustering & Deduplication (CRITICAL FIRST STEP)**\nFirst, analyze ALL `Candidate Articles`. Your primary task is to identify and group all articles from ALL topics that cover the same core news event. From each cluster, select ONLY ONE article—the one that is the most comprehensive, recent, objective, and authoritative. Discard all other articles in that cluster immediately.\n\n"
        "**Step 2: History-Based Filtering**\nNow, take your deduplicated list of 'champion' articles. Compare each one against the `Digest History`. If any of your champion articles reports on the exact same event that has already been sent, DISCARD it.\n\n"
        "**Step 3: Rigorous Relevance & Quality Filtering**\nFor the remaining, unique, and new articles, apply the following strict filtering criteria with full force:\n"
        f"*   **Output Limits:** Adhere strictly to a maximum of **{MAX_TOPICS} topics** and **{MAX_ARTICLES_PER_TOPIC} headlines** per topic.\n"
        "*   **Audience Focus (CRITICAL):** The digest is for a **US-specific audience**. AGGRESSIVELY REJECT articles about local or regional events outside the United States that have no significant impact on a US audience (e.g., local elections in other countries, regional crime stories, municipal news). Retain international news only if it has a clear and significant impact on US interests, politics, or the economy.\n"
        ### NEW CRITERION FOR INFORMATIVENESS ###
        "*   **Headline Informativeness (CRITICAL):** Prioritize headlines that are self-contained statements of fact. AGGRESSIVELY REJECT or heavily down-rank 'content-free' headlines. This includes:\n"
        "    *   **Vague Teasers:** Headlines that require a click to understand the basic story (e.g., 'Here's what experts are saying about the economy').\n"
        "    *   **Unanswered Questions:** Headlines phrased as questions without providing the answer (e.g., 'Will the new law pass?').\n"
        "    *   **Simple Topic Labels:** Headlines that just name a subject without reporting an event (e.g., 'A Look at the Housing Market').\n"
        "    *   **Instead, select headlines that deliver the core news directly.** For example, prefer 'Federal Reserve Holds Interest Rates Steady Amid Inflation Concerns' over 'What Will the Federal Reserve Do Next?'.\n"
        "*   **Content Quality & Style (CRITICAL):** AGGRESSIVELY AVOID AND REJECT headlines that are: Sensationalist, celebrity gossip, clickbait, primarily opinion/op-ed pieces, or resemble 'hot stock tips'. Focus on content-rich, factual, objective reporting.\n\n"
        "**Step 4: Final Selection and Ranking**\nFrom the fully filtered and vetted pool of articles, make your final selection. **For each topic you select, you must assign a numerical `importance_rank`, where 1 is the most significant topic.**\n\n"
        "### Final Output\n"
        "Based on this rigorous process, provide your final, curated selection using the 'format_digest_selection' tool. "
        "You must populate the mandatory `importance_rank` for every topic and return the unique `id` of each selected article."
    )
    # ... The rest of the function remains the same ...
    logging.info(f"Sending request to Gemini for prioritization with {len(candidates_to_send)} top candidates.")
    try:
        response = model.generate_content([prompt], tool_config={"function_calling_config": "any"})
        function_call_part = next((part.function_call for part in response.candidates[0].content.parts if hasattr(part, 'function_call')), None)
        if function_call_part and function_call_part.name == "format_digest_selection":
            args = function_call_part.args
            logging.info("Gemini used tool 'format_digest_selection'.")
            ranked_results = []
            if isinstance(args, (MapComposite, dict)):
                entries = args.get("selected_digest_entries", [])
                if isinstance(entries, (RepeatedComposite, list)):
                    for entry in entries:
                        if isinstance(entry, (MapComposite, dict)):
                            topic = entry.get("topic_name")
                            rank = entry.get("importance_rank")
                            article_ids = [str(aid) for aid in entry.get("selected_article_ids", [])]
                            if rank is None:
                                logging.warning(f"Model failed to provide importance_rank for topic '{topic}'. Defaulting to 99.")
                                rank = 99
                            if topic and article_ids:
                                ranked_results.append((int(rank), topic.strip(), article_ids))
            ranked_results.sort()
            ordered_selected_ids = {topic: ids for rank, topic, ids in ranked_results}
            logging.info(f"Transformed and sorted ID output from Gemini: {ordered_selected_ids}")
            return ordered_selected_ids
        else:
            logging.warning("Gemini did not use the expected tool."); return {}
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}", exc_info=True); return {}
     
# --- Main Execution ---
def main():
    logging.info("--- Main execution starting ---")
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.critical("Missing GEMINI_API_KEY. Exiting.")
            sys.exit(1)

        # 1. Load history and build in-memory deduplication set.
        recent_headlines_for_llm = load_history_from_db(MAX_HISTORY_DIGESTS)
        recent_raw_titles = load_recent_titles_for_dedup(hours_to_check=48)
        published_title_fingerprints = {create_title_fingerprint(t) for t in recent_raw_titles}
        logging.info(f"Created in-memory set with {len(published_title_fingerprints)} unique fingerprints for deduplication.")

        # 2. Fetch, filter, and score all articles locally.
        all_candidate_articles = []
        seen_fingerprints_this_run = set()
        # In the main() function

        all_candidate_articles = []
        seen_fingerprints_this_run = set()
        
        # --- ROBUST BANNED KEYWORD PREPARATION ---
        all_normalized_terms = [normalize(k) for k, v in OVERRIDES.items() if v == "ban"]
        # THEN, we create the final list, filtering out any that became empty strings.
        normalized_banned_terms = [term for term in all_normalized_terms if term]
        if len(all_normalized_terms) != len(normalized_banned_terms):
            logging.warning("Filtered out empty or whitespace-only banned keywords from your overrides list.")
        logging.info(f"Using {len(normalized_banned_terms)} valid banned keywords for filtering.")

        
        logging.info("--- Starting Article Fetching and Filtering Stage ---")
        for topic in TOPIC_WEIGHTS:
            logging.debug(f"\n===== Processing Topic: {topic} =====")
            fetched_articles = fetch_articles_for_topic(topic, int(CONFIG.get("ARTICLES_TO_FETCH_PER_TOPIC", 5)))
            
            if not fetched_articles:
                logging.debug(f"No recent articles returned from RSS feed for topic: '{topic}'.")
                continue

            logging.debug(f"Fetched {len(fetched_articles)} articles for '{topic}'. Evaluating each...")
            for art in fetched_articles:
                title = art['title']
                fingerprint = create_title_fingerprint(title)
                
                logging.debug(f"  -> Evaluating: '{title}'")

                # Perform all deterministic checks first
                if fingerprint in published_title_fingerprints:
                    logging.debug(f"     [FILTERED] Reason: Already published in a recent digest.")
                    continue
                
                if fingerprint in seen_fingerprints_this_run:
                    logging.debug(f"     [FILTERED] Reason: Duplicate article found in this run.")
                    continue

                if contains_banned_keyword(title, normalized_banned_terms):
                    logging.debug(f"     [FILTERED] Reason: Contains a banned keyword.")
                    continue
                
                # If all checks pass, it's a valid candidate
                logging.info(f"     [ACCEPTED] Adding as candidate: '{title}'")
                score = calculate_local_score(title, topic, TOPIC_WEIGHTS, KEYWORD_WEIGHTS)
                all_candidate_articles.append({'score': score, 'article_data': art, 'topic': topic})
                seen_fingerprints_this_run.add(fingerprint)

        if not all_candidate_articles:
            logging.warning("No new, unique headlines found after fetching and filtering. Exiting run.")
            # The detailed logs above will now explain WHY this happened.
            logging.info("--- Worker script finished ---")
            return

        # 3. Select the top candidates to send to the LLM.
        # ... (rest of your main function is unchanged) ...
        all_candidate_articles.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = all_candidate_articles[:MAX_CANDIDATES_FOR_LLM]

        # 4. Prepare data for Gemini with unique IDs for robust mapping.
        id_to_article_map = {}
        candidates_for_gemini = []
        for i, candidate in enumerate(top_candidates):
            article_id = f"art_{i:03d}"
            id_to_article_map[article_id] = candidate['article_data']
            candidates_for_gemini.append({"id": article_id, "topic": candidate['topic'], "title": candidate['article_data']['title']})
        
        logging.info(f"Locally scored and filtered down to {len(candidates_for_gemini)} top candidates to send to Gemini.")

        # 5. Call Gemini with the curated list.
        selected_ids_by_topic = prioritize_with_gemini(
            candidates_to_send=candidates_for_gemini,
            digest_history=recent_headlines_for_llm,
            gemini_api_key=gemini_api_key,
            topic_weights=TOPIC_WEIGHTS,
            keyword_weights=KEYWORD_WEIGHTS,
            overrides=OVERRIDES
        )
        if not selected_ids_by_topic:
            logging.warning("Gemini returned no content. Exiting."); return

        # 6. Map results using the robust IDs and save.
        final_digest = {}
        seen_article_ids = set()
        for topic, article_ids in selected_ids_by_topic.items():
            articles_for_topic = []
            for article_id in article_ids[:MAX_ARTICLES_PER_TOPIC]:
                if article_id in seen_article_ids: continue
                
                article_data = id_to_article_map.get(article_id)
                if article_data:
                    articles_for_topic.append(article_data)
                    seen_article_ids.add(article_id)
                else:
                    logging.warning(f"Gemini returned an unknown article ID '{article_id}' for topic '{topic}'.")
            
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