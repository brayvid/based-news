# Author: Blake Rayvid <https://github.com/brayvid/based-news>

import os
import sys
import csv
import logging
import json
import re
import time
import random
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import requests
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import psycopg2
from psycopg2 import extras
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.data import find
import nltk

# --- SETUP: LOAD ENVIRONMENT VARIABLES FIRST ---
load_dotenv()

try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATABASE_URL = os.environ.get('DATABASE_URL')
CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"
TOPICS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=0&single=true&output=csv"
KEYWORDS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=314441026&single=true&output=csv"
OVERRIDES_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=1760236101&single=true&output=csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    except Exception as e: logging.error(f"Failed to load config: {e}"); return None

CONFIG = load_config_from_sheet(CONFIG_CSV_URL)
if CONFIG is None: sys.exit(1)

MAX_ARTICLE_HOURS = int(CONFIG.get("MAX_ARTICLE_HOURS", 12))
MAX_TOPICS = int(CONFIG.get("MAX_TOPICS", 7))
MAX_ARTICLES_PER_TOPIC = int(CONFIG.get("MAX_ARTICLES_PER_TOPIC", 1))
MAX_HISTORY_DIGESTS = int(CONFIG.get("MAX_HISTORY_DIGESTS", 12))
GEMINI_MODEL_NAME = CONFIG.get("DIGEST_GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
ZONE = ZoneInfo(CONFIG.get("TIMEZONE", "America/New_York"))
MAX_CANDIDATES_FOR_LLM = int(CONFIG.get("MAX_CANDIDATES_FOR_LLM", 150))
BATCH_SIZE = 10

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
    except Exception as e: logging.error(f"Failed to load data: {e}"); return None

TOPIC_WEIGHTS = load_csv_data(TOPICS_CSV_URL)
KEYWORD_WEIGHTS = load_csv_data(KEYWORDS_CSV_URL)
OVERRIDES = load_csv_data(OVERRIDES_CSV_URL, is_overrides=True)

# --- Helpers ---
def normalize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return " ".join([lemmatizer.lemmatize(stemmer.stem(w)) for w in words])

def create_title_fingerprint(title: str) -> str:
    return normalize(title.rsplit(' - ', 1)[0])

def calculate_local_score(article_title, topic, topic_weights, keyword_weights):
    score = topic_weights.get(topic, 0)
    normalized_title = normalize(article_title)
    for keyword, weight in keyword_weights.items():
        if keyword in normalized_title: score += weight
    return score

def fetch_articles_for_batch(topics_batch):
    query_string = " OR ".join([f'"{t}"' for t in topics_batch])
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(f'({query_string})')}&hl=en-US&gl=US&ceid=US:en"
    user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"]
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers={"User-Agent": random.choice(user_agents)}, timeout=25)
            if response.status_code == 503:
                time.sleep((attempt + 1) * 7); continue
            response.raise_for_status()
            root = ET.fromstring(response.content)
            cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(hours=MAX_ARTICLE_HOURS)
            articles = []
            for item in root.findall("./channel/item"):
                title = item.find("title").text.strip()
                link = item.find("link").text
                pubDate = item.find("pubDate").text
                try:
                    pub_dt = parsedate_to_datetime(pubDate).astimezone(ZoneInfo("UTC"))
                    if pub_dt > cutoff: articles.append({"title": title, "link": link, "pubDate": pubDate})
                except: continue
            return articles
        except Exception as e:
            if attempt == 2: logging.error(f"Batch fetch error: {e}")
            time.sleep(2)
    return []

def contains_banned_keyword(text, banned_terms):
    return any(term in normalize(text) for term in banned_terms if term)

# --- DB Logic ---
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def load_recent_titles_for_dedup(hours_to_check: int):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT a.title FROM articles a JOIN digests d ON a.digest_id = d.id WHERE d.created_at >= NOW() - INTERVAL '%s hours'", (hours_to_check,))
        titles = [row[0] for row in cur.fetchall()]
        cur.close(); conn.close()
        return titles
    except: return []

def load_history_from_db(max_digests):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT id FROM digests ORDER BY created_at DESC LIMIT %s", (max_digests,))
        ids = [row[0] for row in cur.fetchall()]
        if not ids: return []
        cur.execute("SELECT title FROM articles WHERE digest_id = ANY(%s) ORDER BY pub_date DESC", (ids,))
        headlines = [row[0] for row in cur.fetchmany(150)]
        cur.close(); conn.close()
        return headlines
    except: return []

def save_digest_to_db(digest_list):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("INSERT INTO digests (created_at) VALUES (NOW()) RETURNING id")
        digest_id = cur.fetchone()[0]
        articles_to_insert = []
        for rank, (topic, articles) in enumerate(digest_list):
            for art in articles:
                articles_to_insert.append((digest_id, topic, art["title"], art["link"], parsedate_to_datetime(art["pubDate"]).astimezone(ZoneInfo("UTC")), rank))
        extras.execute_values(cur, "INSERT INTO articles (digest_id, topic, title, link, pub_date, display_order) VALUES %s", articles_to_insert)
        conn.commit(); cur.close(); conn.close()
        logging.info(f"Successfully saved digest {digest_id} to database.")
    except Exception as e: logging.error(f"Database save failed: {e}")

# --- Gemini Interaction ---
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

SELECT_DIGEST_ARTICLES_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="format_digest_selection",
            description="Formats the selected news articles using their unique IDs and assigns an importance rank to each topic.",
            parameters=digest_tool_schema
        )
    ]
)

def prioritize_with_gemini(candidates_to_send, digest_history, gemini_api_key, topic_weights, keyword_weights, overrides):
    client = genai.Client(api_key=gemini_api_key)
    # Reverting to exact original pref_data which includes demoted_terms
    pref_data = {
        "topic_weights": topic_weights, 
        "keyword_weights": keyword_weights, 
        "banned_terms": [k for k, v in overrides.items() if v == "ban"], 
        "demoted_terms": [k for k, v in overrides.items() if v == "demote"]
    }
    
    # EXACT PROMPT PRESERVED
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

    # Retry Logic for 503 Unavailable
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME, 
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[SELECT_DIGEST_ARTICLES_TOOL], 
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    )
                )
            )
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    entries = part.function_call.args.get("selected_digest_entries", [])
                    results = [(int(e.get("importance_rank", 99)), e["topic_name"], e["selected_article_ids"]) for e in entries]
                    results.sort(); return results
            return []
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                logging.warning(f"Gemini 503 (Busy). Retrying in 10s... (Attempt {attempt+1}/3)")
                time.sleep(10)
                continue
            logging.error(f"Gemini error: {e}"); return []
    return []

def main():
    logging.info("--- Main execution starting ---")
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        recent_history = load_history_from_db(MAX_HISTORY_DIGESTS)
        recent_raw = load_recent_titles_for_dedup(48)
        published_fingerprints = {create_title_fingerprint(t) for t in recent_raw}

        all_candidates = []
        seen_run = set()
        banned = [normalize(k) for k, v in OVERRIDES.items() if v == "ban"]

        topic_keys = list(TOPIC_WEIGHTS.keys())
        random.shuffle(topic_keys)
        batches = [topic_keys[i:i + BATCH_SIZE] for i in range(0, len(topic_keys), BATCH_SIZE)]
        
        logging.info(f"--- Starting Article Fetching (Batch Mode: {len(batches)} batches) ---")

        for batch in batches:
            batch_arts = fetch_articles_for_batch(batch)
            time.sleep(random.uniform(1.5, 3.0))

            for art in batch_arts:
                title = art['title']
                norm_title = normalize(title)
                fingerprint = create_title_fingerprint(title)
                
                if fingerprint in published_fingerprints or fingerprint in seen_run: continue
                
                # Robust Attribution Logic
                best_topic, highest_w = None, -1
                for topic in batch:
                    norm_topic = normalize(topic)
                    if norm_topic in norm_title or any(word in norm_title for word in norm_topic.split() if len(word) > 3):
                        weight = TOPIC_WEIGHTS.get(topic, 0)
                        if weight > highest_w: highest_w, best_topic = weight, topic
                
                if not best_topic:
                    best_topic = max(batch, key=lambda t: TOPIC_WEIGHTS.get(t, 0))

                if contains_banned_keyword(title, banned): continue
                
                score = calculate_local_score(title, best_topic, TOPIC_WEIGHTS, KEYWORD_WEIGHTS)
                all_candidates.append({'score': score, 'article_data': art, 'topic': best_topic})
                seen_run.add(fingerprint)

        logging.info(f"Locally scored {len(all_candidates)} unique candidate articles.")

        if not all_candidates:
            logging.warning("No new unique headlines found. Exiting."); return

        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = all_candidates[:MAX_CANDIDATES_FOR_LLM]

        id_map, llm_in = {}, []
        for i, cand in enumerate(top_candidates):
            aid = f"art_{i:03d}"
            id_map[aid] = cand['article_data']
            llm_in.append({"id": aid, "topic": cand['topic'], "title": cand['article_data']['title']})

        logging.info(f"Sending {len(llm_in)} candidates to Gemini for final curation.")
        ranked_topics = prioritize_with_gemini(llm_in, recent_history, gemini_api_key, TOPIC_WEIGHTS, KEYWORD_WEIGHTS, OVERRIDES)
        
        if not ranked_topics:
            logging.warning("Gemini returned no results. Exiting."); return

        logging.info(f"Gemini curated {len(ranked_topics)} topics.")
        final_digest, seen_ids = [], set()
        for rank, topic, aids in ranked_topics[:MAX_TOPICS]:
            topic_arts = []
            for aid in aids[:MAX_ARTICLES_PER_TOPIC]:
                if aid in seen_ids or aid not in id_map: continue
                topic_arts.append(id_map[aid]); seen_ids.add(aid)
            if topic_arts: final_digest.append((topic, topic_arts))
        
        if final_digest: save_digest_to_db(final_digest)

    except Exception as e: logging.critical(f"Unhandled error: {e}", exc_info=True)
    finally: logging.info("--- Worker script finished ---")

if __name__ == "__main__":
    main()
    sys.exit(0)