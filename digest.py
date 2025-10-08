# Author: Blake Rayvid <https://github.com/brayvid/based-news>

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
from nltk.stem import PorterStemmer, WordNetLemmatizer
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from proto.marshal.collections.repeated import RepeatedComposite
from proto.marshal.collections.maps import MapComposite
import psycopg2
from psycopg2 import extras

# Set number of threads for various libraries to 1 if parallelism is not permitted on your system
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Robust BASE_DIR definition
try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:  # __file__ is not defined, e.g., in interactive shell
    BASE_DIR = os.getcwd()

# Load Database URL from environment variables, this is the primary connection method
DATABASE_URL = os.environ.get('DATABASE_URL')

# URLs for remote configuration
CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"
TOPICS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=0&single=true&output=csv"
KEYWORDS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=314441026&single=true&output=csv"
OVERRIDES_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=1760236101&single=true&output=csv"

# Initialize logging
log_path = os.path.join(BASE_DIR, "logs/digest.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Worker script started at {datetime.now()}")

# Initialize NLP tools and load environment variables
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
                    if '.' in val and not val.startswith('0') and val.count('.') == 1:
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
GEMINI_MODEL_NAME = CONFIG.get("DIGEST_GEMINI_MODEL_NAME", "gemini-2.5-flash")

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
    stemmed = [stemmer.stem(w) for w in words]
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]
    return " ".join(lemmatized)

def is_high_confidence_duplicate_in_history(article_title: str, history: dict, threshold: float) -> bool:
    norm_title_tokens = set(normalize(article_title).split())
    if not norm_title_tokens:
        return False

    for articles_in_topic in history.values():
        for past_article_data in articles_in_topic:
            past_title = past_article_data.get("title", "")
            past_tokens = set(normalize(past_title).split())
            if not past_tokens:
                continue
            
            intersection_len = len(norm_title_tokens.intersection(past_tokens))
            union_len = len(norm_title_tokens.union(past_tokens))
            if union_len == 0: continue

            similarity = intersection_len / union_len
            if similarity >= threshold:
                logging.debug(f"Article '{article_title}' is a high-confidence match to past article '{past_title}' with similarity {similarity:.2f} >= {threshold}")
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
                pub_dt_utc = parsedate_to_datetime(pubDate)
                if pub_dt_utc.tzinfo is None:
                    pub_dt_utc = pub_dt_utc.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    pub_dt_utc = pub_dt_utc.astimezone(ZoneInfo("UTC"))
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

def get_db_connection():
    """Establishes a connection to the database."""
    if not DATABASE_URL:
        logging.critical("DATABASE_URL environment variable is not set.")
        sys.exit(1)
    return psycopg2.connect(DATABASE_URL)

def load_recent_headlines_from_db(max_digests):
    """Loads headlines from recent digests stored in the database."""
    history_headlines = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT id FROM digests ORDER BY created_at DESC LIMIT %s", (max_digests,))
        digest_ids = [row[0] for row in cur.fetchall()]
        
        if digest_ids:
            cur.execute("SELECT title FROM articles WHERE digest_id = ANY(%s)", (digest_ids,))
            history_headlines = [row[0] for row in cur.fetchall()]
            
        cur.close()
        conn.close()
        logging.info(f"Loaded {len(history_headlines)} unique headlines from the {len(digest_ids)} most recent digests in the DB.")
    except Exception as e:
        logging.error(f"Could not load digest history from database: {e}")
    
    return list(dict.fromkeys(history_headlines))

def save_digest_to_db(digest_content):
    """Saves a new digest and its articles to the database."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("INSERT INTO digests (created_at) VALUES (NOW()) RETURNING id")
        digest_id = cur.fetchone()[0]
        logging.info(f"Created new digest entry with ID: {digest_id}")
        
        display_order_counter = 0
        articles_to_insert = []
        for topic, articles in digest_content.items():
            for article in articles:
                pub_dt_utc = parsedate_to_datetime(article["pubDate"]).astimezone(ZoneInfo("UTC"))
                articles_to_insert.append((
                    digest_id,
                    topic,
                    article["title"],
                    article["link"],
                    pub_dt_utc,
                    display_order_counter
                ))
                display_order_counter += 1
        
        extras.execute_values(
            cur,
            "INSERT INTO articles (digest_id, topic, title, link, pub_date, display_order) VALUES %s",
            articles_to_insert
        )
        
        logging.info(f"Inserted {len(articles_to_insert)} articles into the database for digest ID {digest_id}.")

        max_history_digests = int(CONFIG.get("MAX_HISTORY_DIGESTS", 12))
        cur.execute("""
            DELETE FROM digests
            WHERE id IN (
                SELECT id FROM digests
                ORDER BY created_at DESC
                OFFSET %s
            )
        """, (max_history_digests,))
        deleted_count = cur.rowcount
        if deleted_count > 0:
            logging.info(f"Pruned {deleted_count} old digest(s) from the database.")
            
        conn.commit()
        cur.close()

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Failed to save digest to database: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def build_user_preferences(topics, keywords, overrides):
    preferences = []
    if topics:
        preferences.append("User topics (ranked 1-5 in importance, 5 is most important):")
        for topic, score in sorted(topics.items(), key=lambda x: -x[1]):
            preferences.append(f"- {topic}: {score}")
    if keywords:
        preferences.append("\nHeadline keywords (ranked 1-5 in importance, 5 is most important):")
        for keyword, score in sorted(keywords.items(), key=lambda x: -x[1]):
            preferences.append(f"- {keyword}: {score}")
    banned = [k for k, v in overrides.items() if v == "ban"]
    demoted = [k for k, v in overrides.items() if v == "demote"]
    if banned:
        preferences.append("\nBanned terms (must not appear in topics or headlines):")
        preferences.extend(f"- {term}" for term in banned)
    if demoted:
        preferences.append(f"\nDemoted terms (consider headlines with these terms {DEMOTE_FACTOR} times less important to the user, all else equal):")
        preferences.extend(f"- {term}" for term in demoted)
    return "\n".join(preferences)

def safe_parse_json(raw_json_string: str) -> dict:
    if not raw_json_string:
        logging.warning("safe_parse_json received empty string.")
        return {}
    text = raw_json_string.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        logging.warning("JSON string is empty after stripping wrappers.")
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON.loads failed: {e}. Attempting cleaning.")
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = re.sub(r",\s*([\]}])", r"\1", text)
        text = text.replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            parsed_data = ast.literal_eval(text)
            if isinstance(parsed_data, dict):
                return parsed_data
            else:
                logging.warning(f"ast.literal_eval parsed to non-dict type: {type(parsed_data)}. Raw: {text[:100]}")
                return {}
        except (ValueError, SyntaxError, TypeError) as e_ast:
            logging.warning(f"ast.literal_eval also failed: {e_ast}. Trying regex for quotes.")
            try:
                text = re.sub(r'(?<=([{,]\s*))([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', text)
                text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
                return json.loads(text)
            except json.JSONDecodeError as e2:
                logging.error(f"JSON.loads failed after all cleaning attempts: {e2}. Raw content (first 500 chars): {raw_json_string[:500]}")
                return {}

digest_tool_schema = {
    "type": "object",
    "properties": {
        "selected_digest_entries": {
            "type": "array",
            "description": (
                f"A list of selected news topics. Each entry in the list should be an object "
                f"containing a 'topic_name' (string) and 'headlines' (a list of strings). "
                f"Select up to {MAX_TOPICS} topics, and for each topic, select up to "
                f"{MAX_ARTICLES_PER_TOPIC} of the most relevant headlines."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "The name of the news topic (e.g., 'Technology', 'Climate Change')."
                    },
                    "headlines": {
                        "type": "array",
                        "items": {"type": "string", "description": "A selected headline string for this topic."},
                        "description": f"A list of up to {MAX_ARTICLES_PER_TOPIC} most important headline strings for this topic."
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
                f"Formats the selected news topics and headlines for the user's digest. "
                f"You must select up to {MAX_TOPICS} of the most important topics. "
                f"For each selected topic, return up to {MAX_ARTICLES_PER_TOPIC} most important headlines. "
                "The output should be structured as a list of objects, where each object contains a 'topic_name' "
                "and a list of 'headlines' corresponding to that topic."
            ),
            parameters=digest_tool_schema,
        )
    ]
)

def contains_banned_keyword(text, banned_terms):
    if not text: return False
    norm_text = normalize(text)
    return any(banned_term in norm_text for banned_term in banned_terms if banned_term)

def prioritize_with_gemini(
    headlines_to_send: dict,
    digest_history: list,
    gemini_api_key: str,
    topic_weights: dict,
    keyword_weights: dict,
    overrides: dict
) -> dict:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        tools=[SELECT_DIGEST_ARTICLES_TOOL]
    )

    digest_history_json = json.dumps(digest_history, indent=2)

    pref_data = {
        "topic_weights": topic_weights,
        "keyword_weights": keyword_weights,
        "banned_terms": [k for k, v in overrides.items() if v == "ban"],
        "demoted_terms": [k for k, v in overrides.items() if v == "demote"]
    }
    user_preferences_json = json.dumps(pref_data, indent=2)

    prompt = (
        "You are a News Curation AI. Your task is to select the best articles for a user's news digest based on a strict set of rules.\n\n"
        "### PRIMARY OBJECTIVE\n"
        "Produce a high-quality, non-redundant news digest by rigorously following the processing order below. "
        "Your final output MUST be a single call to the `format_digest_selection` tool. Do not provide any other text.\n\n"
        "### Inputs\n"
        f"1.  **User Preferences:** Defines topic weights and banned/demoted terms.\n```json\n{user_preferences_json}\n```\n"
        f"2.  **Candidate Headlines:** The pool of new articles to choose from.\n```json\n{json.dumps(dict(sorted(headlines_to_send.items())), indent=2)}\n```\n"
        f"3.  **Digest History:** Articles the user has already seen. DO NOT repeat these.\n```json\n{digest_history_json}\n```\n\n"
        "--- MANDATORY PROCESSING ORDER ---\n"
        "You MUST execute these steps in this exact order.\n\n"
        "**STEP 1: GLOBAL DE-DUPLICATION (MOST IMPORTANT FIRST STEP)**\n"
        "1.  Analyze ALL `Candidate Headlines` across ALL topics at once.\n"
        "2.  Identify groups of headlines that describe the SAME core news event. Example: 'Fed Holds Rates Steady' and 'Federal Reserve Pauses Hikes' are the same event.\n"
        "3.  From each group, select ONLY the single best, most informative headline.\n"
        "4.  IMMEDIATELY DISCARD all other redundant headlines from the candidate pool.\n"
        "5.  Proceed to the next step ONLY with this new, de-duplicated list of headlines.\n\n"
        "**STEP 2: HISTORY FILTERING**\n"
        "1.  Take your de-duplicated list from Step 1.\n"
        "2.  Compare each headline against the `Digest History`.\n"
        "3.  If a headline reports on an event already present in the history, DISCARD IT.\n"
        "4.  Only keep headlines that are genuinely new information for the user.\n\n"
        "**STEP 3: QUALITY & RELEVANCE FILTERING**\n"
        "Apply these rules strictly to the remaining headlines.\n\n"
        "**A. Content to REMOVE (Reject immediately):**\n"
        "-   Headlines containing any 'banned_terms' from User Preferences.\n"
        "-   Local news with no clear national or major international impact for a U.S. audience.\n"
        "-   Advertisements, sponsored content, or promotional articles.\n"
        "-   Investment advice, stock tips, or 'buy now' recommendations (e.g., \"Top 5 Stocks to Buy\"). Factual market reports (e.g., \"S&P 500 hits record high\") are OK.\n"
        "-   Sensationalist, clickbait, or emotionally manipulative headlines (e.g., using words like \"shocking,\" \"unbelievable,\" or withholding key info).\n"
        "-   Celebrity gossip, fluff, or low-value entertainment pieces.\n"
        "-   Opinion/Op-Ed pieces. Prioritize factual, reported news.\n"
        "-   Headlines phrased as questions or listicles (e.g., \"Is X the future?\", \"7 Reasons Why...\").\n\n"
        "**B. Content to STRONGLY DE-PRIORITIZE:**\n"
        "-   Headlines containing 'demoted_terms' from User Preferences. Treat their importance as nearly zero. Only include if the story is exceptionally significant despite the term.\n\n"
        "**STEP 4: FINAL SELECTION & CRITICAL SORTING FOR TOOL CALL**\n"
        "1.  From the final, fully-filtered pool of high-quality headlines, select your final choices, respecting the limits: "
        f"max **{MAX_TOPICS} topics** and max **{MAX_ARTICLES_PER_TOPIC} headlines** per topic.\n"
        "2.  **FINAL SORTING RULES (MANDATORY):**\n"
        "    - **Topic Order:** The final list of topics you return MUST be sorted by importance, NOT alphabetically. To do this, find the single most important headline overall; its topic is #1. Then, find the most important headline from the *remaining* topics; its topic is #2. Continue this process until all topics are ordered.\n"
        "    - ***CRITICAL WARNING: DO NOT SORT TOPICS ALPHABETICALLY.*** Sorting MUST follow the importance-based method described above.\n"
        "    - **Headline Order:** Within each topic, sort its headlines from most to least important.\n"
        "3.  Call the `format_digest_selection` tool with your final, importance-sorted data. This is your only output."
    )

    logging.info("Sending request to Gemini for prioritization with history.")
    
    try:
        response = model.generate_content(
            [prompt],
            tool_config={"function_calling_config": {"mode": "ANY", "allowed_function_names": ["format_digest_selection"]}}
        )

        finish_reason_display_str = "N/A"
        raw_finish_reason_value = None

        if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
            raw_finish_reason_value = response.candidates[0].finish_reason
            if hasattr(raw_finish_reason_value, 'name'):
                finish_reason_display_str = raw_finish_reason_value.name
            else:
                finish_reason_display_str = str(raw_finish_reason_value)

        has_tool_call = False
        function_call_part = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call_part = part.function_call
                    has_tool_call = True
                    finish_reason_display_str = "TOOL_CALLS"
                    break

        logging.info(f"Gemini response. finish_reason: {finish_reason_display_str}, raw_finish_reason_value: {raw_finish_reason_value}, has_tool_call: {has_tool_call}")

        if function_call_part:
            if function_call_part.name == "format_digest_selection":
                args = function_call_part.args
                logging.info(f"Gemini used tool 'format_digest_selection' with args (type: {type(args)}): {str(args)[:1000]}...")

                transformed_output = {}
                if isinstance(args, (MapComposite, dict)):
                    entries_list_proto = args.get("selected_digest_entries")
                    if isinstance(entries_list_proto, (RepeatedComposite, list)):
                        for entry_proto in entries_list_proto:
                            if isinstance(entry_proto, (MapComposite, dict)):
                                topic_name = entry_proto.get("topic_name")
                                headlines_proto = entry_proto.get("headlines")

                                headlines_python_list = []
                                if isinstance(headlines_proto, (RepeatedComposite, list)):
                                    for h_item in headlines_proto:
                                        headlines_python_list.append(str(h_item))

                                if isinstance(topic_name, str) and topic_name.strip() and headlines_python_list:
                                    transformed_output[topic_name.strip()] = headlines_python_list
                                else:
                                    logging.warning(f"Skipping invalid entry from tool: topic '{topic_name}', headlines '{headlines_python_list}'")
                            else:
                                logging.warning(f"Skipping non-dict/MapComposite item in 'selected_digest_entries' from tool: {type(entry_proto)}")
                    else:
                        logging.warning(f"'selected_digest_entries' from tool is not a list/RepeatedComposite or is missing. Type: {type(entries_list_proto)}")
                else:
                    logging.warning(f"Gemini tool call 'args' is not a MapComposite or dict. Type: {type(args)}")

                logging.info(f"Transformed output from Gemini tool call: {transformed_output}")
                return transformed_output
            else:
                logging.warning(f"Gemini called an unexpected tool: {function_call_part.name}")
                return {}
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            text_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            if text_content.strip():
                logging.warning("Gemini did not use the tool, returned text instead. Attempting to parse.")
                parsed_json_fallback = safe_parse_json(text_content)
                if isinstance(parsed_json_fallback, dict) and "selected_digest_entries" in parsed_json_fallback:
                    entries = parsed_json_fallback.get("selected_digest_entries", [])
                    if isinstance(entries, list):
                        transformed_output = {}
                        for item in entries:
                            if isinstance(item, dict):
                                topic = item.get("topic_name")
                                headlines = item.get("headlines")
                                if isinstance(topic, str) and isinstance(headlines, list):
                                    transformed_output[topic] = headlines
                        if transformed_output:
                             logging.info(f"Successfully parsed tool-like structure from Gemini text response: {transformed_output}")
                             return transformed_output
                logging.warning(f"Could not parse Gemini's text response into expected format. Raw text: {text_content[:500]}")
                return {}
            else:
                 logging.warning("Gemini returned no usable function call and no parsable text content.")
                 return {}
        else:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini response has prompt feedback: {response.prompt_feedback}")
            logging.warning(f"Gemini returned no candidates or no content parts.")
            return {}

    except Exception as e:
        logging.error(f"Error during Gemini API call or processing response: {e}", exc_info=True)
        return {}

def main():
    # Load user-facing history from the DATABASE.
    recent_digest_headlines = load_recent_headlines_from_db(MAX_HISTORY_DIGESTS)
    
    # Create a history object compatible with the pre-filter function
    history_for_prefilter = { "all_history": [{"title": h} for h in recent_digest_headlines] }

    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.error("Missing GEMINI_API_KEY environment variable. Exiting.")
            sys.exit(1)

        headlines_to_send_to_llm = {}
        full_articles_map_this_run = {}

        banned_terms_list = [k for k, v in OVERRIDES.items() if v == "ban"]
        normalized_banned_terms = [normalize(term) for term in banned_terms_list if term]

        articles_to_fetch_per_topic = int(CONFIG.get("ARTICLES_TO_FETCH_PER_TOPIC", 10))

        # --- HYBRID PRE-FILTERING STAGE ---
        for topic_name in TOPIC_WEIGHTS:
            fetched_topic_articles = fetch_articles_for_topic(topic_name, articles_to_fetch_per_topic)
            if fetched_topic_articles:
                current_topic_headlines_for_llm = []
                for art in fetched_topic_articles:
                    if is_high_confidence_duplicate_in_history(art["title"], history_for_prefilter, MATCH_THRESHOLD):
                        logging.debug(f"Skipping (high-confidence history match): {art['title']}")
                        continue

                    if contains_banned_keyword(art["title"], normalized_banned_terms):
                        logging.debug(f"Skipping (banned keyword): {art['title']}")
                        continue

                    current_topic_headlines_for_llm.append(art["title"])
                    norm_title_key = normalize(art["title"])
                    if norm_title_key not in full_articles_map_this_run:
                         full_articles_map_this_run[norm_title_key] = art

                if current_topic_headlines_for_llm:
                    headlines_to_send_to_llm[topic_name] = current_topic_headlines_for_llm

        gemini_processed_content = {}

        if not headlines_to_send_to_llm:
            logging.info("No new, non-banned, non-duplicate headlines available to send to LLM.")
        else:
            total_headlines_count = sum(len(v) for v in headlines_to_send_to_llm.values())
            logging.info(f"Sending {total_headlines_count} candidate headlines across {len(headlines_to_send_to_llm)} topics to Gemini.")

            selected_content_raw_from_llm = prioritize_with_gemini(
                headlines_to_send=headlines_to_send_to_llm,
                digest_history=recent_digest_headlines,
                gemini_api_key=gemini_api_key,
                topic_weights=TOPIC_WEIGHTS,
                keyword_weights=KEYWORD_WEIGHTS,
                overrides=OVERRIDES
            )

            if not selected_content_raw_from_llm or not isinstance(selected_content_raw_from_llm, dict):
                logging.warning("Gemini returned no content or invalid format.")
            else:
                logging.info(f"Processing {len(selected_content_raw_from_llm)} topics returned by Gemini.")
                seen_normalized_titles_in_llm_output = set()

                for topic_from_llm, titles_from_llm in selected_content_raw_from_llm.items():
                    if not isinstance(titles_from_llm, list):
                        logging.warning(f"LLM returned non-list for topic '{topic_from_llm}'. Skipping.")
                        continue

                    current_topic_articles_for_digest = []
                    for title_from_llm in titles_from_llm[:MAX_ARTICLES_PER_TOPIC]:
                        if not isinstance(title_from_llm, str):
                            logging.warning(f"LLM returned non-string headline: {title_from_llm}. Skipping.")
                            continue

                        norm_llm_title = normalize(title_from_llm)
                        if not norm_llm_title or norm_llm_title in seen_normalized_titles_in_llm_output:
                            continue

                        article_data = full_articles_map_this_run.get(norm_llm_title)
                        if article_data:
                            current_topic_articles_for_digest.append(article_data)
                            seen_normalized_titles_in_llm_output.add(norm_llm_title)
                        else:
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

        final_digest_for_db = gemini_processed_content

        if final_digest_for_db:
            logging.info(f"Final digest contains {len(final_digest_for_db)} topics. Saving to database.")
            save_digest_to_db(final_digest_for_db)
        else:
            logging.info("No topics from Gemini this run. Database will not be updated.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
    finally:
        logging.info(f"Worker script finished at {datetime.now(ZONE)}")
        
if __name__ == "__main__":
    main()