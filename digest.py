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
GEMINI_MODEL_NAME = CONFIG.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")
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
    stemmed = [stemmer.stem(w) for w in words]
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]
    return " ".join(lemmatized)

def is_high_confidence_duplicate_in_history(article_title: str, history: dict, threshold: float) -> bool:
    """
    Checks if a given article title is a high-confidence (near-identical) duplicate
    of any article title already in the history log, based on word overlap.
    """
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
        # logging.info(f"Fetched {len(articles)} articles for topic '{topic}'")
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

    # Manifest is sorted newest to oldest, so we take the top N
    digests_to_load = manifest[:max_digests]
    logging.info(f"Loading history from the {len(digests_to_load)} most recent digests.")

    for entry in digests_to_load:
        # Construct the full path to the historical digest file
        digest_file_path = os.path.join(os.path.dirname(manifest_file), entry["file"])
        if os.path.exists(digest_file_path):
            try:
                with open(digest_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Simple but effective regex to find all headlines within the <a> tags
                    found_headlines = re.findall(r'<a href="[^"]+" target="_blank">([^<]+)</a>', content)
                    history_headlines.extend(html.unescape(h) for h in found_headlines) # Unescape HTML entities
            except IOError as e:
                logging.warning(f"Could not read historical digest file {digest_file_path}: {e}")

    unique_history = list(dict.fromkeys(history_headlines))
    logging.info(f"Loaded {len(unique_history)} unique headlines from recent digest history.")
    return unique_history

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

    # Build the preferences JSON from the arguments passed into the function.
    # This makes the function self-contained and removes global dependencies.
    pref_data = {
        "topic_weights": topic_weights,
        "keyword_weights": keyword_weights,
        "banned_terms": [k for k, v in overrides.items() if v == "ban"],
        "demoted_terms": [k for k, v in overrides.items() if v == "demote"]
    }
    user_preferences_json = json.dumps(pref_data, indent=2)

    # The prompt is now fully self-contained and uses the data passed in.
    prompt = (
        "You are an Advanced News Synthesis Engine. Your function is to act as an expert, hyper-critical news curator. Your single most important mission is to produce a high-signal, non-redundant, and deeply relevant news digest for a user. You must be ruthless in eliminating noise, repetition, and low-quality content.\n\n"
        "### Inputs Provided\n"
        f"1.  **User Preferences:** A JSON object defining topic interests, importance weights (1-5), and banned/demoted keywords.\n```json\n{user_preferences_json}\n```\n"
        f"2.  **Candidate Headlines:** A pool of new articles available for today's digest, organized by their machine-assigned topic.\n```json\n{json.dumps(dict(sorted(headlines_to_send.items())), indent=2)}\n```\n"
        f"3.  **Digest History:** A list of headlines the user has already seen in recent digests. You MUST NOT select headlines that are substantively identical to these.\n```json\n{digest_history_json}\n```\n\n"
        "### Core Processing Pipeline (Follow these steps sequentially)\n\n"
        "**Step 1: Cross-Topic Semantic Clustering & Deduplication (CRITICAL FIRST STEP)**\n"
        "First, analyze ALL `Candidate Headlines`. Your primary task is to identify and group all headlines from ALL topics that cover the same core news event. An 'event' is the underlying real-world occurrence, not the specific wording of a headline.\n"
        "-   **Group by Meaning:** Cluster headlines based on their substantive meaning. For example, 'Fed Pauses Rate Hikes,' 'Federal Reserve Holds Interest Rates Steady,' and 'Powell Announces No Change to Fed Funds Rate' all belong to the same cluster.\n"
        "-   **Select One Champion:** From each cluster, select ONLY ONE headline—the one that is the most comprehensive, recent, objective, and authoritative. Discard all other headlines in that cluster immediately.\n\n"
        "**Step 2: History-Based Filtering**\n"
        "Now, take your deduplicated list of 'champion' headlines. Compare each one against the `Digest History`. If any of your champion headlines reports on the exact same event that has already been sent, DISCARD it. Only select news that provides a significant, new update.\n\n"
        "**Step 3: Rigorous Relevance & Quality Filtering**\n"
        "For the remaining, unique, and new headlines, apply the following strict filtering criteria with full force:\n\n"
        f"*   **Output Limits:** Adhere strictly to a maximum of **{MAX_TOPICS} topics** and **{MAX_ARTICLES_PER_TOPIC} headlines** per topic.\n"
        "*   **Geographic Focus:**\n"
        "    - Focus on national (U.S.) or major international news.\n"
        "    - AVOID news that is *solely* of local interest (e.g., specific to a small town) *unless* it has clear and direct national or major international implications relevant to a U.S. audience.\n"
        "*   **Banned/Demoted Content:**\n"
        "    - Strictly REJECT any headlines containing terms flagged as 'banned' in user preferences.\n"
        "    - Headlines with 'demote' terms should be *strongly deprioritized* (effectively treated as having an importance score of 0.1 on a 1-5 scale) and only selected if their relevance is exceptionally high.\n"
        "*   **Commercial Content (APPLY WITH EXTREME PREJUDICE):**\n"
        "    - REJECT advertisements, sponsored content, and articles that are primarily promotional.\n"
        "    - REJECT mentions of specific products/services UNLESS it's highly newsworthy criticism, a major market-moving announcement (e.g., a massive product recall by a major company), or a significant technological breakthrough discussed in a news context, not a promotional one.\n"
        "    - STRICTLY REJECT articles that primarily offer investment advice, promote specific stocks/cryptocurrencies as 'buy now' opportunities, or resemble 'hot stock tips' (e.g., \"Top X Stocks to Invest In,\" \"This Coin Will Explode,\" \"X Stocks Worth Buying\"). News about broad market trends (e.g., \"S&P 500 reaches record high\"), factual company earnings reports (without buy/sell advice), or major regulatory changes IS acceptable. The key is to distinguish objective financial news from investment solicitation.\n"
        "*   **Content Quality & Style (CRITICAL):**\n"
        "    - Ensure a healthy diversity of subjects if possible; do not let one single event dominate the entire digest.\n"
        "    - PRIORITIZE content-rich, factual, objective, and neutrally-toned reporting.\n"
        "    - AGGRESSIVELY AVOID AND REJECT headlines that are:\n"
        "        - Sensationalist, using hyperbole, excessive superlatives (e.g., \"terrifying,\" \"decimated,\" \"gross failure\"), or fear-mongering.\n"
        "        - Purely for entertainment, celebrity gossip, or \"fluff\" pieces lacking substantial news value.\n"
        "        - Clickbait (e.g., withholding key information, using vague teasers like \"You won't believe what happened next!\").\n"
        "        - Primarily opinion/op-ed pieces, especially those with inflammatory or biased language. Focus on reported news.\n"
        "        - Phrased as questions (e.g., \"Is X the new Y?\") or promoting listicles (e.g., \"5 reasons why...\").\n\n"
        "**Step 4: Final Selection and Ordering**\n"
        "From the fully filtered and vetted pool of headlines, make your final selection.\n"
        "1.  **Topic Ordering:** Order the selected topics from most to least significant. Significance is a blend of the user's preference weight and the objective importance of the news you've selected for that topic.\n"
        "2.  **Headline Ordering:** Within each topic, order the selected headlines from most to least newsworthy/comprehensive.\n\n"
        "### Final Output\n"
        "Before calling the tool, perform a final mental check. Ask yourself:\n"
        "- \"Is this headline truly distinct from everything else, including the history?\"\n"
        "- \"Is this trying to sell me a stock, a product, or is it just reporting market news?\"\n"
        "- \"Is this headline objective, or is it heavily opinionated/sensationalist clickbait?\"\n"
        "- \"Is my final topic and headline ordering logical and based on true significance?\"\n\n"
        "Based on this rigorous process, provide your final, curated selection using the 'format_digest_selection' tool."
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
                # The returned value from Gemini is a dict of {topic: [headlines]}
                # Let's check if the parsed content matches this structure.
                if isinstance(parsed_json_fallback, dict) and "selected_digest_entries" in parsed_json_fallback:
                    # It seems the model might return JSON matching the tool structure, even in text.
                    # We should handle this gracefully.
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

        # --- MODIFIED GIT ADD LOGIC ---
        files_for_git_add = []
        history_file_abs = os.path.join(base_dir, "history.json")
        digest_state_file_abs = os.path.join(base_dir, "content.json")
        digests_dir_abs = os.path.join(base_dir, "public/digests")
        digest_manifest_abs = os.path.join(base_dir, "public/digest-manifest.json")
        latest_digest_abs = os.path.join(base_dir, "public/digest.html")


        if os.path.exists(history_file_abs): files_for_git_add.append(os.path.relpath(history_file_abs, base_dir))
        if os.path.exists(digest_state_file_abs): files_for_git_add.append(os.path.relpath(digest_state_file_abs, base_dir))
        if os.path.exists(digests_dir_abs): files_for_git_add.append(os.path.relpath(digests_dir_abs, base_dir))
        if os.path.exists(digest_manifest_abs): files_for_git_add.append(os.path.relpath(digest_manifest_abs, base_dir))
        if os.path.exists(latest_digest_abs): files_for_git_add.append(os.path.relpath(latest_digest_abs, base_dir))

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
        # --- END MODIFIED GIT ADD LOGIC ---

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
    # Load the full history log. This is used for the high-confidence pre-filter
    # and for the final history update.
    history = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            logging.warning("history.json is empty or invalid. Starting with an empty history log.")
        except Exception as e:
            logging.error(f"Error loading history.json: {e}. Starting with empty history log.")

    # Load the user-facing history from recent HTML digests. This is the context passed to the LLM.
    recent_digest_headlines = load_recent_digest_history(DIGESTS_DIR, DIGEST_MANIFEST_FILE, MAX_HISTORY_DIGESTS)

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
                    if is_high_confidence_duplicate_in_history(art["title"], history, MATCH_THRESHOLD):
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

        final_digest_for_display_and_state = gemini_processed_content

        if final_digest_for_display_and_state:
            logging.info(f"Final digest contains {len(final_digest_for_display_and_state)} topics, ordered by Gemini.")
            
            # ** CORE PERFORMANCE FIX LOGIC **
            # 1. Generate the latest digest HTML content string
            new_digest_html = generate_digest_html_content(final_digest_for_display_and_state, ZONE)

            # 2. Write that content to your established `digest.html` file
            with open(LATEST_DIGEST_HTML_FILE, "w", encoding="utf-8") as f:
                f.write(new_digest_html)
            logging.info(f"Latest digest content written to {LATEST_DIGEST_HTML_FILE}")

            # 3. Inject the same HTML content into the template to create the final index.html
            try:
                template_path = os.path.join(BASE_DIR, "public", "index.template.html")
                final_index_path = os.path.join(BASE_DIR, "public", "index.html")

                with open(template_path, "r", encoding="utf-8") as f:
                    index_content = f.read()

                # Replace the placeholder with the actual latest digest HTML
                index_content = index_content.replace("<!--LATEST_DIGEST_CONTENT_PLACEHOLDER-->", new_digest_html)

                with open(final_index_path, "w", encoding="utf-8") as f:
                    f.write(index_content)
                logging.info(f"Generated final index.html with embedded digest at {final_index_path}")

            except Exception as e:
                logging.error(f"Failed to embed latest digest into index.html from template: {e}")
            # ** END CORE PERFORMANCE FIX LOGIC **

            # Update historical digests and manifest
            update_digest_history(new_digest_html, ZONE)

            # Update state/history files
            try:
                content_json_to_save = {}
                now_utc_iso = datetime.now(ZoneInfo("UTC")).isoformat()
                for topic, articles in final_digest_for_display_and_state.items():
                    content_json_to_save[topic] = { "articles": articles, "last_updated_ts": now_utc_iso }

                with open(DIGEST_STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(content_json_to_save, f, indent=2)
                logging.info(f"Snapshot of current digest saved to {DIGEST_STATE_FILE}")
            except IOError as e:
                logging.error(f"Failed to write digest state file {DIGEST_STATE_FILE}: {e}")

        else:
            logging.info("No topics from Gemini this run. Files are not modified.")

        update_history_file(final_digest_for_display_and_state, history, HISTORY_FILE, ZONE)

        if CONFIG.get("ENABLE_GIT_PUSH", False):
            # The git operation will automatically pick up the new index.html
            perform_git_operations(BASE_DIR, ZONE, CONFIG)
        else:
            logging.info("Git push is disabled in config. Skipping.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
    finally:
        logging.info(f"Script finished at {datetime.now(ZONE)}")
        
if __name__ == "__main__":
    main()