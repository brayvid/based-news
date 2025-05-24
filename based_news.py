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
DIGEST_STATE_FILE = os.path.join(BASE_DIR, "current_digest_content.json") # New persistent state file

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
log_path = os.path.join(BASE_DIR, "logs/based_news.log")
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
MAX_TOPICS = int(CONFIG.get("MAX_TOPICS", 7))
MAX_ARTICLES_PER_TOPIC = int(CONFIG.get("MAX_ARTICLES_PER_TOPIC", 1))
DEMOTE_FACTOR = float(CONFIG.get("DEMOTE_FACTOR", 0.5))
MATCH_THRESHOLD = float(CONFIG.get("DEDUPLICATION_MATCH_THRESHOLD", 0.4))
GEMINI_MODEL_NAME = CONFIG.get("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") 

STALE_TOPIC_THRESHOLD_HOURS = int(CONFIG.get("STALE_TOPIC_THRESHOLD_HOURS", 72))

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

def is_in_history(article_title, history):
    norm_title_tokens = set(normalize(article_title).split())
    if not norm_title_tokens: return False

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
            if similarity >= MATCH_THRESHOLD:
                logging.debug(f"Article '{article_title}' matched past article '{past_title}' with similarity {similarity:.2f}")
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
        logging.info(f"Fetched {len(articles)} articles for topic '{topic}'")
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


def prioritize_with_gemini(headlines_to_send: dict, user_preferences: str, gemini_api_key: str) -> dict:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        tools=[SELECT_DIGEST_ARTICLES_TOOL]
    )
    prompt = (
        "You are an expert news curator. Your task is to choose the most relevant news topics and headlines "
        "for a user's email digest based on their specific preferences and a list of available articles.\n\n"
        f"User Preferences:\n{user_preferences}\n\n"
        f"Available Topics and Headlines (candidate articles):\n{json.dumps(dict(sorted(headlines_to_send.items())), indent=2)}\n\n"
        "Selection Criteria:\n"
        f"- Select up to {MAX_TOPICS} of the most important topics for the digest.\n"
        f"- For each selected topic, choose the top {MAX_ARTICLES_PER_TOPIC} most important headlines.\n"
        "- Strictly avoid returning multiple copies of the same or very similar headlines, even if they appear under different candidate topics. Deduplicate aggressively.\n"
        "- Avoid all local news (e.g., headlines specific to or referencing a small town or county). Focus on national (U.S.) or major international news relevant to a U.S. audience.\n"
        "- Strictly adhere to the user's topic and keyword importance preferences (1=lowest, 5=highest).\n"
        f"- Reject any headlines containing terms flagged 'banned'. Demote headlines with 'demote' terms (treat them as {DEMOTE_FACTOR} times less important).\n"
        "- Reject advertisements and mentions of specific products/services unless it's newsworthy criticism or a major announcement.\n"
        "- Ensure a healthy diversity of subjects. Do not over-focus on a single theme.\n"
        "- Prioritize headlines that are content-rich, factual, objective, and informative.\n"
        "- Actively avoid and deprioritize headlines that are:\n"
        "    - Sensationalist or designed for shock value (e.g., using excessive superlatives, fear-mongering).\n"
        "    - Purely for entertainment, celebrity gossip (unless of major national/international significance), or \"fluff\" pieces lacking substantial news value.\n"
        "    - Clickbait (e.g., withholding key information, using vague teasers).\n"
        "    - Phrased as questions or promoting listicles, unless the underlying content is exceptionally newsworthy.\n"
        "- Ensure the selected articles reflect genuine newsworthiness and are relevant to an informed general audience seeking serious news updates.\n\n"
        "Based on all the above, provide your selections using the 'format_digest_selection' tool."
    )
    logging.info("Sending request to Gemini for prioritization.")
    try:
        response = model.generate_content(
            prompt,
            tool_config={"function_calling_config": {"mode": "ANY", "allowed_function_names": ["format_digest_selection"]}}
        )
        
        finish_reason_str = "N/A"
        if response.candidates:
            try:
                finish_reason_val = response.candidates[0].finish_reason
                if response.candidates[0].content and response.candidates[0].content.parts and any(p.function_call for p in response.candidates[0].content.parts):
                    finish_reason_str = "TOOL_CALLS" 
                elif isinstance(finish_reason_val, int):
                    reason_map = {0: "FINISH_REASON_UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                    finish_reason_str = reason_map.get(finish_reason_val, f"UNKNOWN_REASON_{finish_reason_val}")
                else: 
                    finish_reason_str = str(finish_reason_val)
            except Exception as e_fr:
                logging.warning(f"Could not determine finish_reason string: {e_fr}")
        
        logging.info(f"Gemini response received. Finish reason: {finish_reason_str}")

        function_call_part = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call_part = part.function_call
                    break
        
        if function_call_part:
            if function_call_part.name == "format_digest_selection":
                args = function_call_part.args 
                logging.info(f"Gemini used tool 'format_digest_selection' with args (type: {type(args)}): {args}")
                
                if isinstance(args, MapComposite):
                    entries_list_proto = args.get("selected_digest_entries")
                elif isinstance(args, dict):
                    entries_list_proto = args.get("selected_digest_entries")
                else:
                    entries_list_proto = None

                if entries_list_proto is None or not (isinstance(entries_list_proto, list) or isinstance(entries_list_proto, RepeatedComposite)):
                    logging.warning(f"'selected_digest_entries' from Gemini is not a list/RepeatedComposite or is missing. Type: {type(entries_list_proto)}, Value: {entries_list_proto}")
                    return {}

                transformed_output = {}
                for entry_proto in entries_list_proto: 
                    if isinstance(entry_proto, (dict, MapComposite)):
                        topic_name = entry_proto.get("topic_name")
                        headlines_proto = entry_proto.get("headlines")

                        headlines_python_list = []
                        if isinstance(headlines_proto, (list, RepeatedComposite)):
                            headlines_python_list = [str(h) for h in headlines_proto if isinstance(h, (str, bytes))]
                        elif headlines_proto is not None: 
                            logging.warning(f"Headlines for topic '{topic_name}' is not a list/RepeatedComposite. Type: {type(headlines_proto)}")

                        if isinstance(topic_name, str) and topic_name.strip() and headlines_python_list:
                            topic_name_clean = topic_name.strip()
                            if topic_name_clean in transformed_output:
                                transformed_output[topic_name_clean].extend(headlines_python_list)
                            else:
                                transformed_output[topic_name_clean] = headlines_python_list
                            transformed_output[topic_name_clean] = list(dict.fromkeys(transformed_output[topic_name_clean]))
                        else:
                            logging.warning(f"Skipping invalid entry: topic '{topic_name}' (type {type(topic_name)}), headlines '{headlines_python_list}'")
                    else:
                        logging.warning(f"Skipping non-dict/MapComposite item in 'selected_digest_entries': type {type(entry_proto)}, value {entry_proto}")
                
                logging.info(f"Transformed output from Gemini tool call: {transformed_output}")
                return transformed_output
            else:
                logging.warning(f"Gemini called an unexpected tool: {function_call_part.name}")
                return {}
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            text_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            if text_content.strip():
                logging.warning("Gemini did not use the tool, returned text instead. Attempting to parse.")
                logging.debug(f"Gemini raw text response: {text_content}")
                parsed_json = safe_parse_json(text_content)
                
                if "selected_digest_entries" in parsed_json and isinstance(parsed_json["selected_digest_entries"], list):
                    transformed_output = {}
                    for entry in parsed_json["selected_digest_entries"]:
                        if isinstance(entry, dict):
                            topic_name = entry.get("topic_name")
                            headlines_list = entry.get("headlines")
                            if isinstance(topic_name, str) and topic_name.strip() and isinstance(headlines_list, list):
                                valid_headlines = [h for h in headlines_list if isinstance(h, str)]
                                if valid_headlines: 
                                    topic_name_clean = topic_name.strip()
                                    if topic_name_clean in transformed_output:
                                        transformed_output[topic_name_clean].extend(valid_headlines)
                                    else:
                                        transformed_output[topic_name_clean] = valid_headlines
                                    transformed_output[topic_name_clean] = list(dict.fromkeys(transformed_output[topic_name_clean]))
                            else:
                                logging.warning(f"Skipping invalid entry in parsed text JSON: {entry}")
                        else:
                             logging.warning(f"Skipping non-dict item in parsed text 'selected_digest_entries': {entry}")
                    if transformed_output:
                        logging.info(f"Successfully parsed and transformed text response from Gemini: {transformed_output}")
                        return transformed_output
                    else:
                        logging.warning(f"Parsed text response from Gemini did not yield usable digest entries.")
                        return {}
                else:
                    logging.warning(f"Gemini text response could not be parsed into the expected digest structure. Raw text: {text_content[:500]}")
                    return {}
            else:
                logging.warning("Gemini returned no usable function call and no parsable text content.")
                return {}
        else: 
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini response has prompt feedback: {response.prompt_feedback}")
            else:
                logging.warning(f"Gemini returned no candidates or no content parts. Full response (if available): {response}")
            return {}

    except Exception as e:
        logging.error(f"Error during Gemini API call or processing response: {e}", exc_info=True)
        try:
            if 'response' in locals() and response:
                logging.error(f"Gemini response object on error (prompt_feedback): {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                if hasattr(response, 'candidates') and response.candidates:
                     logging.error(f"Gemini response object on error (first candidate): {response.candidates[0]}")
        except Exception as e_log:
            logging.error(f"Error logging response details during exception: {e_log}")
        return {}


def write_digest_html(digest_data, base_dir, current_zone):
    digest_path = os.path.join(base_dir, "public", "digest.html")
    os.makedirs(os.path.dirname(digest_path), exist_ok=True)

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
    
    last_updated_dt = datetime.now(current_zone)
    last_updated_str_for_footer = last_updated_dt.strftime("%A, %d %B %Y %I:%M %p %Z")
    
    footer_html = (
        f"<div class='timestamp' id='last-updated' style='display: none;'>" 
        f"Last updated: {last_updated_str_for_footer}"
        f"</div>\n"
    )
    html_parts.append(footer_html)

    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))


def update_history_file(newly_selected_articles_by_topic, current_history, history_file_path, current_zone):
    if not newly_selected_articles_by_topic or not isinstance(newly_selected_articles_by_topic, dict):
        logging.info("No newly selected articles provided to update_history_file, or invalid format.")
        # Still proceed to prune existing history below

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

def perform_git_operations(base_dir):
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        github_repository_owner_slash_repo = os.getenv("GITHUB_REPOSITORY")
        
        if not github_token or not github_repository_owner_slash_repo:
            logging.error("GITHUB_TOKEN or GITHUB_REPOSITORY not set. Cannot push to GitHub.")
            return
        
        remote_url = f"https://oauth2:{github_token}@github.com/{github_repository_owner_slash_repo}.git"
        
        commit_author_name = os.getenv("GITHUB_USER", CONFIG.get("GIT_USER_NAME", "Automated Digest Bot"))
        commit_author_email = os.getenv("GITHUB_EMAIL", CONFIG.get("GIT_USER_EMAIL", "bot@example.com"))

        logging.info(f"Using Git Commit Author Name: '{commit_author_name}', Email: '{commit_author_email}'")

        subprocess.run(["git", "config", "user.name", commit_author_name], check=True, cwd=base_dir)
        subprocess.run(["git", "config", "user.email", commit_author_email], check=True, cwd=base_dir)

        try:
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, cwd=base_dir, capture_output=True)
        except subprocess.CalledProcessError:
            logging.info("Failed to set-url (maybe remote 'origin' doesn't exist). Attempting to add.")
            subprocess.run(["git", "remote", "add", "origin", remote_url], check=True, cwd=base_dir, capture_output=True)
        
        paths_to_add_relative = []
        if os.path.exists(HISTORY_FILE):
            paths_to_add_relative.append(os.path.relpath(HISTORY_FILE, base_dir))
        
        digest_html_path_abs = os.path.join(base_dir, "public/digest.html")
        # Only add digest.html if it exists (it might not if no content was ever generated)
        if os.path.exists(digest_html_path_abs):
             paths_to_add_relative.append(os.path.relpath(digest_html_path_abs, base_dir))

        if os.path.exists(DIGEST_STATE_FILE): 
            paths_to_add_relative.append(os.path.relpath(DIGEST_STATE_FILE, base_dir))


        if not paths_to_add_relative:
            logging.info("No standard files (history.json, digest.html, current_digest_content.json) found to add to git.")
        else:
            logging.info(f"Attempting to git add: {paths_to_add_relative}")
            add_process = subprocess.run(["git", "add"] + paths_to_add_relative, 
                                         capture_output=True, text=True, cwd=base_dir) 
            if add_process.returncode != 0:
                logging.error(f"git add command failed. RC: {add_process.returncode}, Stdout: {add_process.stdout}, Stderr: {add_process.stderr}")
            else:
                logging.info(f"git add successful for specified paths.")
        
        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=base_dir)
        if not status_result.stdout.strip():
            logging.info("No changes staged for commit. Nothing to commit.")
            return
        
        branch_result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True, cwd=base_dir)
        current_branch = branch_result.stdout.strip()

        commit_message = f"Auto-update digest content - {datetime.now(ZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=base_dir, capture_output=True)
        
        logging.info(f"Pushing changes to origin/{current_branch}...")
        subprocess.run(["git", "push", "origin", current_branch], check=True, cwd=base_dir, capture_output=True)
        logging.info(f"Content committed and pushed to GitHub on branch '{current_branch}'.")

    except subprocess.CalledProcessError as e:
        output_str = e.output.decode(errors='ignore') if e.output else ""
        stderr_str = e.stderr.decode(errors='ignore') if e.stderr else ""
        logging.error(f"Git operation failed: {e}. Command: '{e.cmd}'. Output: {output_str}. Stderr: {stderr_str}")
    except Exception as e:
        logging.error(f"General error during Git operations: {e}", exc_info=True)

def main():
    history = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            logging.warning("history.json is empty or invalid. Starting with an empty history.")
        except Exception as e:
            logging.error(f"Error loading history.json: {e}. Starting with empty history.")

    persisted_digest_state = {}
    if os.path.exists(DIGEST_STATE_FILE):
        try:
            with open(DIGEST_STATE_FILE, "r", encoding="utf-8") as f:
                persisted_digest_state = json.load(f)
            if not isinstance(persisted_digest_state, dict): 
                logging.warning(f"{DIGEST_STATE_FILE} does not contain a valid JSON object. Resetting.")
                persisted_digest_state = {}
            else: 
                for topic_name, topic_data in list(persisted_digest_state.items()): 
                    if not (isinstance(topic_data, dict) and \
                            "articles" in topic_data and isinstance(topic_data["articles"], list) and \
                            "last_updated_ts" in topic_data and isinstance(topic_data["last_updated_ts"], str)):
                        logging.warning(f"Invalid structure for topic '{topic_name}' in {DIGEST_STATE_FILE}. Removing it.")
                        del persisted_digest_state[topic_name]
        except json.JSONDecodeError:
            logging.warning(f"{DIGEST_STATE_FILE} is invalid JSON. Starting with an empty digest state.")
        except Exception as e:
            logging.error(f"Error loading {DIGEST_STATE_FILE}: {e}. Starting with empty digest state.")

    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logging.error("Missing GEMINI_API_KEY environment variable. Exiting.")
            sys.exit(1)

        user_preferences = build_user_preferences(TOPIC_WEIGHTS, KEYWORD_WEIGHTS, OVERRIDES)
        
        headlines_to_send_to_llm = {} 
        full_articles_map_this_run = {} 
        
        banned_terms_list = [k for k, v in OVERRIDES.items() if v == "ban"]
        normalized_banned_terms = [normalize(term) for term in banned_terms_list if term] 

        for topic_name in TOPIC_WEIGHTS:
            fetched_topic_articles = fetch_articles_for_topic(topic_name, 15) 
            if fetched_topic_articles:
                current_topic_headlines_for_llm = []
                for art in fetched_topic_articles:
                    if is_in_history(art["title"], history):
                        logging.debug(f"Skipping (in history): {art['title']}")
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
        
        final_digest_from_llm_this_run = {} 

        if not headlines_to_send_to_llm:
            logging.info("No new, non-banned, non-historical headlines available to send to LLM.")
        else:
            total_headlines_count = sum(len(v) for v in headlines_to_send_to_llm.values())
            logging.info(f"Sending {total_headlines_count} candidate headlines across {len(headlines_to_send_to_llm)} topics to Gemini.")
            
            selected_content_raw_from_llm = prioritize_with_gemini(headlines_to_send_to_llm, user_preferences, gemini_api_key)

            if not selected_content_raw_from_llm or not isinstance(selected_content_raw_from_llm, dict):
                logging.warning("Gemini returned no content or invalid format. No new topics from LLM this run.")
            else:
                seen_normalized_titles_in_llm_output = set()
                for topic_from_llm, titles_from_llm in selected_content_raw_from_llm.items():
                    if not isinstance(titles_from_llm, list):
                        logging.warning(f"LLM returned non-list for topic '{topic_from_llm}': {titles_from_llm}. Skipping.")
                        continue
                    
                    current_topic_articles_for_digest = []
                    for title_from_llm in titles_from_llm:
                        if not isinstance(title_from_llm, str):
                            logging.warning(f"LLM returned non-string headline: {title_from_llm} for topic '{topic_from_llm}'. Skipping.")
                            continue
                        
                        norm_llm_title = normalize(title_from_llm)
                        if not norm_llm_title: continue

                        if norm_llm_title in seen_normalized_titles_in_llm_output:
                            logging.info(f"Deduplicating LLM output: '{title_from_llm}' already selected under another topic by LLM.")
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
                                        logging.info(f"Matched LLM title '{title_from_llm}' to stored '{stored_article_data['title']}' via fallback.")
                                        found_fallback = True
                                        break
                            if not found_fallback:
                                logging.warning(f"Could not map LLM title '{title_from_llm}' (normalized: '{norm_llm_title}') back to a fetched article.")
                    
                    if current_topic_articles_for_digest:
                        final_digest_from_llm_this_run[topic_from_llm] = current_topic_articles_for_digest
        
        now_utc = datetime.now(ZoneInfo("UTC"))
        now_utc_iso = now_utc.isoformat()

        if final_digest_from_llm_this_run: 
            for topic_name, articles in final_digest_from_llm_this_run.items():
                persisted_digest_state[topic_name] = {
                    "articles": articles,
                    "last_updated_ts": now_utc_iso
                }
            logging.info(f"Merged {len(final_digest_from_llm_this_run)} topics from LLM into digest state.")
        else:
            logging.info("No new topics/articles from LLM to merge into digest state this run.")

        topics_to_remove_due_to_staleness = []
        stale_threshold_datetime_utc = now_utc - timedelta(hours=STALE_TOPIC_THRESHOLD_HOURS)

        for topic_name, topic_data in persisted_digest_state.items():
            if final_digest_from_llm_this_run and topic_name in final_digest_from_llm_this_run:
                continue 

            last_updated_ts_str = topic_data.get("last_updated_ts")
            if not isinstance(last_updated_ts_str, str):
                logging.warning(f"Topic '{topic_name}' in persisted state has missing/invalid 'last_updated_ts'. Marking stale.")
                topics_to_remove_due_to_staleness.append(topic_name)
                continue
            
            try:
                last_updated_dt_utc = datetime.fromisoformat(last_updated_ts_str.replace('Z', '+00:00'))
                if last_updated_dt_utc.tzinfo is None: 
                    last_updated_dt_utc = last_updated_dt_utc.replace(tzinfo=ZoneInfo("UTC"))
                
                if last_updated_dt_utc < stale_threshold_datetime_utc:
                    topics_to_remove_due_to_staleness.append(topic_name)
                    logging.info(f"Topic '{topic_name}' (last updated: {last_updated_ts_str}) is stale. Queued for removal.")
            except ValueError:
                logging.warning(f"Could not parse 'last_updated_ts' ('{last_updated_ts_str}') for topic '{topic_name}'. Marking stale.")
                topics_to_remove_due_to_staleness.append(topic_name)

        for topic_name in topics_to_remove_due_to_staleness:
            if topic_name in persisted_digest_state:
                del persisted_digest_state[topic_name]

        display_candidates = []
        for topic_name, topic_data in persisted_digest_state.items():
            ts_str = topic_data.get("last_updated_ts")
            articles_list = topic_data.get("articles")
            if not isinstance(articles_list, list): 
                logging.warning(f"Articles for topic '{topic_name}' is not a list in persisted state. Skipping for display.")
                continue

            if isinstance(ts_str, str):
                try:
                    dt_utc = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if dt_utc.tzinfo is None: dt_utc = dt_utc.replace(tzinfo=ZoneInfo("UTC"))
                    display_candidates.append({"name": topic_name, "articles": articles_list, "timestamp_utc": dt_utc})
                except ValueError: 
                    logging.warning(f"Topic '{topic_name}' has unparsable 'last_updated_ts' ('{ts_str}') during display prep. Skipping.")
            else: 
                logging.warning(f"Topic '{topic_name}' missing 'last_updated_ts' during display prep. Skipping.")
        
        display_candidates.sort(key=lambda x: x["timestamp_utc"], reverse=True)

        digest_to_write_to_html = {}
        for i, item in enumerate(display_candidates):
            if i < MAX_TOPICS:
                digest_to_write_to_html[item["name"]] = item["articles"]
            else:
                logging.info(f"Topic '{item['name']}' (updated {item['timestamp_utc']}) not included in HTML due to MAX_TOPICS={MAX_TOPICS} limit, but retained in persisted state.")
        
        if digest_to_write_to_html:
            write_digest_html(digest_to_write_to_html, BASE_DIR, ZONE)
            logging.info(f"Digest HTML written/updated with {len(digest_to_write_to_html)} topics.")
        else:
            digest_html_path = os.path.join(BASE_DIR, "public", "digest.html")
            if os.path.exists(digest_html_path):
                logging.info("No topics to display in digest after processing. Existing digest.html will NOT be modified or deleted.")
            else:
                logging.info("No topics to display in digest after processing, and digest.html does not exist. It will not be created.")
        
        try:
            with open(DIGEST_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(persisted_digest_state, f, indent=2)
            logging.info(f"Full digest state (with {len(persisted_digest_state)} topics) saved to {DIGEST_STATE_FILE}")
        except IOError as e:
            logging.error(f"Failed to write digest state file {DIGEST_STATE_FILE}: {e}")
        
        if final_digest_from_llm_this_run: 
            update_history_file(final_digest_from_llm_this_run, history, HISTORY_FILE, ZONE)
        else:
            logging.info("No new articles from LLM this run, running history update for pruning old entries.")
            update_history_file({}, history, HISTORY_FILE, ZONE) # Pass empty dict to still trigger pruning logic


        if CONFIG.get("ENABLE_GIT_PUSH", False):
            perform_git_operations(BASE_DIR)
        else:
            logging.info("Git push is disabled in config. Skipping.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
    finally:
        logging.info(f"Script finished at {datetime.now(ZONE)}")

if __name__ == "__main__":
    main()