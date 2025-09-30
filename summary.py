# Author: Blake Rayvid <https://github.com/brayvid/based-news>

import os
import sys
import json
import logging
import csv
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess
import html
import re

# --- Configuration ---
CONFIG_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTWCrmL5uXBJ9_pORfhESiZyzD3Yw9ci0Y-fQfv0WATRDq6T8dX0E7yz1XNfA6f92R7FDmK40MFSdH4/pub?gid=446667252&single=true&output=csv"

# --- Setup Paths ---
try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "public", "summary.html")
SUMMARIES_LOG_FILE = os.path.join(BASE_DIR, "summaries.json")
LOGFILE = os.path.join(BASE_DIR, "logs/summary.log")
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

# --- Logging ---
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Summary script started.")

# --- Load environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
                key, val = row[0].strip(), row[1].strip()
                try:
                    if '.' in val and val.replace('.', '', 1).isdigit():
                        config[key] = float(val)
                    else:
                        config[key] = int(val)
                except ValueError:
                    if val.lower() == 'true': config[key] = True
                    elif val.lower() == 'false': config[key] = False
                    else: config[key] = val
        logging.info("Successfully loaded config from Google Sheet.")
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {url}: {e}")
        return None

# --- Load Configuration ---
CONFIG = load_config_from_sheet(CONFIG_CSV_URL)
if CONFIG is None:
    logging.critical("Fatal: Unable to load CONFIG from sheet. Exiting.")
    sys.exit(1)

GEMINI_MODEL_NAME = CONFIG.get("SUMMARY_GEMINI_MODEL_NAME", "gemini-2.5-pro")
USER_TIMEZONE = CONFIG.get("TIMEZONE", "America/New_York")
try:
    ZONE = ZoneInfo(USER_TIMEZONE)
except Exception:
    ZONE = ZoneInfo("America/New_York")
    logging.warning(f"Invalid TIMEZONE. Falling back to '{ZONE}'.")

# --- START: SCRIPT FIX ---
def format_digest_for_summary(data):
    """Formats the history.json structure for the LLM prompt, including dates."""
    parts = []
    for topic, articles in data.items():
        if articles:
            parts.append(f"### {html.unescape(topic)}")
            for article in articles:
                # Use the correct key 'pubDate'
                date_str = article.get("pubDate", "")
                formatted_date = ""
                if date_str:
                    try:
                        # Parse the specific date format: "Tue, 23 Sep 2025 19:41:15 GMT"
                        dt_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                        formatted_date = dt_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        logging.warning(f"Could not parse date: {date_str}")
                        formatted_date = "" # Fallback to empty if format is wrong

                title = html.unescape(article.get('title', 'No Title'))
                parts.append(f"- {formatted_date} - {title}")
    return "\n".join(parts)
# --- END: SCRIPT FIX ---

def generate_summary(digest_content: dict) -> str:
    """Generates a summary from the digest content using Gemini."""
    if not GEMINI_API_KEY:
        logging.error("Missing GEMINI_API_KEY. Cannot generate summary.")
        return ""

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    # Updated prompt to guide the model on using the provided dates
    prompt = (
        "Give a brief report on how the world has been doing lately based on the attached headlines. Use the dates provided with each headline to correctly identify the timeframe of your report (e.g., 'In late September 2025'). "
        "Use simple language, cite figures, and be specific with people, places, things, etc. Do not use bullet points, section headings, or any markdown formatting. Use only complete sentences. "
        "Don't state that it's a report, simply present the findings. "
        "At the end, in a separate paragraph of about 50 words, use all available clues to predict what will likely occur in the near future, what is less likely but still possible, and the potential ramifications. "
        "Separate paragraphs with a single newline."
    )

    # Combine prompt and data into a single request
    full_prompt = [
        prompt,
        "\n---BEGIN HEADLINES---\n",
        format_digest_for_summary(digest_content),
        "\n---END HEADLINES---"
    ]

    try:
        logging.info(f"Sending prompt to Gemini model '{GEMINI_MODEL_NAME}' (Search grounding disabled).")
        response = model.generate_content(full_prompt)
        
        if response.text:
            logging.info("Successfully received summary from Gemini.")
            return response.text.strip()
        else:
            logging.warning("Gemini returned an empty response. There may have been a content safety block.")
            if response.prompt_feedback:
                logging.warning(f"Prompt Feedback: {response.prompt_feedback}")
            return ""
    except Exception as e:
        logging.error(f"Gemini request failed: {e}", exc_info=True)
        return ""

def perform_git_operations(base_dir, current_zone, config_obj):
    """Performs a robust sequence of Git operations to commit and push changes."""
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        github_repository = os.getenv("GITHUB_REPOSITORY")
        if not github_token or not github_repository:
            logging.error("GITHUB_TOKEN or GITHUB_REPOSITORY not set. Cannot push.")
            return

        remote_url = f"https://oauth2:{github_token}@github.com/{github_repository}.git"
        author_name = config_obj.get("GIT_USER_NAME", "Automated Digest Bot")
        author_email = config_obj.get("GIT_USER_EMAIL", "bot@example.com")

        # Set Git config
        subprocess.run(["git", "config", "user.name", author_name], check=True, cwd=base_dir)
        subprocess.run(["git", "config", "user.email", author_email], check=True, cwd=base_dir)
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, cwd=base_dir, capture_output=True)

        current_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True, cwd=base_dir).stdout.strip()

        # Safely pull changes
        logging.info("Stashing local changes before pull...")
        subprocess.run(["git", "stash", "push", "-u", "-m", "WIP_Summary_Script"], check=True, cwd=base_dir, capture_output=True)
        logging.info(f"Pulling with rebase from origin/{current_branch}...")
        subprocess.run(["git", "pull", "--rebase", "origin", current_branch], check=True, cwd=base_dir, capture_output=True)
        logging.info("Popping stashed changes...")
        subprocess.run(["git", "stash", "pop"], check=True, cwd=base_dir, capture_output=True)

        # Add, Commit, and Push
        files_to_add = [os.path.relpath(SUMMARY_FILE, base_dir), os.path.relpath(SUMMARIES_LOG_FILE, base_dir)]
        logging.info(f"Staging files: {files_to_add}")
        subprocess.run(["git", "add"] + files_to_add, check=True, cwd=base_dir)

        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=base_dir)
        if not status_result.stdout.strip():
            logging.info("No changes to commit.")
            return

        commit_message = f"Auto-update news summary - {datetime.now(current_zone).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=base_dir)
        logging.info(f"Pushing changes to origin/{current_branch}...")
        subprocess.run(["git", "push", "origin", current_branch], check=True, cwd=base_dir)
        logging.info("Summary committed and pushed to GitHub.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {e.cmd}\n{e.stderr.decode() if e.stderr else e.stdout.decode()}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Git operations: {e}", exc_info=True)


def main():
    # 1. Load the latest digest content
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            digest_data = json.load(f)
        logging.info(f"Successfully loaded headlines from: {HISTORY_FILE}")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logging.critical(f"Failed to load or parse {HISTORY_FILE}: {e}. Cannot generate summary.")
        sys.exit(1)

    # 2. Generate the summary
    summary_text = generate_summary(digest_data)
    if not summary_text:
        logging.warning("No summary was generated. Script will exit without updating files.")
        sys.exit(0)

    # 3. Format for HTML and write to file
    formatted_html_body = re.sub(r'\n+', '<br><br>', summary_text)
    
    timestamp = datetime.now(ZONE).strftime("%A, %d %B %Y %I:%M %p %Z")
    html_output = (
        f"<p>{formatted_html_body}</p>\n"
        f"<div class='timestamp' id='summary-last-updated' style='display: none;'>Last updated: {timestamp}</div>"
    )
    
    try:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(html_output)
        logging.info(f"Summary written to {SUMMARY_FILE}")
    except IOError as e:
        logging.error(f"Failed to write summary.html: {e}")

    # 4. Append to the persistent JSON log of all summaries
    summary_entry = {
        "timestamp": datetime.now(ZONE).isoformat(),
        "summary_html": formatted_html_body
    }
    try:
        summaries = []
        if os.path.exists(SUMMARIES_LOG_FILE):
            with open(SUMMARIES_LOG_FILE, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        summaries.append(summary_entry) # Add newest to the bottom
        with open(SUMMARIES_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        logging.info(f"Summary appended to {SUMMARIES_LOG_FILE}")
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to append to summaries log: {e}")
        
    # 5. Push changes to Git if enabled
    if CONFIG.get("ENABLE_GIT_PUSH", False):
        perform_git_operations(BASE_DIR, ZONE, CONFIG)
    else:
        logging.info("Git push is disabled in config. Skipping.")

    logging.info("Summary script finished successfully.")


if __name__ == "__main__":
    main()