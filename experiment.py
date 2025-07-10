import json
import os
import datetime
import re
import logging
import sys
import subprocess # NEW IMPORT
from datetime import timezone # NEW IMPORT
from dateutil import parser as date_parser
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define all paths relative to the script's directory
SUMMARIES_FILE = os.path.join(script_dir, 'summaries.json')
EVALUATED_PREDICTIONS_FILE = os.path.join(script_dir, 'predictions.json')
LAST_RUN_TIMESTAMP_FILE = os.path.join(script_dir, 'timestamps.txt')
LOG_DIR = os.path.join(script_dir, 'logs')

PREDICTION_WINDOW_DAYS = 7
PREDICTION_DELIMITER = "In the near future, "
ALT_DELIMITERS = ["Outlook:", "Looking ahead,"]

# --- Load Environment Variables ---
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# NEW: Read Git push flag from environment, safely converting to boolean
ENABLE_GIT_PUSH = os.getenv('ENABLE_GIT_PUSH', 'false').lower() == 'true'


# --- Gemini Configuration ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
    GENERATION_CONFIG = {"temperature": 0.2, "response_mime_type": "application/json"}
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Failed to configure Gemini: {e}")
    sys.exit(1)

# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# File handler
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'experiment.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)
# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
# Add handlers if they aren't already present to avoid duplicates
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# --- Git Operations Function (Adapted from your provided guide) ---
def perform_git_operations(base_dir):
    """
    Performs a robust sequence of Git operations to commit and push generated files.
    """
    try:
        logging.info("--- Starting Git Operations ---")
        github_token = os.getenv("GITHUB_TOKEN")
        github_repo = os.getenv("GITHUB_REPOSITORY")
        github_email = os.getenv("GITHUB_EMAIL")
        github_user = os.getenv("GITHUB_USER", "Automated Bot")

        if not all([github_token, github_repo, github_email]):
            logging.error("Missing GITHUB_TOKEN, GITHUB_REPOSITORY, or GITHUB_EMAIL. Cannot push.")
            return

        remote_url = f"https://oauth2:{github_token}@github.com/{github_repo}.git"

        # Configure Git
        subprocess.run(["git", "config", "user.name", github_user], check=True, cwd=base_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", github_email], check=True, cwd=base_dir, capture_output=True)
        
        # Check current branch
        branch_result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True, cwd=base_dir)
        current_branch = branch_result.stdout.strip()
        if not current_branch or current_branch == "HEAD":
            current_branch = "main"

        # Check for local changes before pulling
        status_before_pull = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=base_dir).stdout.strip()
        if status_before_pull:
            logging.warning(f"Uncommitted changes detected before pull. Stashing is recommended for cron jobs. Proceeding with caution.")

        # Pull latest changes
        logging.info(f"Pulling latest changes from origin/{current_branch}...")
        pull_cmd = ["git", "pull", "origin", current_branch]
        pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, cwd=base_dir)

        if pull_result.returncode != 0:
            logging.error(f"'git pull' failed. Stderr: {pull_result.stderr.strip()}")
            logging.warning("Skipping push this cycle due to pull failure.")
            return
        
        # Add the files this script generates
        files_to_add = [EVALUATED_PREDICTIONS_FILE, LAST_RUN_TIMESTAMP_FILE]
        relative_files_to_add = [os.path.relpath(f, base_dir) for f in files_to_add if os.path.exists(f)]
        
        if not relative_files_to_add:
            logging.info("No generated files exist to add.")
        else:
            subprocess.run(["git", "add"] + relative_files_to_add, check=True, cwd=base_dir)

        # Check if there are any changes to commit
        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=base_dir)
        if not status_result.stdout.strip():
            logging.info("No changes to commit. Working tree is clean.")
            return

        # Commit the changes
        commit_message = f"Automated: Update prediction data on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=base_dir)
        logging.info(f"Committed changes with message: '{commit_message}'")

        # Push the changes
        logging.info(f"Pushing changes to origin/{current_branch}...")
        push_cmd = ["git", "push", remote_url, f"HEAD:{current_branch}"]
        subprocess.run(push_cmd, check=True, cwd=base_dir, capture_output=True)
        logging.info("Successfully pushed changes to GitHub.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {e.cmd}")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stderr: {e.stderr.decode(errors='ignore').strip() if e.stderr else 'N/A'}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during Git operations: {e}", exc_info=True)


# --- Helper Functions (Your existing functions go here) ---
# ... (load_json_data, save_json_data, parse_report_summary, etc. ... no changes needed)
def load_json_data(filepath, default_data=[]):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {filepath}. Returning default data.")
                return default_data
    return default_data

def save_json_data(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_report_summary(summary_text):
    predictions_text_list = []
    past_events_text = summary_text
    delimiter_to_use = None
    all_delimiters = [PREDICTION_DELIMITER] + ALT_DELIMITERS
    for delim in all_delimiters:
        match = re.search(re.escape(delim), summary_text, re.IGNORECASE)
        if match:
            delimiter_to_use = match.group(0)
            break
    if delimiter_to_use:
        try:
            parts = summary_text.split(delimiter_to_use, 1)
            past_events_text = parts[0].strip()
            if len(parts) > 1:
                full_prediction_block = parts[1].strip()
                raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])', full_prediction_block)
                predictions_text_list = [s.strip() for s in raw_sentences if s.strip()]
        except Exception as e:
            logging.warning(f"Error splitting summary with delimiter '{delimiter_to_use}': {e}")
            predictions_text_list = []
    past_events_text = re.sub(r'<br\s*/?>', '\n', past_events_text).strip()
    predictions_text_list = [re.sub(r'<br\s*/?>', '\n', p).strip() for p in predictions_text_list]
    return past_events_text, predictions_text_list

def generate_prediction_id(report_ts, prediction_text):
    return f"{report_ts}_{abs(hash(prediction_text))}"

def get_keywords(text):
    if not text: return set()
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "will", "be", "to", "of", "and", "in", "on", "it", "for", "with", "this", "that", "as", "at", "by", "likely", "less", "may", "might", "could", "would", "should"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {word for word in words if word not in stop_words and not word.isdigit()}

def check_prediction_occurrence(prediction_text, future_report_past_events_text, threshold=0.25):
    if not prediction_text or not future_report_past_events_text: return False, 0.0, []
    pred_keywords = get_keywords(prediction_text)
    event_keywords = get_keywords(future_report_past_events_text)
    if not pred_keywords: return False, 0.0, []
    common_keywords = pred_keywords.intersection(event_keywords)
    score = len(common_keywords) / len(pred_keywords) if pred_keywords else 0
    if score >= threshold and len(common_keywords) >= 2: return True, score, sorted(list(common_keywords))
    return False, score, sorted(list(common_keywords))

def _call_gemini_with_json_parsing(prompt_text, log_prefix="Gemini"):
    try:
        logging.info(f"{log_prefix}: Sending prompt to Gemini for JSON output...")
        response = gemini_model.generate_content(prompt_text)
        if not response.candidates:
            logging.error(f"{log_prefix}: No candidates returned. Full response: {response}")
            return {"error": "no_candidates", "message": "No candidates in response."}
        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            logging.warning(f"{log_prefix}: Generation did not finish normally. Reason: {candidate.finish_reason.name}. Safety Ratings: {candidate.safety_ratings}")
            return {"error": "generation_not_stopped", "message": f"Finish Reason: {candidate.finish_reason.name}"}
        json_str = response.text
        return json.loads(json_str)
    except json.JSONDecodeError as jde:
        raw_text = response.text if 'response' in locals() else 'N/A'
        logging.error(f"{log_prefix}: Failed to parse JSON from Gemini response: {jde}. Raw text: '{raw_text}'")
        return {"error": "json_decode_error", "message": str(jde)}
    except Exception as e:
        logging.error(f"{log_prefix}: Gemini request failed with unhandled exception: {e}", exc_info=True)
        return {"error": "unhandled_request_failed", "message": str(e)}

def check_prediction_vacuousness_gemini(prediction_text, context_summary_text, report_date_str):
    prompt = f"""
    You are an AI assistant evaluating the quality of a geo-political/economic prediction. Your task is to assess if the given prediction is specific and testable, or if it is too vague, obvious, or untestable. A "vacuous" prediction: 1. States something almost always true (e.g., "markets will fluctuate"). 2. Lacks specific actors, actions, or measurable outcomes. 3. Merely rephrases an ongoing situation from the provided context without predicting a distinct new development. Prediction: "{prediction_text}" Context from week ending {report_date_str}: --- {context_summary_text[:4000]} --- Your output MUST be a single JSON object with the following keys: - "is_obvious_or_vacuous": boolean - "obviousness_score": float (0.0 for highly specific, 1.0 for extremely obvious) - "reasoning": string (brief explanation)
    """
    return _call_gemini_with_json_parsing(prompt, log_prefix=f"VacuousnessCheck for '{prediction_text[:30]}...'")

def get_prior_likelihood_from_gemini(prediction_text, context_summary_text, report_date_str):
    prompt = f"""
    You are an AI assistant estimating the likelihood of a prediction. Based ONLY on the provided context (events from the week ending {report_date_str}) and general knowledge up to that date, estimate the probability of the prediction coming true in the next 1-2 weeks. Prediction: "{prediction_text}" Context from week ending {report_date_str}: --- {context_summary_text[:4000]} --- Your output MUST be a single JSON object with the following keys: - "prior_probability": float (from 0.0 to 1.0) - "likelihood_category": string (e.g., "Very Low", "Low", "Moderate", "High", "Very High") - "rationale": string (1-2 sentence explanation)
    """
    return _call_gemini_with_json_parsing(prompt, log_prefix=f"PriorLikelihood for '{prediction_text[:30]}...'")

def get_file_fingerprint(filepath):
    if not os.path.exists(filepath): return (0, 0)
    stat = os.stat(filepath)
    return (stat.st_mtime, stat.st_size)

def read_last_run_fingerprint():
    if not os.path.exists(LAST_RUN_TIMESTAMP_FILE): return (0, 0)
    with open(LAST_RUN_TIMESTAMP_FILE, 'r') as f:
        try:
            mtime, size = f.read().strip().split(',')
            return (float(mtime), int(size))
        except ValueError: return (0,0)

def write_current_fingerprint(fingerprint):
    with open(LAST_RUN_TIMESTAMP_FILE, 'w') as f:
        f.write(f"{fingerprint[0]},{fingerprint[1]}")

def _gemini_check_previously_attempted_and_concluded(check_result_dict):
    if check_result_dict is None: return False
    if "error" in check_result_dict: return True
    return True

# --- Main Logic ---
def main():
    if not os.path.exists(SUMMARIES_FILE):
        logging.info(f"{SUMMARIES_FILE} not found. Exiting.")
        return

    current_summaries_fingerprint = get_file_fingerprint(SUMMARIES_FILE)
    last_run_summaries_fingerprint = read_last_run_fingerprint()

    # We check made_eval_data_changes_in_run later to decide if we need to push
    made_eval_data_changes_in_run = False 

    if current_summaries_fingerprint == last_run_summaries_fingerprint and os.path.exists(EVALUATED_PREDICTIONS_FILE):
        logging.info(f"No changes in {SUMMARIES_FILE} since last run. Exiting without push.")
        return
    
    logging.info(f"Detected changes in {SUMMARIES_FILE} or first run. Starting processing...")
    
    # ... (the entire body of your main processing loop goes here, unchanged) ...
    all_summaries = load_json_data(SUMMARIES_FILE)
    if not all_summaries:
        logging.warning(f"No summaries loaded from {SUMMARIES_FILE}. Exiting.")
        write_current_fingerprint(current_summaries_fingerprint)
        return

    all_summaries.sort(key=lambda x: date_parser.isoparse(x['timestamp']))
    evaluated_predictions_dict = {p['prediction_id']: p for p in load_json_data(EVALUATED_PREDICTIONS_FILE)}
    
    new_predictions_this_run, gemini_vacuous_calls_this_run, gemini_prior_calls_this_run, predictions_filtered_as_obvious_this_run = 0, 0, 0, 0

    for i, report_n in enumerate(all_summaries):
        report_n_timestamp_str = report_n['timestamp']
        report_n_dt = date_parser.isoparse(report_n_timestamp_str)
        report_n_past_events_text, report_n_predictions_list = parse_report_summary(report_n['summary'])
        if not report_n_predictions_list: continue

        for pred_text in report_n_predictions_list:
            if not pred_text.strip() or len(pred_text.strip().split()) < 4: continue
            
            pred_id = generate_prediction_id(report_n_timestamp_str, pred_text)
            this_pred_entry_modified_in_run = False

            if pred_id not in evaluated_predictions_dict:
                new_predictions_this_run += 1; this_pred_entry_modified_in_run = True
                current_eval_entry = { "prediction_id": pred_id, "source_report_timestamp": report_n_timestamp_str, "prediction_text": pred_text, "predicted_period_start": (report_n_dt).isoformat(), "predicted_period_end": (report_n_dt + datetime.timedelta(days=PREDICTION_WINDOW_DAYS)).isoformat(), "status": "pending_processing", "vacuousness_check": None, "prior_likelihood": None, "verification_details": {} }
                evaluated_predictions_dict[pred_id] = current_eval_entry
            else:
                current_eval_entry = evaluated_predictions_dict[pred_id]
            
            if not _gemini_check_previously_attempted_and_concluded(current_eval_entry.get("vacuousness_check")):
                logging.info(f"Running vacuousness check for {pred_id}...")
                vac_check_result = check_prediction_vacuousness_gemini(pred_text, report_n_past_events_text, report_n_timestamp_str)
                gemini_vacuous_calls_this_run += 1; this_pred_entry_modified_in_run = True
                current_eval_entry["vacuousness_check"] = vac_check_result
                if "error" in vac_check_result: current_eval_entry["status"] = "error_vacuousness_check"
                elif vac_check_result.get("is_obvious_or_vacuous"): current_eval_entry["status"] = "filtered_obvious"; predictions_filtered_as_obvious_this_run += 1
                else: current_eval_entry["status"] = "pending_prior_estimation"
                evaluated_predictions_dict[pred_id] = current_eval_entry

            if current_eval_entry["status"] == "pending_prior_estimation" and not _gemini_check_previously_attempted_and_concluded(current_eval_entry.get("prior_likelihood")):
                logging.info(f"Estimating prior likelihood for {pred_id}...")
                prior_result = get_prior_likelihood_from_gemini(pred_text, report_n_past_events_text, report_n_timestamp_str)
                gemini_prior_calls_this_run += 1; this_pred_entry_modified_in_run = True
                current_eval_entry["prior_likelihood"] = prior_result
                if "error" in prior_result: current_eval_entry["status"] = "error_prior_estimation"
                else: current_eval_entry["status"] = "pending_verification"
                evaluated_predictions_dict[pred_id] = current_eval_entry

            if current_eval_entry["status"] in ["pending_verification", "verified_not_occurred"]:
                original_status = current_eval_entry["status"]
                if "verification_details" not in current_eval_entry or not isinstance(current_eval_entry.get("verification_details"), dict): current_eval_entry["verification_details"] = {}
                verification_dict = current_eval_entry["verification_details"]
                verification_dict.setdefault("checked_against_reports", []); verification_dict.setdefault("evidence_report_timestamp", None); verification_dict.setdefault("match_score", 0.0)
                current_pred_end_dt = date_parser.isoparse(current_eval_entry["predicted_period_end"])
                for j in range(i + 1, len(all_summaries)):
                    report_n_plus_k = all_summaries[j]
                    report_n_plus_k_timestamp_str = report_n_plus_k['timestamp']
                    if report_n_plus_k_timestamp_str in verification_dict["checked_against_reports"]: continue
                    verification_dict["checked_against_reports"].append(report_n_plus_k_timestamp_str); this_pred_entry_modified_in_run = True
                    future_report_past_events, _ = parse_report_summary(report_n_plus_k['summary'])
                    if not future_report_past_events: continue
                    occurred, score, common_kws = check_prediction_occurrence(pred_text, future_report_past_events)
                    if occurred and score > verification_dict["match_score"]:
                        current_eval_entry["status"] = "verified_occurred"; verification_dict["evidence_report_timestamp"] = report_n_plus_k_timestamp_str; verification_dict["match_score"] = round(score, 3); verification_dict["matching_keywords"] = common_kws
                        if score > 0.4: break
                if current_eval_entry["status"] == "pending_verification":
                    last_summary_dt = date_parser.isoparse(all_summaries[-1]['timestamp'])
                    if last_summary_dt > current_pred_end_dt + datetime.timedelta(days=PREDICTION_WINDOW_DAYS + 3): current_eval_entry["status"] = "verified_not_occurred"
                if current_eval_entry["status"] != original_status: this_pred_entry_modified_in_run = True
                evaluated_predictions_dict[pred_id] = current_eval_entry
            if this_pred_entry_modified_in_run: made_eval_data_changes_in_run = True

    if made_eval_data_changes_in_run:
        logging.info(f"Saving updated evaluations to {EVALUATED_PREDICTIONS_FILE}.")
        save_json_data(EVALUATED_PREDICTIONS_FILE, list(evaluated_predictions_dict.values()))
    else:
        logging.info(f"No significant changes made to evaluation data during this run.")

    write_current_fingerprint(current_summaries_fingerprint)
    logging.info(f"Processing complete. New predictions: {new_predictions_this_run}. Filtered: {predictions_filtered_as_obvious_this_run}. Gemini Vacuousness Calls: {gemini_vacuous_calls_this_run}. Gemini Prior Calls: {gemini_prior_calls_this_run}.")
    
    run_analysis(list(evaluated_predictions_dict.values()))

    # --- NEW: Call Git operations at the end if enabled and changes were made ---
    if ENABLE_GIT_PUSH and made_eval_data_changes_in_run:
        perform_git_operations(script_dir)
    elif ENABLE_GIT_PUSH:
        logging.info("Git push enabled, but no data changes were made. Skipping push.")


def run_analysis(predictions):
    # ... (your existing run_analysis function, unchanged)
    logging.info("\n--- Event Prediction Accuracy Analysis ---")
    final_predictions = [p for p in predictions if p['status'] not in ["filtered_obvious", "pending_processing", "pending_prior_estimation", "error_vacuousness_check", "error_prior_estimation"]]
    if not final_predictions:
        logging.info("No predictions are ready for final analysis.")
        return
    verified_occurred = [p for p in final_predictions if p['status'] == 'verified_occurred']
    verified_not_occurred = [p for p in final_predictions if p['status'] == 'verified_not_occurred']
    pending_verification = [p for p in final_predictions if p['status'] == 'pending_verification']
    logging.info(f"Total Evaluable Predictions (passed filters): {len(final_predictions)}")
    logging.info(f"  - Verified Occurred: {len(verified_occurred)}")
    logging.info(f"  - Verified Not Occurred: {len(verified_not_occurred)}")
    logging.info(f"  - Still Pending Verification: {len(pending_verification)}")
    num_definitively_evaluated = len(verified_occurred) + len(verified_not_occurred)
    if num_definitively_evaluated > 0:
        hit_rate = len(verified_occurred) / num_definitively_evaluated
        logging.info(f"Hit Rate (Occurred / Total Definitively Evaluated): {hit_rate:.2%}")
    else:
        logging.info("Not enough definitively evaluated predictions for a hit rate.")
    preds_for_brier = [p for p in (verified_occurred + verified_not_occurred) if p.get("prior_likelihood") and isinstance(p["prior_likelihood"].get("prior_probability"), (float, int))]
    if preds_for_brier:
        brier_scores = [(p["prior_likelihood"]["prior_probability"] - (1.0 if p['status'] == 'verified_occurred' else 0.0))**2 for p in preds_for_brier]
        avg_brier_score = sum(brier_scores) / len(brier_scores)
        logging.info(f"Average Brier Score (for {len(brier_scores)} preds): {avg_brier_score:.4f} (lower is better)")
    else:
        logging.info("No predictions available for Brier score calculation.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical("An unhandled exception occurred in main execution.", exc_info=True)
        sys.exit(1)