import json
import os
import datetime
import re
import logging
import sys
from dateutil import parser as date_parser
import google.generativeai as genai
from dotenv import load_dotenv # Import load_dotenv

# --- Configuration ---
SUMMARIES_FILE = 'summaries.json'
EVALUATED_PREDICTIONS_FILE = 'predictions.json'
PREDICTION_WINDOW_DAYS = 7
PREDICTION_DELIMITER = "In the near future, "
ALT_DELIMITERS = ["Outlook:", "Looking ahead,"]
LAST_RUN_TIMESTAMP_FILE = 'timestamps.txt' # To track last processed summary file state

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file

# --- Gemini Configuration ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set or found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
    GENERATION_CONFIG = {
        "temperature": 0.3, "top_p": 1, "top_k": 1, "max_output_tokens": 512,
    }
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME,
                                        generation_config=GENERATION_CONFIG,
                                        safety_settings=SAFETY_SETTINGS)
except ValueError as e:
    logging.error(f"Gemini API Key Error: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error configuring Gemini: {e}")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (largely unchanged, slight modifications for new file check logic) ---
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
    # (Same as before)
    predictions_text_list = []
    past_events_text = summary_text
    delimiter_to_use = None
    delimiter_pos = -1

    if PREDICTION_DELIMITER.lower() in summary_text.lower():
        delimiter_to_use = PREDICTION_DELIMITER
    else:
        for alt_delim in ALT_DELIMITERS:
            if alt_delim.lower() in summary_text.lower():
                delimiter_to_use = alt_delim
                break
    
    if delimiter_to_use:
        try:
            match = re.search(re.escape(delimiter_to_use), summary_text, re.IGNORECASE)
            if match:
                delimiter_pos = match.start()
                past_events_text = summary_text[:delimiter_pos].strip()
                full_prediction_block = summary_text[delimiter_pos + len(delimiter_to_use):].strip()
                raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', full_prediction_block)
                predictions_text_list = [s.strip() for s in raw_sentences if s.strip()]
            else:
                 predictions_text_list = []
        except Exception as e:
            logging.warning(f"Error splitting summary with delimiter '{delimiter_to_use}': {e}")
            predictions_text_list = []
    
    past_events_text = re.sub(r'<br\s*/?>', '\n', past_events_text).strip()
    predictions_text_list = [re.sub(r'<br\s*/?>', '\n', p).strip() for p in predictions_text_list]
    return past_events_text, predictions_text_list


def generate_prediction_id(report_ts, prediction_text):
    # (Same as before)
    return f"{report_ts}_{hash(prediction_text)}"

def get_keywords(text, top_n=10):
    # (Same as before)
    if not text: return set()
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "will", "be", "to", "of", "and", "in", "on", "it", "for", "with", "this", "that", "as", "at", "by"}
    words = re.findall(r'\b\w{3,}\b', text.lower())
    keywords = {word for word in words if word not in stop_words}
    return keywords

def check_prediction_occurrence(prediction_text, future_report_past_events_text, threshold=0.3):
    # (Same as before)
    if not prediction_text or not future_report_past_events_text:
        return False, 0.0, []
    pred_keywords = get_keywords(prediction_text)
    event_keywords = get_keywords(future_report_past_events_text)
    if not pred_keywords:
        return False, 0.0, []
    common_keywords = pred_keywords.intersection(event_keywords)
    score = len(common_keywords) / len(pred_keywords) if pred_keywords else 0
    if score >= threshold and common_keywords:
        return True, score, sorted(list(common_keywords))
    return False, score, sorted(list(common_keywords))

def get_prior_likelihood_from_gemini(prediction_text, context_summary_text, report_date_str):
    # (Same as before - with robust JSON parsing and error handling)
    question = f"""
    Given the following summary of world events from the week ending {report_date_str}:
    --- CONTEXT START ---
    {context_summary_text}
    --- CONTEXT END ---

    Consider the following prediction made for the near future (next 1-2 weeks) based *only* on the context above and general knowledge available up to {report_date_str}:
    Prediction: "{prediction_text}"

    Please estimate the prior likelihood of this specific prediction coming true.
    Provide your answer in JSON format with the following keys:
    - "prior_probability": A numerical probability from 0.0 (almost impossible) to 1.0 (almost certain).
    - "likelihood_category": A category (e.g., "Very Low", "Low", "Moderate", "High", "Very High").
    - "rationale": A brief explanation for your estimation (1-2 sentences).

    Example JSON output:
    {{
      "prior_probability": 0.65,
      "likelihood_category": "High",
      "rationale": "Given the escalating trade tensions mentioned, further market instability is a probable outcome."
    }}
    """
    prompt = question
    try:
        logging.info(f"Sending prompt to Gemini for prediction: '{prediction_text[:50]}...'")
        response = gemini_model.generate_content(prompt)
        cleaned_text = response.text.strip()
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            first_brace = cleaned_text.find('{')
            last_brace = cleaned_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = cleaned_text[first_brace : last_brace+1]
            else:
                logging.warning(f"Could not find a clear JSON block in Gemini response for: '{prediction_text[:50]}...' Response: {cleaned_text}")
                return None
        parsed_answer = json.loads(json_str)
        if not all(k in parsed_answer for k in ["prior_probability", "likelihood_category", "rationale"]):
            logging.warning(f"Gemini response missing required keys for: '{prediction_text[:50]}...'. Response: {parsed_answer}")
            return None
        if not isinstance(parsed_answer["prior_probability"], (int, float)) or not (0.0 <= parsed_answer["prior_probability"] <= 1.0):
            logging.warning(f"Invalid prior_probability from Gemini: {parsed_answer['prior_probability']}")
            try:
                if isinstance(parsed_answer["prior_probability"], str) and parsed_answer["prior_probability"].endswith('%'):
                    parsed_answer["prior_probability"] = float(parsed_answer["prior_probability"].rstrip('%')) / 100.0
                    if not (0.0 <= parsed_answer["prior_probability"] <= 1.0): return None
                elif not isinstance(parsed_answer["prior_probability"], (int, float)): return None
            except: return None
        logging.info(f"Gemini returned prior likelihood for '{prediction_text[:50]}...': {parsed_answer['likelihood_category']} ({parsed_answer['prior_probability']})")
        return parsed_answer
    except genai.types.generation_types.BlockedPromptException as bpe:
        logging.error(f"Gemini prompt was blocked: {bpe}. Prediction: '{prediction_text[:50]}...'")
        return None
    except genai.types.generation_types.StopCandidateException as sce:
        logging.error(f"Gemini generation stopped unexpectedly: {sce}. Prediction: '{prediction_text[:50]}...'")
        return None
    except json.JSONDecodeError as jde:
        logging.error(f"Failed to parse JSON from Gemini response: {jde}. Raw response: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        if hasattr(e, 'message') and "API key not valid" in str(e.message):
            logging.error(f"Gemini API Key Error: {e}")
            sys.exit("Exiting due to Gemini API Key error.")
        logging.error(f"Gemini request failed for '{prediction_text[:50]}...': {e}", exc_info=True)
        return None

def get_file_fingerprint(filepath):
    """Returns a fingerprint (mtime, size) for a file."""
    if not os.path.exists(filepath):
        return (0, 0)
    stat = os.stat(filepath)
    return (stat.st_mtime, stat.st_size)

def read_last_run_fingerprint():
    if os.path.exists(LAST_RUN_TIMESTAMP_FILE):
        with open(LAST_RUN_TIMESTAMP_FILE, 'r') as f:
            try:
                mtime, size = f.read().strip().split(',')
                return (float(mtime), int(size))
            except ValueError:
                return (0,0) # Invalid format
    return (0, 0) # No previous run data

def write_current_fingerprint(fingerprint):
    with open(LAST_RUN_TIMESTAMP_FILE, 'w') as f:
        f.write(f"{fingerprint[0]},{fingerprint[1]}")

# --- Main Script Execution ---
def main():
    # --- Check for new summaries ---
    if not os.path.exists(SUMMARIES_FILE):
        logging.info(f"{SUMMARIES_FILE} not found. Exiting.")
        return

    current_summaries_fingerprint = get_file_fingerprint(SUMMARIES_FILE)
    last_run_summaries_fingerprint = read_last_run_fingerprint()

    # If summaries file hasn't changed since last successful processing of it, exit.
    if current_summaries_fingerprint == last_run_summaries_fingerprint:
        logging.info("No new changes in summaries.json since last successful run. Exiting.")
        return
    
    # --- Proceed with processing ---
    logging.info("Detected changes in summaries.json or first run. Starting processing...")

    all_summaries = load_json_data(SUMMARIES_FILE)
    if not all_summaries:
        logging.info(f"No summaries loaded from {SUMMARIES_FILE} despite file change. Exiting.")
        # Potentially update fingerprint here if we consider an empty file "processed"
        # write_current_fingerprint(current_summaries_fingerprint)
        return

    all_summaries.sort(key=lambda x: date_parser.isoparse(x['timestamp']))
    
    evaluated_predictions = load_json_data(EVALUATED_PREDICTIONS_FILE)
    evaluated_predictions_dict = {p['prediction_id']: p for p in evaluated_predictions}

    new_predictions_extracted_this_run = 0
    predictions_sent_to_gemini_this_run = 0
    made_changes_to_evaluations = False # Flag to track if evaluated_predictions_dict is modified

    for i, report_n in enumerate(all_summaries):
        report_n_timestamp_str = report_n['timestamp']
        report_n_dt = date_parser.isoparse(report_n_timestamp_str)
        
        report_n_past_events_text, report_n_predictions_list = parse_report_summary(report_n['summary'])

        if not report_n_predictions_list:
            continue

        prediction_period_start_dt = report_n_dt
        prediction_period_end_dt = report_n_dt + datetime.timedelta(days=PREDICTION_WINDOW_DAYS)

        for pred_text in report_n_predictions_list:
            if not pred_text.strip(): continue
            
            pred_id = generate_prediction_id(report_n_timestamp_str, pred_text)
            original_eval_entry = evaluated_predictions_dict.get(pred_id, {}).copy() # For change detection

            if pred_id not in evaluated_predictions_dict:
                new_predictions_extracted_this_run += 1
                logging.info(f"New prediction found (ID: {pred_id}). Querying Gemini for prior likelihood.")
                prior_info = get_prior_likelihood_from_gemini(pred_text, report_n_past_events_text, report_n_timestamp_str)
                predictions_sent_to_gemini_this_run += 1
                
                current_eval_entry = {
                    "prediction_id": pred_id,
                    "source_report_timestamp": report_n_timestamp_str,
                    "prediction_text": pred_text,
                    "predicted_period_start": prediction_period_start_dt.isoformat(),
                    "predicted_period_end": prediction_period_end_dt.isoformat(),
                    "estimated_prior_probability": prior_info["prior_probability"] if prior_info else None,
                    "estimated_prior_likelihood_category": prior_info["likelihood_category"] if prior_info else None,
                    "prior_estimation_rationale": prior_info["rationale"] if prior_info else "Failed to obtain from Gemini.",
                    "status": "pending_verification",
                    "verification_details": {
                        "checked_against_reports": [], "evidence_report_timestamp": None,
                        "evidence_text_snippet": None, "match_score": 0.0,
                        "matching_keywords": [], "notes": ""
                    }
                }
                evaluated_predictions_dict[pred_id] = current_eval_entry
                made_changes_to_evaluations = True
            else:
                current_eval_entry = evaluated_predictions_dict[pred_id]
                # Backfill priors if missing
                if "estimated_prior_probability" not in current_eval_entry or \
                   (current_eval_entry.get("estimated_prior_probability") is None and \
                    current_eval_entry.get("prior_estimation_rationale") != "Failed to obtain from Gemini." and \
                    current_eval_entry.get("prior_estimation_rationale") != "Prior estimation not attempted pre-Gemini integration.") :
                    logging.info(f"Existing prediction {pred_id} missing prior. Querying Gemini.")
                    prior_info = get_prior_likelihood_from_gemini(
                        current_eval_entry["prediction_text"], 
                        report_n_past_events_text, # Context from the current report_n
                        current_eval_entry["source_report_timestamp"]
                    )
                    predictions_sent_to_gemini_this_run +=1
                    if prior_info:
                        current_eval_entry["estimated_prior_probability"] = prior_info["prior_probability"]
                        current_eval_entry["estimated_prior_likelihood_category"] = prior_info["likelihood_category"]
                        current_eval_entry["prior_estimation_rationale"] = prior_info["rationale"]
                    else:
                        current_eval_entry["estimated_prior_probability"] = None
                        current_eval_entry["estimated_prior_likelihood_category"] = None
                        current_eval_entry["prior_estimation_rationale"] = "Failed to obtain from Gemini."
                    evaluated_predictions_dict[pred_id] = current_eval_entry
                    # made_changes_to_evaluations = True # This change alone might not warrant saving if no status changes

            # --- Verification Logic ---
            # Only proceed if status allows for update or if new checks might change it
            if current_eval_entry["status"] == "pending_verification" or current_eval_entry["status"] == "verified_not_occurred":
                initial_status = current_eval_entry["status"]
                initial_match_score = current_eval_entry["verification_details"].get("match_score", 0.0)

                found_occurrence_in_current_run_for_this_pred = False # Tracks if *this specific prediction* was updated
                best_match_score_this_pred = current_eval_entry["verification_details"].get("match_score", 0.0)

                for j in range(i + 1, len(all_summaries)):
                    report_n_plus_k = all_summaries[j]
                    report_n_plus_k_timestamp_str = report_n_plus_k['timestamp']
                    report_n_plus_k_dt = date_parser.isoparse(report_n_plus_k_timestamp_str)

                    if report_n_plus_k_timestamp_str in current_eval_entry["verification_details"]["checked_against_reports"] and current_eval_entry['status'] != 'pending_verification':
                        continue
                    if report_n_plus_k_dt < date_parser.isoparse(current_eval_entry["predicted_period_start"]): # Use prediction's start date
                        continue

                    if report_n_plus_k_timestamp_str not in current_eval_entry["verification_details"]["checked_against_reports"]:
                         current_eval_entry["verification_details"]["checked_against_reports"].append(report_n_plus_k_timestamp_str)
                    
                    future_report_past_events, _ = parse_report_summary(report_n_plus_k['summary'])
                    if not future_report_past_events: continue
                    
                    pred_end_dt_for_check = date_parser.isoparse(current_eval_entry["predicted_period_end"])
                    is_prime_candidate = pred_end_dt_for_check <= report_n_plus_k_dt < (pred_end_dt_for_check + datetime.timedelta(days=PREDICTION_WINDOW_DAYS/2 + 2))
                    occurred, score, common_kws = check_prediction_occurrence(current_eval_entry["prediction_text"], future_report_past_events)
                    
                    if occurred:
                        if score > best_match_score_this_pred or \
                           (score == best_match_score_this_pred and is_prime_candidate and not current_eval_entry["verification_details"]["evidence_report_timestamp"]): # Prioritize prime if scores are equal
                            found_occurrence_in_current_run_for_this_pred = True
                            best_match_score_this_pred = score
                            current_eval_entry["status"] = "verified_occurred"
                            current_eval_entry["verification_details"]["evidence_report_timestamp"] = report_n_plus_k_timestamp_str
                            current_eval_entry["verification_details"]["evidence_text_snippet"] = future_report_past_events[:500] + "..."
                            current_eval_entry["verification_details"]["match_score"] = round(score, 3)
                            current_eval_entry["verification_details"]["matching_keywords"] = common_kws
                            current_eval_entry["verification_details"]["notes"] = f"Matched in report {report_n_plus_k_timestamp_str}."
                            if is_prime_candidate and score > 0.5: break 
                
                if not found_occurrence_in_current_run_for_this_pred and current_eval_entry["status"] == "pending_verification":
                    last_summary_dt = date_parser.isoparse(all_summaries[-1]['timestamp'])
                    pred_end_dt_for_check = date_parser.isoparse(current_eval_entry["predicted_period_end"])
                    if last_summary_dt > pred_end_dt_for_check + datetime.timedelta(days=PREDICTION_WINDOW_DAYS + 2):
                         current_eval_entry["status"] = "verified_not_occurred"
                         current_eval_entry["verification_details"]["notes"] = "No strong match found in subsequent reports checked after sufficient time."
                
                if current_eval_entry["status"] != initial_status or \
                   current_eval_entry["verification_details"].get("match_score", 0.0) != initial_match_score or \
                   current_eval_entry != original_eval_entry: # General check if anything changed in the entry
                    made_changes_to_evaluations = True
                evaluated_predictions_dict[pred_id] = current_eval_entry

    if new_predictions_extracted_this_run > 0 or predictions_sent_to_gemini_this_run > 0:
        made_changes_to_evaluations = True # If we sent to Gemini, we likely made changes or want to save new prior info

    if made_changes_to_evaluations:
        logging.info(f"Saving updated evaluations to {EVALUATED_PREDICTIONS_FILE}.")
        save_json_data(EVALUATED_PREDICTIONS_FILE, list(evaluated_predictions_dict.values()))
        # Update the fingerprint only if we actually processed and potentially saved changes
        write_current_fingerprint(current_summaries_fingerprint)
        logging.info(f"Processed reports. Extracted {new_predictions_extracted_this_run} new predictions. Sent {predictions_sent_to_gemini_this_run} requests to Gemini.")
    else:
        logging.info("No new predictions extracted and no existing evaluations were modified. Evaluations file remains unchanged.")
        # Do not update the fingerprint here, so next run re-evaluates if summaries are still the same "new"
        # Or, if we consider a full pass without changes as "processed", then update fingerprint.
        # For now, let's assume if no data changed in EVALUATED_PREDICTIONS_FILE, we might want to re-run if summaries haven't changed.
        # A better approach: only update fingerprint if made_changes_to_evaluations is true.
        if not (new_predictions_extracted_this_run == 0 and predictions_sent_to_gemini_this_run == 0):
             # This case implies we ran through, might have updated checked_against_reports or something minor
             # If we want to ensure we don't re-process this same state of summaries.json unless it *truly* changes,
             # we should update the fingerprint here.
             write_current_fingerprint(current_summaries_fingerprint)


    # --- Analysis (same as before) ---
    logging.info("\n--- Event Prediction Accuracy Analysis (with Priors) ---")
    # (Analysis code remains the same)
    final_predictions_list = list(evaluated_predictions_dict.values())
    total_preds = len(final_predictions_list)
    
    verified_occurred = [p for p in final_predictions_list if p['status'] == 'verified_occurred']
    verified_not_occurred = [p for p in final_predictions_list if p['status'] == 'verified_not_occurred']
    pending = [p for p in final_predictions_list if p['status'] == 'pending_verification']
    
    preds_with_priors = [p for p in final_predictions_list if p.get("estimated_prior_probability") is not None]

    logging.info(f"Total Predictions Processed: {total_preds}")
    logging.info(f"  Verified Occurred: {len(verified_occurred)}")
    logging.info(f"  Verified Not Occurred: {len(verified_not_occurred)}")
    logging.info(f"  Pending Verification: {len(pending)}")
    logging.info(f"  Predictions with Prior Estimates: {len(preds_with_priors)}")


    if (len(verified_occurred) + len(verified_not_occurred)) > 0:
        hit_rate = len(verified_occurred) / (len(verified_occurred) + len(verified_not_occurred))
        logging.info(f"Overall Hit Rate (Occurred / (Occurred + Not Occurred)): {hit_rate:.2%}")

    brier_scores = []
    for p in preds_with_priors:
        if p['status'] == 'verified_occurred':
            outcome = 1.0
            brier_scores.append((p["estimated_prior_probability"] - outcome)**2)
        elif p['status'] == 'verified_not_occurred':
            outcome = 0.0
            brier_scores.append((p["estimated_prior_probability"] - outcome)**2)
    
    if brier_scores:
        avg_brier_score = sum(brier_scores) / len(brier_scores)
        logging.info(f"Average Brier Score (for {len(brier_scores)} evaluated preds with priors): {avg_brier_score:.4f} (lower is better, 0 is perfect)")
    
    if verified_occurred and preds_with_priors:
        avg_prior_for_hits = sum(p["estimated_prior_probability"] for p in verified_occurred if p.get("estimated_prior_probability") is not None and len([pr for pr in verified_occurred if pr.get("estimated_prior_probability") is not None]) > 0) / \
                             max(1, sum(1 for p in verified_occurred if p.get("estimated_prior_probability") is not None)) # Avoid division by zero
        logging.info(f"Average Estimated Prior Probability for 'Occurred' events: {avg_prior_for_hits:.2f}")
    
    if verified_not_occurred and preds_with_priors:
        avg_prior_for_misses = sum(p["estimated_prior_probability"] for p in verified_not_occurred if p.get("estimated_prior_probability") is not None and len([pr for pr in verified_not_occurred if pr.get("estimated_prior_probability") is not None]) > 0) / \
                               max(1, sum(1 for p in verified_not_occurred if p.get("estimated_prior_probability") is not None)) # Avoid division by zero
        logging.info(f"Average Estimated Prior Probability for 'Not Occurred' events: {avg_prior_for_misses:.2f}")

    surprising_hits = sorted([
        p for p in verified_occurred 
        if p.get("estimated_prior_probability") is not None and p["estimated_prior_probability"] < 0.3
    ], key=lambda x: x["estimated_prior_probability"])

    if surprising_hits:
        logging.info(f"\n--- Surprising Correct Predictions (Prior < 0.3 and Occurred) ---")
        for p in surprising_hits:
            logging.info(f"  Prior: {p['estimated_prior_probability']:.2f} ({p['estimated_prior_likelihood_category']}) - Text: {p['prediction_text'][:100]}...")


if __name__ == "__main__":
    main()