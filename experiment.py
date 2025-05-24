import json
import os
import datetime
import re
import logging
import sys
from dateutil import parser as date_parser
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
SUMMARIES_FILE = 'summaries.json' # Your input file with weekly summaries
EVALUATED_PREDICTIONS_FILE = 'predictions.json' # Output file for evaluated predictions
PREDICTION_WINDOW_DAYS = 7
PREDICTION_DELIMITER = "In the near future, "
ALT_DELIMITERS = ["Outlook:", "Looking ahead,"]
LAST_RUN_TIMESTAMP_FILE = 'timestamps.txt' # File to track last processed summary file state

# --- Load Environment Variables ---
load_dotenv()

# --- Gemini Configuration ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set or found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Or your preferred model
    GENERATION_CONFIG = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 768,
    }
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}, # Or BLOCK_NONE if truly necessary and understood
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}, # Keep this one stricter
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

# --- Helper Functions ---
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

    primary_match = re.search(re.escape(PREDICTION_DELIMITER), summary_text, re.IGNORECASE)
    if primary_match:
        delimiter_to_use = primary_match.group(0)
    else:
        for alt_delim_pattern in ALT_DELIMITERS:
            alt_match = re.search(re.escape(alt_delim_pattern), summary_text, re.IGNORECASE)
            if alt_match:
                delimiter_to_use = alt_match.group(0)
                break
    
    if delimiter_to_use:
        try:
            parts = summary_text.split(delimiter_to_use, 1)
            past_events_text = parts[0].strip()
            if len(parts) > 1:
                full_prediction_block = parts[1].strip()
                raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])', full_prediction_block)
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
    return f"{report_ts}_{abs(hash(prediction_text))}"

def get_keywords(text):
    if not text: return set()
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "will", "be", "to", "of", "and", "in", "on", "it", "for", "with", "this", "that", "as", "at", "by", "likely", "less", "may", "might", "could", "would", "should"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = {word for word in words if word not in stop_words and not word.isdigit()}
    return keywords

def check_prediction_occurrence(prediction_text, future_report_past_events_text, threshold=0.25):
    if not prediction_text or not future_report_past_events_text:
        return False, 0.0, []
    pred_keywords = get_keywords(prediction_text)
    event_keywords = get_keywords(future_report_past_events_text)

    if not pred_keywords:
        return False, 0.0, []

    common_keywords = pred_keywords.intersection(event_keywords)
    score = len(common_keywords) / len(pred_keywords) if pred_keywords else 0

    if score >= threshold and len(common_keywords) >= 2 :
        return True, score, sorted(list(common_keywords))
    return False, score, sorted(list(common_keywords))

# In _call_gemini_with_json_parsing function:

def _call_gemini_with_json_parsing(prompt_text, log_prefix="Gemini"):
    try:
        logging.info(f"{log_prefix}: Sending prompt to Gemini...")
        # Ensure stream=False if you're not explicitly handling streaming
        response = gemini_model.generate_content(prompt_text, stream=False) 
        
        # --- New Detailed Response Inspection ---
        if not response.candidates:
            logging.error(f"{log_prefix}: No candidates returned from Gemini. Full response: {response}")
            return {"error": "no_candidates", "message": "No candidates in response."}

        candidate = response.candidates[0] # Assuming one candidate usually

        if candidate.finish_reason.name != "STOP": # Check if generation stopped normally
            logging.warning(
                f"{log_prefix}: Generation did not finish normally. "
                f"Finish Reason: {candidate.finish_reason.name} ({candidate.finish_reason.value}). "
                f"Safety Ratings: {[str(rating) for rating in candidate.safety_ratings]}"
            )
            # If blocked by safety, .text will fail
            if candidate.finish_reason.name == "SAFETY":
                 return {"error": "blocked_by_safety", 
                         "message": f"Blocked by safety. Ratings: {[str(rating) for rating in candidate.safety_ratings]}"}
            elif candidate.finish_reason.name in ["MAX_TOKENS", "RECITATION", "OTHER"]:
                 # If max tokens, there might still be partial content, but .text might still fail if no valid parts
                 # For other reasons, it's also likely no usable text.
                 if not candidate.content.parts:
                    logging.error(f"{log_prefix}: No content parts returned. Finish Reason: {candidate.finish_reason.name}")
                    return {"error": "no_content_parts", 
                            "message": f"No content parts. Finish Reason: {candidate.finish_reason.name}"}


        # Check if there are content parts before accessing .text
        if not candidate.content or not candidate.content.parts:
            logging.error(f"{log_prefix}: No content parts in candidate even if finish_reason was STOP. Candidate: {candidate}")
            # This could happen if the model generated an empty response for some reason.
            return {"error": "empty_content_parts", "message": "Candidate has no content parts despite STOP finish reason."}
        # --- End of New Detailed Response Inspection ---

        # Now, it should be safer to access .text, but we'll keep the original try-except for .text just in case
        try:
            cleaned_text = response.text.strip() # This is what was failing
        except ValueError as ve: # Catch the specific error we saw
            logging.error(f"{log_prefix}: Error accessing response.text: {ve}. Candidate Finish Reason: {candidate.finish_reason.name}. Safety Ratings: {[str(rating) for rating in candidate.safety_ratings]}")
            return {"error": "text_accessor_failed", "message": str(ve)}


        json_str = None
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            first_brace = cleaned_text.find('{')
            last_brace = cleaned_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = cleaned_text[first_brace : last_brace+1]
            else:
                if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                    json_str = cleaned_text

        if not json_str:
            logging.warning(f"{log_prefix}: Could not extract JSON from Gemini response. Raw: {cleaned_text[:500]}")
            return None # Keep this as None to indicate parsing failure, distinct from API errors

        parsed_answer = json.loads(json_str)
        return parsed_answer

    except json.JSONDecodeError as jde:
        # (rest of the error handling remains the same)
        logging.error(f"{log_prefix}: Failed to parse JSON from Gemini response: {jde}. JSON string attempted: '{json_str if 'json_str' in locals() and json_str else 'N/A'}' Raw response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
        return None
    # except genai.types.generation_types.BlockedPromptException as bpe: # This is less likely if finish_reason is SAFETY
    #     logging.error(f"{log_prefix}: Gemini prompt was blocked before generation: {bpe}.")
    #     return {"error": "blocked_prompt_exception", "message": str(bpe)}
    except Exception as e:
        if hasattr(e, 'message') and "API key not valid" in str(e.message):
            logging.error(f"{log_prefix}: Gemini API Key Error: {e}")
            sys.exit("Exiting due to Gemini API Key error.")
        # Check for common API call related errors, e.g., permission denied, quota exceeded
        # These might not be caught by specific Gemini exceptions but general `google.api_core.exceptions`
        logging.error(f"{log_prefix}: Gemini request failed with unhandled exception: {e}", exc_info=True)
        return {"error": "unhandled_request_failed", "message": str(e)}
    
def check_prediction_vacuousness_gemini(prediction_text, context_summary_text, report_date_str):
    prompt = f"""
    Context: You are an AI assistant evaluating the quality of a geo-political/economic prediction.
    Task: Assess if the given prediction is specific, informative, and testable, or if it's too vague, obvious given the context, generally uninformative, or essentially untestable.
    A prediction is "obvious or vacuous" if it:
    1. States something almost always true (e.g., "markets will fluctuate").
    2. Lacks specific actors, actions, or measurable outcomes.
    3. Simply rephrases an ongoing situation from the provided context without predicting a distinct *new* development or change.
    4. Is so broad that almost any event could be seen as confirming it.

    Prediction to evaluate: "{prediction_text}"

    Supporting context (summary of world events from the week ending {report_date_str} that led to this prediction):
    --- CONTEXT START ---
    {context_summary_text}
    --- CONTEXT END ---

    Output your assessment STRICTLY in JSON format with the following keys:
    - "is_obvious_or_vacuous": boolean (true if obvious/vacuous/untestable, false otherwise).
    - "obviousness_score": float (0.0 for highly specific/non-obvious, 1.0 for extremely obvious/vacuous).
    - "reasoning": string (brief explanation).
    """
    parsed_answer = _call_gemini_with_json_parsing(prompt, log_prefix=f"VacuousnessCheck for '{prediction_text[:30]}...'")
    if parsed_answer and "error" not in parsed_answer:
        if not all(k in parsed_answer for k in ["is_obvious_or_vacuous", "obviousness_score", "reasoning"]):
            logging.warning(f"Vacuousness check: Gemini response missing required keys. Parsed: {parsed_answer}")
            return None
        return parsed_answer
    return None

def get_prior_likelihood_from_gemini(prediction_text, context_summary_text, report_date_str):
    prompt = f"""
    Context: You are an AI assistant estimating the likelihood of a geo-political/economic prediction.
    Task: Based *only* on the provided context (world events from the week ending {report_date_str}) and general knowledge available up to that date, estimate the prior likelihood of the specific prediction coming true in the near future (next 1-2 weeks).

    Prediction to evaluate: "{prediction_text}"

    Supporting context (summary of world events from the week ending {report_date_str}):
    --- CONTEXT START ---
    {context_summary_text}
    --- CONTEXT END ---

    Output your estimation STRICTLY in JSON format with the following keys:
    - "prior_probability": float (numerical probability from 0.0 to 1.0).
    - "likelihood_category": string (e.g., "Very Low", "Low", "Moderate", "High", "Very High").
    - "rationale": string (brief explanation for your estimation, 1-2 sentences).
    """
    parsed_answer = _call_gemini_with_json_parsing(prompt, log_prefix=f"PriorLikelihood for '{prediction_text[:30]}...'")
    if parsed_answer and "error" not in parsed_answer:
        if not all(k in parsed_answer for k in ["prior_probability", "likelihood_category", "rationale"]):
            logging.warning(f"Prior likelihood: Gemini response missing required keys. Parsed: {parsed_answer}")
            return None
        prob = parsed_answer["prior_probability"]
        if isinstance(prob, str):
            try: prob = float(prob)
            except ValueError: prob = -1.0
        if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
            logging.warning(f"Invalid prior_probability from Gemini: {prob}")
            if isinstance(parsed_answer["prior_probability"], str) and parsed_answer["prior_probability"].endswith('%'):
                try:
                    prob_val = float(parsed_answer["prior_probability"].rstrip('%')) / 100.0
                    if 0.0 <= prob_val <= 1.0:
                        parsed_answer["prior_probability"] = prob_val
                    else: return None
                except ValueError: return None
            else: return None
        return parsed_answer
    return None

def get_file_fingerprint(filepath):
    if not os.path.exists(filepath): return (0, 0)
    stat = os.stat(filepath)
    return (stat.st_mtime, stat.st_size)

def read_last_run_fingerprint():
    if os.path.exists(LAST_RUN_TIMESTAMP_FILE):
        with open(LAST_RUN_TIMESTAMP_FILE, 'r') as f:
            try:
                mtime, size = f.read().strip().split(',')
                return (float(mtime), int(size))
            except ValueError: return (0,0)
    return (0, 0)

def write_current_fingerprint(fingerprint):
    with open(LAST_RUN_TIMESTAMP_FILE, 'w') as f:
        f.write(f"{fingerprint[0]},{fingerprint[1]}")

# Make sure to include all necessary imports at the top of your script:
# import json, os, datetime, re, logging, sys
# from dateutil import parser as date_parser
# import google.generativeai as genai
# from dotenv import load_dotenv

# ... (all your helper functions: load_json_data, save_json_data, parse_report_summary, etc. up to get_file_fingerprint) ...

# --- Helper to check if a Gemini-dependent stage is complete or terminally failed ---
def _gemini_check_previously_attempted_and_concluded(check_result_dict):
    """
    Checks if a Gemini check (vacuousness or prior) has a conclusive result.
    Conclusive means either successful data or a terminal error state from Gemini (like 'blocked').
    It returns False if the field is None (not attempted) or has a transient error
    that might warrant a retry (though current script doesn't implement auto-retry for transient).
    """
    if check_result_dict is None:
        return False # Not attempted
    if "error" in check_result_dict:
        # Define which errors are considered "terminal" for a specific check, meaning we shouldn't retry.
        # 'blocked_by_safety' is a good candidate for terminal.
        # 'request_failed' might be transient or permanent depending on the cause.
        # For simplicity now, if there's an error field, we consider the attempt made.
        # A more nuanced approach could check specific error types.
        return True # Attempt was made, resulted in an error we've recorded.
    # If no error key and it's not None, it means it was successful.
    return True


# --- Main Script Execution ---
def main():
    if not os.path.exists(SUMMARIES_FILE):
        logging.info(f"{SUMMARIES_FILE} not found. Exiting.")
        return

    current_summaries_fingerprint = get_file_fingerprint(SUMMARIES_FILE)
    last_run_summaries_fingerprint = read_last_run_fingerprint()

    if current_summaries_fingerprint == last_run_summaries_fingerprint:
        logging.info(f"No new changes in {SUMMARIES_FILE} since last successful processing. Exiting.")
        return
    
    logging.info(f"Detected changes in {SUMMARIES_FILE} or first run. Starting processing...")

    all_summaries = load_json_data(SUMMARIES_FILE)
    if not all_summaries:
        logging.info(f"No summaries loaded from {SUMMARIES_FILE} despite file change. Exiting.")
        write_current_fingerprint(current_summaries_fingerprint)
        return

    all_summaries.sort(key=lambda x: date_parser.isoparse(x['timestamp']))
    
    evaluated_predictions_dict = {p['prediction_id']: p for p in load_json_data(EVALUATED_PREDICTIONS_FILE)}
    
    new_predictions_this_run = 0
    gemini_vacuous_calls_this_run = 0
    gemini_prior_calls_this_run = 0
    predictions_filtered_as_obvious_this_run = 0
    made_eval_data_changes_in_run = False # Overall flag for the entire run

    for i, report_n in enumerate(all_summaries):
        report_n_timestamp_str = report_n['timestamp']
        report_n_dt = date_parser.isoparse(report_n_timestamp_str)
        report_n_past_events_text, report_n_predictions_list = parse_report_summary(report_n['summary'])

        if not report_n_predictions_list: continue

        for pred_text in report_n_predictions_list:
            if not pred_text.strip() or len(pred_text.strip().split()) < 4:
                continue 
            
            pred_id = generate_prediction_id(report_n_timestamp_str, pred_text)
            
            this_pred_entry_modified_in_run = False # Flag for changes to this specific prediction entry

            if pred_id not in evaluated_predictions_dict:
                new_predictions_this_run += 1
                this_pred_entry_modified_in_run = True
                current_eval_entry = {
                    "prediction_id": pred_id, "source_report_timestamp": report_n_timestamp_str,
                    "prediction_text": pred_text,
                    "predicted_period_start": (report_n_dt).isoformat(),
                    "predicted_period_end": (report_n_dt + datetime.timedelta(days=PREDICTION_WINDOW_DAYS)).isoformat(),
                    "status": "pending_processing", 
                    "vacuousness_check": None, "prior_likelihood": None, "verification_details": {}
                }
                evaluated_predictions_dict[pred_id] = current_eval_entry
            else:
                current_eval_entry = evaluated_predictions_dict[pred_id]

            # --- Stage 1: Vacuousness Check ---
            # Call Gemini only if this check hasn't been conclusively done before.
            if not _gemini_check_previously_attempted_and_concluded(current_eval_entry.get("vacuousness_check")):
                logging.info(f"Running vacuousness check for {pred_id} (not previously concluded)...")
                vac_check_result = check_prediction_vacuousness_gemini(pred_text, report_n_past_events_text, report_n_timestamp_str)
                gemini_vacuous_calls_this_run += 1
                this_pred_entry_modified_in_run = True # Gemini call means potential modification

                if vac_check_result:
                    current_eval_entry["vacuousness_check"] = vac_check_result
                    if vac_check_result.get("is_obvious_or_vacuous") is True: # Explicitly check for boolean True
                        current_eval_entry["status"] = "filtered_obvious"
                        predictions_filtered_as_obvious_this_run += 1
                        logging.info(f"Prediction {pred_id} filtered as obvious/vacuous. Reason: {vac_check_result.get('reasoning')}")
                    elif vac_check_result.get("is_obvious_or_vacuous") is False: # Explicitly check for boolean False
                         current_eval_entry["status"] = "pending_prior_estimation"
                    else: # Neither True nor False, likely an error in Gemini's output structure or our parsing
                        current_eval_entry["status"] = "error_vacuousness_check_parsing"
                        logging.warning(f"Vacuousness check for {pred_id} returned unexpected structure: {vac_check_result}")
                else: # Gemini call itself failed (e.g. API error, no JSON returned)
                    current_eval_entry["vacuousness_check"] = {"error": "Gemini call failed or no valid JSON"}
                    current_eval_entry["status"] = "error_vacuousness_check_api"
                evaluated_predictions_dict[pred_id] = current_eval_entry
            
            # --- Stage 2: Prior Likelihood Estimation ---
            # Proceed if status is 'pending_prior_estimation' AND prior check hasn't been conclusively done.
            if current_eval_entry["status"] == "pending_prior_estimation" and \
               not _gemini_check_previously_attempted_and_concluded(current_eval_entry.get("prior_likelihood")):
                logging.info(f"Estimating prior likelihood for {pred_id} (not previously concluded)...")
                prior_result = get_prior_likelihood_from_gemini(pred_text, report_n_past_events_text, report_n_timestamp_str)
                gemini_prior_calls_this_run += 1
                this_pred_entry_modified_in_run = True # Gemini call means potential modification

                if prior_result:
                    current_eval_entry["prior_likelihood"] = prior_result
                    if "error" not in prior_result: # Check if Gemini returned an error object itself
                        current_eval_entry["status"] = "pending_verification"
                    else: # Gemini call worked but returned an error structure from our helper
                        current_eval_entry["status"] = "error_prior_estimation_gemini_reported"
                        logging.warning(f"Prior estimation for {pred_id} returned error structure: {prior_result}")

                else: # Gemini call itself failed (e.g. API error, no JSON returned)
                    current_eval_entry["prior_likelihood"] = {"error": "Gemini call failed or no valid JSON"}
                    current_eval_entry["status"] = "error_prior_estimation_api"
                evaluated_predictions_dict[pred_id] = current_eval_entry
            
            # --- Stage 3: Verification ---
            if current_eval_entry["status"] == "pending_verification" or current_eval_entry["status"] == "verified_not_occurred":
                # Store original state to detect changes within this verification block
                original_status = current_eval_entry["status"]
                original_verification_details_str = json.dumps(current_eval_entry.get("verification_details", {}), sort_keys=True)


                current_pred_start_dt = date_parser.isoparse(current_eval_entry["predicted_period_start"])
                current_pred_end_dt = date_parser.isoparse(current_eval_entry["predicted_period_end"])
                
                current_eval_entry.setdefault("verification_details", {
                    "checked_against_reports": [], "evidence_report_timestamp": None,
                    "evidence_text_snippet": None, "match_score": 0.0,
                    "matching_keywords": [], "notes": "", "is_prime_match": False
                })
                
                found_match_in_this_verification_pass = False
                best_match_score_this_pred = current_eval_entry["verification_details"].get("match_score", 0.0)
                # Ensure 'checked_against_reports' is a list
                if not isinstance(current_eval_entry["verification_details"]["checked_against_reports"], list):
                    current_eval_entry["verification_details"]["checked_against_reports"] = []


                for j in range(i + 1, len(all_summaries)):
                    report_n_plus_k = all_summaries[j]
                    report_n_plus_k_timestamp_str = report_n_plus_k['timestamp']
                    report_n_plus_k_dt = date_parser.isoparse(report_n_plus_k_timestamp_str)

                    if report_n_plus_k_dt < current_pred_start_dt: continue
                    
                    # If already solidly verified_occurred, only add to checked_against_reports if new, don't change status
                    if current_eval_entry["status"] == "verified_occurred":
                        if report_n_plus_k_timestamp_str not in current_eval_entry["verification_details"]["checked_against_reports"]:
                            current_eval_entry["verification_details"]["checked_against_reports"].append(report_n_plus_k_timestamp_str)
                            this_pred_entry_modified_in_run = True
                        continue # Don't re-evaluate matching if already occurred

                    # If not yet verified occurred, or was not_occurred, proceed with matching
                    if report_n_plus_k_timestamp_str not in current_eval_entry["verification_details"]["checked_against_reports"]:
                         current_eval_entry["verification_details"]["checked_against_reports"].append(report_n_plus_k_timestamp_str)
                         this_pred_entry_modified_in_run = True # Adding a checked report is a modification

                    future_report_past_events, _ = parse_report_summary(report_n_plus_k['summary'])
                    if not future_report_past_events: continue

                    is_prime_candidate = current_pred_end_dt <= report_n_plus_k_dt < (current_pred_end_dt + datetime.timedelta(days=PREDICTION_WINDOW_DAYS/2 + 3))
                    occurred, score, common_kws = check_prediction_occurrence(pred_text, future_report_past_events)

                    if occurred:
                        if score > best_match_score_this_pred or \
                           (score == best_match_score_this_pred and is_prime_candidate and not current_eval_entry["verification_details"].get("is_prime_match", False)):
                            found_match_in_this_verification_pass = True # Indicates a match was found or improved in this pass
                            best_match_score_this_pred = score
                            current_eval_entry["status"] = "verified_occurred"
                            current_eval_entry["verification_details"]["evidence_report_timestamp"] = report_n_plus_k_timestamp_str
                            current_eval_entry["verification_details"]["evidence_text_snippet"] = future_report_past_events[:600] + ("..." if len(future_report_past_events) > 600 else "")
                            current_eval_entry["verification_details"]["match_score"] = round(score, 3)
                            current_eval_entry["verification_details"]["matching_keywords"] = common_kws
                            current_eval_entry["verification_details"]["is_prime_match"] = is_prime_candidate
                            current_eval_entry["verification_details"]["notes"] = f"Matched in report {report_n_plus_k_timestamp_str}."
                            this_pred_entry_modified_in_run = True
                            if is_prime_candidate and score > 0.4:
                                break 
                
                # If no match was found/improved in this pass AND status is still pending_verification
                if not found_match_in_this_verification_pass and current_eval_entry["status"] == "pending_verification":
                    last_summary_dt = date_parser.isoparse(all_summaries[-1]['timestamp'])
                    if last_summary_dt > current_pred_end_dt + datetime.timedelta(days=PREDICTION_WINDOW_DAYS + 3):
                         current_eval_entry["status"] = "verified_not_occurred"
                         current_eval_entry["verification_details"]["notes"] = "No strong match found in subsequent reports after sufficient time and checks."
                         this_pred_entry_modified_in_run = True # Status change is a modification
                
                # More robust check for modifications in this verification block
                current_verification_details_str = json.dumps(current_eval_entry.get("verification_details", {}), sort_keys=True)
                if current_eval_entry["status"] != original_status or \
                   current_verification_details_str != original_verification_details_str:
                    this_pred_entry_modified_in_run = True
                
                evaluated_predictions_dict[pred_id] = current_eval_entry
            
            if this_pred_entry_modified_in_run:
                made_eval_data_changes_in_run = True # If any prediction entry was modified, flag the overall run

    # --- End of loops ---

    if made_eval_data_changes_in_run:
        logging.info(f"Saving updated evaluations to {EVALUATED_PREDICTIONS_FILE}.")
        save_json_data(EVALUATED_PREDICTIONS_FILE, list(evaluated_predictions_dict.values()))
    else:
        logging.info(f"No significant changes made to evaluation data. {EVALUATED_PREDICTIONS_FILE} remains unchanged.")

    write_current_fingerprint(current_summaries_fingerprint)
    logging.info(f"Processing complete. New predictions: {new_predictions_this_run}. Filtered as obvious: {predictions_filtered_as_obvious_this_run}. Gemini Vacuousness Calls: {gemini_vacuous_calls_this_run}. Gemini Prior Calls: {gemini_prior_calls_this_run}.")

    # --- Analysis Section (remains largely the same as before) ---
    logging.info(f"\n--- Event Prediction Accuracy Analysis (Filtered, with Priors) ---")
    # Filter out statuses that are not yet ready for final evaluation metrics
    final_predictions_list = [p for p in evaluated_predictions_dict.values() if p['status'] not in [
        "filtered_obvious", "pending_processing", 
        "error_vacuousness_check_api", "error_vacuousness_check_parsing", "error_vacuousness_check_gemini_reported",
        "pending_prior_estimation", 
        "error_prior_estimation_api", "error_prior_estimation_gemini_reported"
    ]]
    
    total_evaluable = len(final_predictions_list) # These are predictions that *should* have a prior (if not error) and are ready for verification outcomes
    verified_occurred = [p for p in final_predictions_list if p['status'] == 'verified_occurred']
    verified_not_occurred = [p for p in final_predictions_list if p['status'] == 'verified_not_occurred']
    pending_verification = [p for p in final_predictions_list if p['status'] == 'pending_verification']
    
    logging.info(f"Total Evaluable Predictions (passed filters, ready for/in verification): {total_evaluable}")
    logging.info(f"  Verified Occurred: {len(verified_occurred)}")
    logging.info(f"  Verified Not Occurred: {len(verified_not_occurred)}")
    logging.info(f"  Still Pending Verification (need more future reports): {len(pending_verification)}")

    num_definitively_evaluated = len(verified_occurred) + len(verified_not_occurred)
    if num_definitively_evaluated > 0:
        hit_rate = len(verified_occurred) / num_definitively_evaluated
        logging.info(f"Hit Rate (Occurred / Definitively Evaluated): {hit_rate:.2%}")
    else:
        logging.info("Not enough definitively evaluated predictions for a hit rate.")

    brier_scores = []
    # Brier score only for those that have a prior_probability AND a definitive outcome
    preds_for_brier = [
        p for p in final_predictions_list 
        if p.get("prior_likelihood") and \
           isinstance(p["prior_likelihood"].get("prior_probability"), (float, int)) and \
           p["status"] in ["verified_occurred", "verified_not_occurred"]
    ]
    
    for p in preds_for_brier:
        prior_prob = p["prior_likelihood"]["prior_probability"]
        outcome = 1.0 if p['status'] == 'verified_occurred' else 0.0
        brier_scores.append((prior_prob - outcome)**2)
    
    if brier_scores:
        avg_brier_score = sum(brier_scores) / len(brier_scores)
        logging.info(f"Average Brier Score (for {len(brier_scores)} preds with priors & outcomes): {avg_brier_score:.4f} (lower is better)")
    else:
        logging.info("No predictions available for Brier score calculation.")

    # Average Priors for Occurred vs. Not Occurred
    priors_for_occurred = [
        p["prior_likelihood"]["prior_probability"] for p in verified_occurred 
        if p.get("prior_likelihood") and isinstance(p["prior_likelihood"].get("prior_probability"), (float, int))
    ]
    priors_for_not_occurred = [
        p["prior_likelihood"]["prior_probability"] for p in verified_not_occurred 
        if p.get("prior_likelihood") and isinstance(p["prior_likelihood"].get("prior_probability"), (float, int))
    ]

    if priors_for_occurred:
        logging.info(f"Average Estimated Prior for 'Occurred' events ({len(priors_for_occurred)} preds): {sum(priors_for_occurred)/len(priors_for_occurred):.2f}")
    if priors_for_not_occurred:
        logging.info(f"Average Estimated Prior for 'Not Occurred' events ({len(priors_for_not_occurred)} preds): {sum(priors_for_not_occurred)/len(priors_for_not_occurred):.2f}")

    # Surprising Correct Predictions (Low Prior, Occurred)
    surprising_hits = sorted([
        p for p in verified_occurred 
        if p.get("prior_likelihood") and \
           isinstance(p["prior_likelihood"].get("prior_probability"), (float, int)) and \
           p["prior_likelihood"]["prior_probability"] < 0.35 
    ], key=lambda x: x["prior_likelihood"]["prior_probability"])

    if surprising_hits:
        logging.info(f"\n--- Surprising Correct Predictions (Prior < 0.35 and Occurred) ---")
        for p_idx, p_item in enumerate(surprising_hits[:5]):
            logging.info(f"  {p_idx+1}. Prior: {p_item['prior_likelihood']['prior_probability']:.2f} ({p_item['prior_likelihood']['likelihood_category']}) - Text: {p_item['prediction_text'][:100]}...")
    else:
        logging.info("No 'surprising correct predictions' (low prior, occurred) found with current criteria.")

# Ensure all other functions (load_json_data, save_json_data, parse_report_summary, 
# generate_prediction_id, get_keywords, check_prediction_occurrence, 
# _call_gemini_with_json_parsing, check_prediction_vacuousness_gemini, 
# get_prior_likelihood_from_gemini, get_file_fingerprint, 
# read_last_run_fingerprint, write_current_fingerprint) are defined above main().

if __name__ == "__main__":
    main()