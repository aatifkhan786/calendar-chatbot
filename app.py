# --- Standard Libraries ---
import os
import datetime
import json
import logging
import traceback
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
import time # Import time for potential debugging waits

# --- Flask ---
from flask import Flask, request, jsonify, render_template, session, redirect, url_for

# --- Environment Variables ---
from dotenv import load_dotenv

# --- Google OAuth & API ---
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.api_core import exceptions as google_exceptions # For refresh errors etc.

# --- Google Generative AI (Gemini) ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For Safety Settings

# --- Basic Configuration ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s') # DEBUG level to see more
load_dotenv()

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a_default_fallback_secret_key_for_dev')
if app.secret_key == 'a_default_fallback_secret_key_for_dev':
    logging.warning("Using default Flask secret key. Set FLASK_SECRET_KEY environment variable for production.")

# --- Define System Prompt String (Minor update to acknowledge history) ---
today_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
GEMINI_SYSTEM_INSTRUCTION = f"""
You are an intelligent assistant specialized ONLY in managing Google Calendar events based on user requests. Your primary capabilities are:
1.  **Schedule Meeting:** Create a new calendar event. Extract 'summary' (title), 'date' (e.g., 'today', 'tomorrow', 'August 15th', '2024-08-15'), 'time' (e.g., '3 PM', '15:00'), 'attendees' (list of emails), and 'duration_minutes' (integer, default to 60 if not specified).
    *   **Crucially, try to resolve the combined date and time into a specific start datetime.** Return this as 'start_datetime_iso' (e.g., '2024-08-15T15:00:00') based on today's date ({today_str}, assume UTC if no timezone specified by user). If you cannot determine a specific time, return `null` for 'start_datetime_iso'.
2.  **List Events:** Show upcoming events. Extract 'date' (target date or range, e.g., 'today', 'tomorrow', 'next week', 'on 2024-08-20'). Default to 'today' if not specified.

You MUST determine if the user's request falls into one of the **allowed capabilities**. You cannot perform other actions like checking weather, telling jokes, canceling/updating events yet, etc.

**You will be provided with the recent conversation history. Use this context to better understand the current user query, especially if it's a follow-up question.**

**Response Format:**
- Respond ONLY with a valid JSON object. NO extra text or markdown ```json ... ```.
- **If the request IS within scope (scheduling or listing):**
    - Identify the primary 'intent' ('schedule_meeting' or 'list_events').
    - Extract relevant 'entities' as key-value pairs. Use `null` if an entity is not found.
    - Return: `{{"in_scope": true, "intent": "identified_intent", "entities": {{"summary": "...", "date": "...", "time": "...", "attendees": ["..."], "duration_minutes": ..., "start_datetime_iso": "..."}}}}` (or null)
- **If the request is OUTSIDE the scope:**
    - Return: `{{"in_scope": false, "intent": "out_of_scope", "entities": {{}}, "reply": "I can only help with scheduling new meetings or listing events on your Google Calendar."}}`
- **If the request is IN scope but AMBIGUOUS or lacks crucial info (use history for context first):**
    - Return: `{{"in_scope": true, "intent": "clarification_needed", "entities": {{}}, "reply": "Okay, I can help with that. Could you please provide [Missing Information e.g., the date and time]?"}}` (Customize the reply).

Today's date (UTC) is: {today_str}. Be precise. For attendees, return a list of strings that look like email addresses.
"""

# --- Google Generative AI (Gemini) Configuration ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=google_api_key)
    # Specify safety settings to be less restrictive if needed, otherwise use defaults
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    # Specify generation config - ensure JSON output
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,
        response_mime_type="application/json"
    )
    gemini_model = genai.GenerativeModel(
        'gemini-1.5-flash-latest', # Use a capable model
        system_instruction=GEMINI_SYSTEM_INSTRUCTION, # Pass system instruction here
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logging.info("Google Generative AI client configured successfully with system instruction.")
except Exception as e:
    logging.critical(f"FATAL: Failed to configure Google Generative AI client: {e}")
    gemini_model = None

# --- Google OAuth Configuration (Same as before) ---
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', "http://127.0.0.1:5000/oauth2callback")
CLIENT_SECRETS_CONFIG = {
    "web": {
        "client_id": GOOGLE_CLIENT_ID,
        "project_id": os.getenv('GOOGLE_PROJECT_ID'),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI, "http://localhost:5000/oauth2callback"]
    }
}
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
]
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
     logging.critical("FATAL: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set. OAuth will fail.")

# --- Chat History Configuration ---
MAX_HISTORY_TURNS = 10 # Store last 10 turns (5 user, 5 model)


# --- Helper Functions (Unchanged from your provided code) ---

def get_google_auth_flow():
    """Creates and returns a Google OAuth Flow instance."""
    # (Same as before)
    if not CLIENT_SECRETS_CONFIG['web']['client_id'] or not CLIENT_SECRETS_CONFIG['web']['client_secret']:
        logging.error("Cannot create OAuth Flow: Client ID or Secret is missing.")
        return None
    if REDIRECT_URI not in CLIENT_SECRETS_CONFIG['web']['redirect_uris']:
        logging.warning(f"REDIRECT_URI ('{REDIRECT_URI}') not explicitly listed in CLIENT_SECRETS_CONFIG['web']['redirect_uris']. Adding it.")
        CLIENT_SECRETS_CONFIG['web']['redirect_uris'].append(REDIRECT_URI)
    try:
        flow = Flow.from_client_config(
            client_config=CLIENT_SECRETS_CONFIG,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI)
        return flow
    except Exception as e:
        logging.error(f"Error creating OAuth Flow: {e}")
        return None

def credentials_to_dict(credentials):
    """Converts Google Credentials object to a dictionary for session storage. Ensures expiry is ISO UTC string."""
    # (Same as before)
    expiry_iso = None
    if credentials.expiry:
        try:
            if isinstance(credentials.expiry, str):
                 expiry_dt = dateutil_parser.isoparse(credentials.expiry)
            elif isinstance(credentials.expiry, datetime.datetime):
                 expiry_dt = credentials.expiry
            else:
                 raise ValueError(f"Unexpected expiry type: {type(credentials.expiry)}")

            if expiry_dt.tzinfo is None:
                expiry_dt_aware = expiry_dt.replace(tzinfo=datetime.timezone.utc)
                logging.debug(f"Credentials expiry was naive ({expiry_dt}), assuming UTC for storage.")
            else:
                expiry_dt_aware = expiry_dt.astimezone(datetime.timezone.utc)

            expiry_iso = expiry_dt_aware.isoformat()
            logging.debug(f"Serialized expiry to ISO UTC for session storage: {expiry_iso}")

        except Exception as e:
            logging.error(f"Error processing credentials expiry before saving to session: {e}. Expiry will be None in session dict.")
            expiry_iso = None

    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes,
        'id_token': getattr(credentials, 'id_token', None),
        'expiry': expiry_iso # Store ISO string or None
    }

# --- REVISED Credential Retrieval and Refresh Logic (Unchanged from your provided code) ---
def get_credentials_from_session():
    """
    Retrieves credentials dictionary, performs manual expiry check using stored
    ISO string, attempts proactive refresh if needed, stores updated dict,
    and returns a usable Credentials object *without* passing expiry to its constructor.
    Returns None if authentication is required or refresh fails irrecoverably.
    """
    # (Same as your revised function)
    logging.debug("Attempting to get credentials from session and potentially refresh.")
    credentials_dict = session.get('credentials')
    if not credentials_dict or not credentials_dict.get('token'):
        logging.info("No valid credentials dictionary found in session.")
        session.pop('credentials', None)
        session.pop('chat_history', None) # Also clear history if creds are gone
        return None

    # --- Manual Expiry Check & Proactive Refresh ---
    needs_refresh = False
    can_refresh = bool(credentials_dict.get('refresh_token') and
                       credentials_dict.get('client_id') and
                       credentials_dict.get('client_secret') and
                       credentials_dict.get('token_uri'))

    expiry_str = credentials_dict.get('expiry')
    expiry_dt_utc = None # This is ONLY used for the manual check now

    if expiry_str:
        try:
            expiry_dt_utc = dateutil_parser.isoparse(expiry_str)
            if expiry_dt_utc.tzinfo is None:
                 logging.warning(f"Stored expiry '{expiry_str}' parsed as naive. Assuming UTC for check.")
                 expiry_dt_utc = expiry_dt_utc.replace(tzinfo=datetime.timezone.utc)
            elif expiry_dt_utc.tzinfo != datetime.timezone.utc:
                 expiry_dt_utc = expiry_dt_utc.astimezone(datetime.timezone.utc)

            refresh_threshold = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
            if expiry_dt_utc <= refresh_threshold:
                logging.info(f"Token expiry ({expiry_dt_utc}) is near/past refresh threshold ({refresh_threshold}). Needs refresh.")
                needs_refresh = True
            else:
                logging.debug(f"Token expiry ({expiry_dt_utc}) is okay based on manual check.")

        except Exception as e:
            logging.warning(f"Could not parse or compare stored expiry string '{expiry_str}': {e}. Proceeding without proactive refresh based on time.")
            needs_refresh = False

    # --- Attempt Refresh if Needed and Possible ---
    if needs_refresh and can_refresh:
        logging.info("Attempting proactive token refresh.")
        temp_creds_for_refresh = None
        try:
            # Create a temporary Credentials object for refreshing.
            refresh_dict = credentials_dict.copy()
            refresh_dict.pop('expiry', None) # Remove expiry before passing to this constructor too

            valid_keys_for_refresh = ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret', 'scopes', 'id_token']
            filtered_refresh_dict = {k: v for k, v in refresh_dict.items() if k in valid_keys_for_refresh and v is not None}

            if not filtered_refresh_dict.get('refresh_token'):
                 raise ValueError("Refresh token missing in dict, cannot create object for refresh.")

            logging.debug(f"Creating temporary Credentials object for refresh (keys: {list(filtered_refresh_dict.keys())})")
            temp_creds_for_refresh = Credentials(**filtered_refresh_dict)

            # Perform the refresh
            temp_creds_for_refresh.refresh(GoogleAuthRequest())
            logging.info("Proactive token refresh successful.")

            # Update the original dictionary in memory with refreshed data
            refreshed_data = credentials_to_dict(temp_creds_for_refresh) # This will get the new expiry string
            credentials_dict.update(refreshed_data)
            session['credentials'] = credentials_dict # Save refreshed data back to session
            logging.debug(f"Updated session with refreshed credentials (new expiry: {credentials_dict.get('expiry')})")
            needs_refresh = False # Mark as refreshed

        except google_exceptions.RefreshError as refresh_error:
             logging.error(f"Failed to refresh token proactively (RefreshError): {refresh_error}. Grant might be revoked or refresh token invalid.")
             session.clear() # Clear everything on RefreshError
             logging.info("Cleared session due to RefreshError during proactive refresh.")
             return None
        except Exception as refresh_error:
            logging.error(f"Unexpected error during proactive credential refresh: {refresh_error}", exc_info=True)
            logging.warning("Proceeding with potentially expired token after refresh attempt failed.")
            needs_refresh = False # Avoid retry loop

    elif needs_refresh and not can_refresh:
        logging.warning("Token needs refresh based on stored expiry, but cannot refresh (missing refresh token or client details). API calls may fail.")

    # --- Create Final Credentials Object (WITHOUT passing expiry) ---
    try:
        # Prepare dict from current state (potentially refreshed)
        final_constructor_dict = credentials_dict.copy()

        # *** KEY CHANGE: Remove 'expiry' key before passing to constructor ***
        final_constructor_dict.pop('expiry', None)
        logging.debug("Removed 'expiry' key before creating final Credentials object.")

        # Define keys the Credentials constructor *actually* accepts (excluding expiry now)
        valid_keys = ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret', 'scopes', 'id_token']
        filtered_final_dict = {k: v for k, v in final_constructor_dict.items() if k in valid_keys and v is not None}

        if not filtered_final_dict.get('token'):
             logging.error("Credentials dictionary missing 'token' before final object creation.")
             session.pop('credentials', None)
             session.pop('chat_history', None) # Also clear history
             return None

        logging.debug(f"Calling final Credentials constructor with keys: {list(filtered_final_dict.keys())}")

        # Create the actual Credentials object to be returned
        final_credentials = Credentials(**filtered_final_dict)

        # Check if the object considers itself valid (primarily checks for token existence)
        if not final_credentials.valid:
            logging.warning("Created Credentials object reported as invalid by the library (likely missing token).")
            if not final_credentials.token:
                 logging.error("Final credentials object has no token. Clearing session.")
                 session.pop('credentials', None)
                 session.pop('chat_history', None) # Also clear history
                 return None
            else:
                 logging.warning("Credentials object invalid but token exists? Proceeding cautiously.")


        logging.debug("Successfully created usable Credentials object (without passing expiry).")
        return final_credentials

    except TypeError as e:
        # If TypeError still happens here, it's likely due to another field's type
        logging.error(f"TypeError creating final Credentials object (even without expiry): {e}. Check session data types for other fields.")
        log_cred_err = {k: v for k, v in credentials_dict.items() if k != 'client_secret'}
        logging.error(f"Credentials dict causing error (masked): {log_cred_err}", exc_info=True) # Log traceback
        session.pop('credentials', None)
        session.pop('chat_history', None) # Also clear history
        return None
    except Exception as e:
        logging.error(f"Unexpected error creating final Credentials object: {e}", exc_info=True)
        session.pop('credentials', None)
        session.pop('chat_history', None) # Also clear history
        return None
# --- END OF REVISED Credential Retrieval ---


# --- build_api_service Function (Unchanged from your provided code) ---
def build_api_service(service_name, version):
    """Builds a Google API service client using credentials from session (handles refresh internally)."""
    logging.debug(f"Attempting to build API service: {service_name} v{version}")
    credentials = get_credentials_from_session() # Use the revised function
    if not credentials:
        logging.warning(f"Cannot build {service_name}: No valid credentials obtained from session (may require re-auth).")
        return None # Session might have been cleared by get_credentials_from_session

    try:
        # The Credentials object (created without expiry passed) might attempt refresh
        # internally if needed when the API call is made, if it has a refresh token.
        service = build(service_name, version, credentials=credentials, cache_discovery=False)
        logging.info(f"Google {service_name} v{version} service built successfully.")
        return service
    except HttpError as e:
         status_code = e.resp.status
         error_details = "Unknown API error"
         try:
             error_content = json.loads(e.content.decode())
             error_details = error_content.get('error', {}).get('message', str(e))
         except: error_details = str(e)
         logging.error(f"Error building/using {service_name} (HttpError {status_code}): {error_details}")
         if status_code in [401, 403]:
              # Check if refresh was attempted and failed, potentially clear session
              logging.error("Authentication/Authorization error during service build/use.")
              # Decide whether to clear session: if it's 401/403, likely invalid grant/token
              if credentials and credentials.refresh_token:
                   logging.warning("Auth error despite having refresh token. Grant may be revoked.")
              session.clear() # Clear potentially bad credentials on 401/403
              logging.info("Cleared session due to auth error.")
         return None # Signal failure
    except Exception as e:
        logging.error(f"Unexpected error building {service_name} service: {e}", exc_info=True)
        return None


# --- Gemini Interaction Function (MODIFIED to accept history) ---
def get_intent_from_gemini(user_query, chat_history):
    """
    Uses Google Gemini to understand user intent, extract entities, check scope,
    considering the provided chat history.
    """
    if not gemini_model:
        logging.error("Gemini client is not initialized. Cannot process query.")
        return {"in_scope": False, "intent": "error", "reply": "AI assistant is currently unavailable."}

    # --- Prepare History for Gemini ---
    # Combine existing history with the new user query
    # The Gemini library expects history as a list of Content objects or dicts
    # The GenerativeModel was initialized with the system prompt, so we don't need to add it here.
    messages_for_gemini = chat_history + [{"role": "user", "parts": [{"text": user_query}]}]
    logging.debug(f"Sending to Gemini with {len(chat_history)} history turns.")
    # Log history content carefully (avoid excessive length)
    if chat_history:
        try:
            history_summary = json.dumps([{'role': msg['role'], 'text_preview': msg['parts'][0]['text'][:50] + '...'} for msg in chat_history[-4:]]) # Log last 2 turns preview
            logging.debug(f"Last history turns (preview): {history_summary}")
        except Exception as log_e:
            logging.warning(f"Could not create history summary for logging: {log_e}")


    try:
        logging.info(f"Sending query to Gemini (with history, expecting JSON): '{user_query[:100]}...'")
        # Use the model instance directly, which has system prompt, config, safety settings
        response = gemini_model.generate_content(messages_for_gemini)

        # --- Robust Response Handling & Parsing (Mostly Unchanged) ---
        response_text = ""
        # 1. Check Safety/Block reasons
        try:
            # Access safety feedback via prompt_feedback for the whole request or candidate.feedback for specific candidate
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_str = response.prompt_feedback.block_reason.name
                logging.error(f"Gemini request blocked by prompt feedback: {block_reason_str}")
                return {"in_scope": False, "intent": "error", "reply": f"Request blocked by AI safety filters ({block_reason_str})."}

            if not response.candidates:
                # Try to get finish reason if available (might be on the response object itself for some errors)
                finish_reason_str = "UNKNOWN"
                try:
                    if hasattr(response, 'finish_reason') and response.finish_reason:
                        finish_reason_str = response.finish_reason.name
                    elif response.prompt_feedback and response.prompt_feedback.block_reason:
                        finish_reason_str = f"BLOCKED_{response.prompt_feedback.block_reason.name}"
                except Exception as fr_e: logging.warning(f"Could not determine finish reason: {fr_e}")

                logging.error(f"Gemini response has no candidates. Finish Reason: {finish_reason_str}")
                reply = "AI returned no valid response."
                if "BLOCK" in finish_reason_str: reply = f"Request blocked by AI ({finish_reason_str})."
                elif finish_reason_str != "STOP": reply = f"AI response stopped unexpectedly ({finish_reason_str})."
                return {"in_scope": False, "intent": "error", "reply": reply}
        except AttributeError as ae: logging.warning(f"AttributeError checking response safety/candidates: {ae}")
        except Exception as e_safe: logging.warning(f"Error checking response safety/candidates: {e_safe}")

        # 2. Process Candidates (assuming the first candidate is the primary one)
        if response.candidates:
            candidate = response.candidates[0]
            # Check candidate-specific finish reason and safety ratings
            finish_reason = candidate.finish_reason
            finish_reason_str = finish_reason.name if finish_reason else "UNKNOWN"

            if finish_reason_str != "STOP":
                 logging.warning(f"Gemini generation finished unexpectedly: {finish_reason_str}")
                 safety_issues = []
                 if hasattr(candidate, 'safety_ratings'):
                      for rating in candidate.safety_ratings:
                           # Check if probability indicates a block or high risk
                           harm_probability = rating.probability.name if rating.probability else "UNKNOWN"
                           if harm_probability not in ['NEGLIGIBLE', 'LOW']: # Stricter check maybe?
                                harm_category = rating.category.name if rating.category else "UNKNOWN"
                                safety_issues.append(f"{harm_category}={harm_probability}")
                 if safety_issues:
                      logging.error(f"Safety issues detected in candidate: {', '.join(safety_issues)}")
                      return {"in_scope": False, "intent": "error", "reply": f"Response stopped due to safety concerns ({finish_reason_str}: {', '.join(safety_issues)})."}
                 else:
                      # If not STOP and no specific safety issue identified, still problematic
                      return {"in_scope": False, "intent": "error", "reply": f"AI response incomplete or flagged ({finish_reason_str})."}

            # 3. Extract Text Content (Assuming JSON response is in text part)
            try:
                 if candidate.content and candidate.content.parts:
                      response_text = candidate.content.parts[0].text
                 else:
                      # Fallback, though less likely with structured output
                      response_text = getattr(response, 'text', '')
                 if not response_text:
                      logging.error("Gemini response text empty despite STOP finish reason.")
                      return {"in_scope": False, "intent": "error", "reply": "AI provided an empty response."}
                 logging.debug(f"Raw Gemini response text: >>>{response_text}<<<")
            except AttributeError as ae:
                 logging.error(f"AttributeError extracting text part: {ae}. Candidate structure: {candidate}", exc_info=True)
                 return {"in_scope": False, "intent": "error", "reply": "Failed to read AI response structure."}
            except Exception as e:
                 logging.error(f"Error extracting text from Gemini response part: {e}", exc_info=True)
                 return {"in_scope": False, "intent": "error", "reply": "Failed to extract AI response text."}
        else:
             # This case should theoretically be caught by the 'no candidates' check above
             logging.error("Processing error: No candidates found in Gemini response after initial checks.")
             return {"in_scope": False, "intent": "error", "reply": "AI returned no response candidates."}

        # 4. Parse JSON (Unchanged logic, applied to extracted text)
        try:
            cleaned_text = response_text.strip()
            if not cleaned_text:
                 logging.error("Gemini response empty after stripping.")
                 return {"in_scope": False, "intent": "error", "reply": "AI provided empty formatted response."}
            logging.debug(f"Attempting JSON parse: >>>{cleaned_text}<<<")
            parsed_response = json.loads(cleaned_text)
            if not isinstance(parsed_response, dict) or 'in_scope' not in parsed_response or 'intent' not in parsed_response:
                 logging.error(f"Parsed JSON lacks expected keys. Parsed: {parsed_response}")
                 # Try to return the raw text if it was supposed to be a simple reply
                 if parsed_response and isinstance(parsed_response.get('reply'), str):
                     logging.warning("Parsed JSON bad format, but found a 'reply' field. Using that.")
                     return {"in_scope": False, "intent": "error", "reply": parsed_response['reply']}
                 return {"in_scope": False, "intent": "error", "reply": f"AI response had unexpected format."}
            logging.info(f"Successfully parsed Gemini JSON response: {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error: {e}. Invalid JSON: >>>{cleaned_text[:500]}<<<")
            snippet = cleaned_text[:100] + ('...' if len(cleaned_text) > 100 else '')
            # Attempt to strip markdown (common issue)
            if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
                 logging.error("Detected markdown. Trying to strip...")
                 stripped_again = cleaned_text[7:-3].strip()
                 try:
                      parsed_response = json.loads(stripped_again)
                      logging.info("Successfully parsed after stripping markdown.")
                      # Re-validate structure
                      if not isinstance(parsed_response, dict) or 'in_scope' not in parsed_response or 'intent' not in parsed_response:
                           logging.error(f"Parsed JSON (stripped) lacks keys. Parsed: {parsed_response}")
                           return {"in_scope": False, "intent": "error", "reply": "AI response unexpected format (even after stripping)." }
                      return parsed_response
                 except json.JSONDecodeError as e2:
                      logging.error(f"JSON Decode Error after stripping markdown: {e2}.")
                      return {"in_scope": False, "intent": "error", "reply": f"AI response not valid JSON. Starts with: '{snippet}'"}
            else:
                 # If not typical markdown, return error
                 return {"in_scope": False, "intent": "error", "reply": f"AI response not valid JSON. Starts with: '{snippet}'"}
        except Exception as e_parse:
            logging.error(f"Unexpected error parsing Gemini JSON: {e_parse}", exc_info=True)
            return {"in_scope": False, "intent": "error", "reply": "Failed to parse AI response."}

    # --- API Call Exception Handling (Unchanged) ---
    except google_exceptions.PermissionDenied as e:
        logging.error(f"Gemini API Permission Denied: {e}")
        return {"in_scope": False, "intent": "error", "reply": "AI service auth error."}
    except google_exceptions.ResourceExhausted as e:
        logging.error(f"Gemini API Quota Exceeded: {e}")
        return {"in_scope": False, "intent": "error", "reply": "AI service busy. Please try again later."}
    except google_exceptions.InvalidArgument as e:
         logging.error(f"Gemini API Invalid Argument: {e}. Check history format or query.")
         return {"in_scope": False, "intent": "error", "reply": f"Invalid request to AI: {e}"}
    except google_exceptions.GoogleAPIError as e:
         # Catch potential specific API errors like 500s from the service
         logging.error(f"Generic Gemini API Error: {e}")
         return {"in_scope": False, "intent": "error", "reply": f"AI communication error: {e}"}
    except Exception as e:
        logging.error(f"Unexpected error during Gemini call: {e}", exc_info=True)
        # Check block reason again if response object exists
        try:
             if 'response' in locals() and response and response.prompt_feedback and response.prompt_feedback.block_reason:
                  return {"in_scope": False, "intent": "error", "reply": f"Request blocked ({response.prompt_feedback.block_reason.name})."}
        except Exception: pass
        return {"in_scope": False, "intent": "error", "reply": "Unexpected error with AI assistant."}


# --- Date/Time Parsing Helper (parse_datetime_entities) (Unchanged from your code) ---
def parse_datetime_entities(date_str, time_str, start_iso_str=None):
    """
    Parses date and time strings (and potentially a pre-parsed ISO string)
    into aware UTC start datetime object.
    Returns start_dt_utc or None if parsing fails.
    """
    # (Same implementation as previous version)
    logging.debug(f"Parsing datetime: date='{date_str}', time='{time_str}', iso='{start_iso_str}'")
    now = datetime.datetime.now(datetime.timezone.utc)
    start_dt_utc = None

    if start_iso_str:
        try:
            start_dt_utc = dateutil_parser.isoparse(start_iso_str)
            if start_dt_utc.tzinfo is None:
                start_dt_utc = start_dt_utc.replace(tzinfo=datetime.timezone.utc)
                logging.debug(f"Parsed ISO string '{start_iso_str}' was naive, assumed UTC.")
            elif start_dt_utc.tzinfo != datetime.timezone.utc:
                start_dt_utc = start_dt_utc.astimezone(datetime.timezone.utc)
                logging.debug(f"Parsed ISO string '{start_iso_str}' had timezone, converted to UTC.")
            logging.info(f"Successfully parsed start_datetime_iso: {start_dt_utc}")
            return start_dt_utc
        except Exception as e:
            logging.warning(f"Failed to parse provided start_datetime_iso '{start_iso_str}': {e}. Falling back.")
            start_dt_utc = None

    # Fallback to date_str and time_str if ISO parsing failed or wasn't provided
    if not date_str or not time_str:
        logging.warning("Cannot parse datetime: Missing date or time string (and no valid ISO string).")
        return None

    try:
        # Handle relative terms before passing to dateutil
        date_part_for_parse = date_str # Default
        if date_str.lower() == 'today':
            date_part_for_parse = now.strftime('%Y-%m-%d')
        elif date_str.lower() == 'tomorrow':
            tomorrow_dt = now + datetime.timedelta(days=1)
            date_part_for_parse = tomorrow_dt.strftime('%Y-%m-%d')
        # Add more relative terms if needed (e.g., 'next monday')

        combined_str = f"{date_part_for_parse} {time_str}"
        logging.debug(f"Attempting to parse combined string: '{combined_str}'")

        # Use fuzzy parsing carefully if needed, but prefer specific formats
        parsed_dt_naive = dateutil_parser.parse(combined_str) # Can raise ValueError

        # Ensure the parsed datetime is timezone-aware (UTC)
        if parsed_dt_naive.tzinfo is None:
             start_dt_utc = parsed_dt_naive.replace(tzinfo=datetime.timezone.utc)
             logging.info(f"Parsed '{combined_str}' as naive. Assuming UTC: {start_dt_utc}")
        else:
             start_dt_utc = parsed_dt_naive.astimezone(datetime.timezone.utc)
             logging.info(f"Parsed '{combined_str}' with timezone, converted to UTC: {start_dt_utc}")

        # Optional: Check if the resulting date is in the past without explicit year, adjust if needed
        # if start_dt_utc < now and str(now.year) not in date_str:
        #     logging.warning(f"Parsed date {start_dt_utc} is in the past, potentially add a year?")
        #     # Add logic here if you want automatic year adjustment

        logging.info(f"Successfully parsed combined date/time string to UTC: {start_dt_utc}")
        return start_dt_utc

    except ValueError as ve:
        logging.error(f"Failed to parse date/time combination '{combined_str}' (ValueError): {ve}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing date/time '{combined_str}': {e}", exc_info=True)
        return None


# --- Flask Routes ---

# --- Route: / (index) (Unchanged) ---
@app.route('/')
def index():
    """Renders the main chat page."""
    user_info = session.get('user_info')
    has_creds_in_session = 'credentials' in session
    logging.debug(f"Rendering index page - UserInfo: {user_info}, Has Creds in Session: {has_creds_in_session}")
    # Clear history if no creds? Optional, but keeps state consistent.
    if not has_creds_in_session and 'chat_history' in session:
        logging.debug("No credentials found, clearing any leftover chat history.")
        session.pop('chat_history', None)
    return render_template('index.html', user_info=user_info, is_authenticated=has_creds_in_session)

# --- Route: /authorize (Unchanged) ---
@app.route('/authorize')
def authorize():
    """Starts the Google OAuth 2.0 flow."""
    # (Same implementation as previous version)
    flow = get_google_auth_flow()
    if not flow:
         logging.error("Failed to create OAuth flow in /authorize.")
         return "Error initiating authorization. Server configuration issue.", 500
    try:
        authorization_url, state = flow.authorization_url(
            access_type='offline',        # Request refresh token
            include_granted_scopes='true',
            prompt='consent'              # Force consent screen for refresh token
        )
        session['oauth_state'] = state
        logging.info(f"Redirecting user to Google for authorization. State stored: {state}")
        return redirect(authorization_url)
    except Exception as e:
        logging.error(f"Error generating authorization URL: {e}", exc_info=True)
        return "Error generating authorization URL. Check OAuth configuration.", 500


# --- Route: /oauth2callback (Unchanged) ---
@app.route('/oauth2callback')
def oauth2callback():
    """Handles the redirect back from Google after user authorization."""
    # (Same implementation as previous version)
    logging.debug(f"OAuth Callback received. URL (query hidden): {request.path}")
    state = session.pop('oauth_state', None)
    request_state = request.args.get('state')

    # State Validation
    if not state or state != request_state:
        logging.error(f"OAuth state mismatch. Session: '{state}', Request: '{request_state}'")
        session.clear() # Clear session on state mismatch for security
        return 'Invalid state parameter. Please try logging in again.', 400

    # Error Handling from Google
    oauth_error = request.args.get('error')
    if oauth_error:
        error_desc = request.args.get('error_description', 'No description provided.')
        logging.error(f"OAuth Error from Google: {oauth_error} - {error_desc}")
        # Provide user-friendly messages for common errors
        if oauth_error == 'access_denied':
             return 'Authorization denied by user. <a href="/">Try again</a>', 403
        elif oauth_error == 'invalid_scope':
             return f'Invalid OAuth scope requested: {error_desc}. Please contact support. <a href="/">Go back</a>', 400
        # Add more specific error handling if needed
        else:
             return f'Authorization failed: {oauth_error} - {error_desc}. <a href="/">Go back</a>', 400

    # Get OAuth Flow instance
    flow = get_google_auth_flow()
    if not flow:
         logging.error("Failed to recreate OAuth flow in callback.")
         return "Server error during authorization callback. Please try again.", 500

    credentials = None
    try:
        logging.info("Attempting to fetch OAuth tokens...")
        # Ensure URL uses HTTPS if not running locally with insecure transport enabled
        auth_response_url = request.url
        if os.getenv('OAUTHLIB_INSECURE_TRANSPORT') != '1' and auth_response_url.startswith('http://'):
             # This replacement is crucial for production environments
             auth_response_url = auth_response_url.replace('http://', 'https://', 1)
             logging.warning(f"Forcing HTTPS for OAuth callback token exchange URL: {auth_response_url}")

        flow.fetch_token(authorization_response=auth_response_url)
        credentials = flow.credentials # Contains access_token, refresh_token (if granted), etc.
        logging.info("OAuth tokens fetched successfully.")

        # Check specifically for refresh token
        if not credentials.refresh_token:
             logging.warning("No refresh token received. User may need to re-authenticate later if access token expires and offline access wasn't granted or already used.")
        else:
             logging.info("Refresh token received and stored.") # It will be in the credentials_dict

    except google_exceptions.OAuth2Error as e:
         # Catch specific OAuth library errors during token exchange
         logging.error(f"OAuth2Error fetching token: {e}", exc_info=True)
         msg = "Error exchanging authorization code for tokens. "
         details = str(e).lower()
         if 'invalid_grant' in details: msg += 'Authorization grant may be expired or invalid. Please try logging in again.'
         elif 'redirect_uri_mismatch' in details: msg += 'Redirect URI mismatch. Server configuration error. Please contact support.'
         else: msg += f'Details: {e}. Please try again.'
         return msg, 400 # Return specific error code
    except Exception as e:
        # Catch any other unexpected errors during token fetch
        logging.error(f"Unexpected error fetching token: {e}", exc_info=True)
        return "An unexpected error occurred during authorization. Please try again.", 500

    # Store credentials securely in session
    session['credentials'] = credentials_to_dict(credentials)
    # Log credential info carefully (avoid logging sensitive parts like secret/token)
    log_cred = {k:v for k,v in session['credentials'].items() if k not in ['client_secret', 'refresh_token', 'token', 'id_token']}
    log_cred['refresh_token_present'] = bool(session['credentials'].get('refresh_token'))
    log_cred['expiry_stored'] = session['credentials'].get('expiry') # Log the stored expiry string
    logging.info(f"Credentials stored in session (sensitive info masked/summarized): {log_cred}")

    # Fetch User Info using the obtained credentials
    user_info_to_store = None
    try:
        logging.info("Attempting to fetch user info...")
        # Build the oauth2 service to get user profile info
        user_info_service = build('oauth2', 'v2', credentials=credentials, cache_discovery=False)
        user_info_raw = user_info_service.userinfo().get().execute()
        # Extract only necessary fields
        user_info_to_store = {
            'name': user_info_raw.get('name'),
            'email': user_info_raw.get('email'),
            'picture': user_info_raw.get('picture') # Profile picture URL
        }
        if not user_info_to_store.get('email'):
             logging.error(f"User info fetched but email is missing: {user_info_raw}")
             # Decide how to handle - maybe fail auth? Or proceed without email?
             return "Could not retrieve user email during authentication. Please try again.", 500

        logging.info(f"User info fetched successfully for: {user_info_to_store.get('email')}")
    except HttpError as e:
         # Handle errors specifically during user info fetching
         status_code = e.resp.status
         error_details = f"Failed to fetch user profile information ({status_code})."
         try:
             # Try to parse error details from Google response
             error_content = json.loads(e.content.decode())
             error_details += f" Details: {error_content.get('error', {}).get('message', str(e))}"
         except: pass # Ignore if parsing fails
         logging.error(f"Error fetching user info (HttpError {status_code}): {e.content.decode(errors='ignore')}")
         # Store error state? Or just fail? Let's fail cleanly here.
         return f"Error fetching user profile: {error_details}. Please try again.", 500
    except Exception as e:
        # Catch other errors during user info fetching
        logging.error(f"Unexpected error fetching user info: {e}", exc_info=True)
        return "An unexpected error occurred while retrieving user profile.", 500

    # Store user info in session
    session['user_info'] = user_info_to_store
    logging.info(f"Stored final user info state: {session.get('user_info')}")

    # Clear any previous chat history upon successful new login
    if 'chat_history' in session:
        logging.debug("Clearing previous chat history on new login.")
        session.pop('chat_history', None)

    logging.info("OAuth callback completed successfully. Redirecting to index.")
    return redirect(url_for('index'))


# --- Route: /chat (MODIFIED to load/save history) ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages, uses Gemini for intent (with history), interacts with Calendar API."""
    start_time = datetime.datetime.now()
    try:
        if not request.is_json:
             logging.warning("Chat request not JSON.")
             return jsonify({'reply': 'Invalid request format.'}), 415
        request_data = request.get_json()
        user_message = request_data.get('message')
        if not user_message or not isinstance(user_message, str) or not user_message.strip():
            logging.warning("Chat request no 'message' field or empty.")
            return jsonify({'reply': 'Please provide a message.'}), 400
        user_message = user_message.strip()
        logging.info(f"Chat request received: '{user_message[:100]}...'")

        # --- Load Chat History ---
        chat_history = session.get('chat_history', [])
        logging.debug(f"Loaded {len(chat_history)} history turns from session.")

        # 1. Check Auth & Get Calendar Service (Crucial First Step)
        # This call might clear the session if refresh fails.
        calendar_service = build_api_service('calendar', 'v3')
        if not calendar_service:
            # Check if session was cleared (meaning auth failed irrecoverably)
            if 'credentials' not in session:
                 logging.warning("Calendar service build failed (auth needed, session likely cleared).")
                 auth_url = url_for('authorize')
                 reply_html = (f'Your session seems to have expired or is invalid. Please <a href="{auth_url}" target="_blank" class="auth-link">re-authenticate with Google</a> to continue.')
                 # Clear history again just in case it wasn't cleared by build_api_service path
                 session.pop('chat_history', None)
                 return jsonify({'reply': reply_html, 'needs_auth': True})
            else: # Other failure (network, API down?) Session still exists but service build failed
                 logging.error("Calendar service build failed (temporary issue? Credentials still in session).")
                 return jsonify({'reply': 'Sorry, there was a problem connecting to Google Calendar at the moment. Please try again shortly.'}), 503
        logging.debug("Google Calendar service obtained successfully.")

                # 2. Process Message with Gemini (passing history)
        # CORRECTED LINE: Use user_message instead of user_query
        gemini_result = get_intent_from_gemini(user_message, chat_history) # Pass history
        bot_reply = "Sorry, I encountered an issue processing your request with the AI assistant." # Default error reply
        needs_auth_flag = False # Flag if reply requires re-authentication
        intent = "error"        # Default intent

        # --- Process Gemini Response ---
        if not gemini_result:
            logging.error("Gemini processing returned None or empty result.")
            # Use the default error reply
        else:
            intent = gemini_result.get('intent', 'error')
            # Use reply from Gemini if provided, otherwise keep the default/action reply
            bot_reply = gemini_result.get('reply', bot_reply)
            logging.info(f"Gemini result - Intent: {intent}. In Scope: {gemini_result.get('in_scope', False)}")
            in_scope = gemini_result.get('in_scope', False)
            entities = gemini_result.get('entities', {})

            # 3. Handle Intent based on Gemini Result
            action_reply = None # Specific reply generated by calendar actions
            if intent == "error":
                 # Error occurred within Gemini processing, use the reply Gemini provided
                 pass # bot_reply already contains Gemini's error message
            elif not in_scope or intent == "out_of_scope":
                 # Request is validly interpreted as out of scope
                 pass # bot_reply contains Gemini's "I can only..." message
            elif intent == "clarification_needed":
                 # AI needs more info
                 pass # bot_reply contains Gemini's clarification question
            elif intent == 'list_events':
                 logging.info(f"Executing 'list_events'. Entities: {entities}")
                 action_reply = "Sorry, I could not retrieve the events." # Default for this action
                 try:
                     now_utc = datetime.datetime.now(datetime.timezone.utc)
                     # Use Gemini extracted date, default to 'today'
                     target_date_str = entities.get('date', 'today')
                     time_min_dt, time_max_dt = None, None

                     # --- Improved Date Range Parsing ---
                     date_desc = target_date_str # For user feedback
                     target_date_str_lower = target_date_str.lower()
                     if target_date_str_lower == 'today':
                         time_min_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                         time_max_dt = time_min_dt + datetime.timedelta(days=1)
                         date_desc = "today"
                     elif target_date_str_lower == 'tomorrow':
                          start_of_tomorrow = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                          time_min_dt = start_of_tomorrow
                          time_max_dt = time_min_dt + datetime.timedelta(days=1)
                          date_desc = "tomorrow"
                     elif target_date_str_lower == 'next week':
                         # Define 'next week' (e.g., next Mon to Sun, or next 7 days)
                         # Let's do next 7 days from tomorrow for simplicity
                         start_of_range = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                         time_min_dt = start_of_range
                         time_max_dt = time_min_dt + datetime.timedelta(days=7)
                         date_desc = "the next 7 days"
                     else: # Try specific date parse
                          try:
                              # Use dateutil parser for flexibility (e.g., "August 15th", "2024-08-15")
                              parsed_date_naive = dateutil_parser.parse(target_date_str).date()
                              # Assume UTC for the date range
                              time_min_dt = datetime.datetime(parsed_date_naive.year, parsed_date_naive.month, parsed_date_naive.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
                              time_max_dt = time_min_dt + datetime.timedelta(days=1) # Events for that specific day
                              date_desc = time_min_dt.strftime('%B %d, %Y')
                          except Exception as e_parse_date:
                              logging.warning(f"Could not parse date '{target_date_str}' for listing events: {e_parse_date}")
                              action_reply = f"Sorry, I couldn't understand the date '{target_date_str}'. Please try 'today', 'tomorrow', 'next week', or a specific date like 'YYYY-MM-DD' or 'Month Day'."

                     # --- Fetch Events if Date Range is Valid ---
                     if time_min_dt and time_max_dt:
                          time_min_iso = time_min_dt.isoformat()
                          time_max_iso = time_max_dt.isoformat() # End time is exclusive in API
                          logging.info(f"Fetching events for '{date_desc}': Range {time_min_iso} to {time_max_iso}")
                          events_result = calendar_service.events().list(
                              calendarId='primary',
                              timeMin=time_min_iso,
                              timeMax=time_max_iso,
                              maxResults=20, # Increase slightly?
                              singleEvents=True,
                              orderBy='startTime'
                          ).execute()
                          events = events_result.get('items', [])

                          # --- Format Event List ---
                          if not events:
                              action_reply = f"You have no events scheduled for {date_desc}."
                          else:
                              lines = [f"Here are your events for {date_desc}:"]
                              for event in events:
                                  summary = event.get('summary', '(No Title)')
                                  start = event.get('start', {})
                                  start_time_str = "Unknown Time"
                                  if 'dateTime' in start: # Timed event
                                       try:
                                           start_dt = dateutil_parser.isoparse(start['dateTime'])
                                           # Format time nicely (consider user's local timezone if available/needed, but API returns UTC/specified)
                                           # For simplicity, show time as returned or convert to simple format
                                           start_time_str = start_dt.strftime('%-I:%M %p') # e.g., 9:30 AM
                                           # Optionally add timezone info if relevant start_dt.strftime('%-I:%M %p %Z')
                                       except Exception as e_fmt:
                                           logging.warning(f"Error formatting event start time {start.get('dateTime')}: {e_fmt}")
                                           start_time_str = start.get('dateTime') # Fallback to ISO string
                                  elif 'date' in start: # All-day event
                                       start_time_str = "All day"
                                       try:
                                           # Show the date for multi-day views if needed
                                           all_day_date = dateutil_parser.parse(start['date']).date()
                                           if time_max_dt - time_min_dt > datetime.timedelta(days=1):
                                               start_time_str = f"All day on {all_day_date.strftime('%b %d')}"
                                       except Exception as e_fmt_all_day:
                                           logging.warning(f"Error formatting all-day event date {start.get('date')}: {e_fmt_all_day}")

                                  lines.append(f" • {summary} ({start_time_str})")
                              action_reply = "\n".join(lines) # Use newline for better readability in HTML

                 except HttpError as error:
                     status = error.resp.status
                     details = f"API error ({status}) listing events."
                     try: details = json.loads(error.content.decode('utf-8', errors='ignore')).get('error', {}).get('message', details)
                     except: pass
                     logging.error(f'Google Calendar HttpError during list_events: {status}, {details}, Response: {error.content.decode("utf-8", errors="ignore")}')
                     if status in [401, 403]: # Authentication / Permission Error
                          # Clear session as credentials are bad
                          session.clear()
                          auth_url = url_for('authorize')
                          action_reply = f'Calendar Access Error ({details}). Your session was cleared. Please <a href="{auth_url}" class="auth-link">re-authenticate with Google</a>.'
                          needs_auth_flag = True
                     else: # Other API errors (rate limit, server error, etc.)
                          action_reply = f"Sorry, Google Calendar returned an error ({status}): {details}. Please try again later."
                 except Exception as e:
                      logging.exception(f"Unexpected error during 'list_events' execution: {e}")
                      action_reply = f"An unexpected internal error occurred while trying to list events. Please try again later."

                 # Set the final reply for this intent
                 if action_reply: bot_reply = action_reply

            elif intent == 'schedule_meeting':
                 logging.info(f"Executing 'schedule_meeting'. Entities: {entities}")
                 action_reply = "Sorry, I couldn't schedule the meeting." # Default for this action
                 try:
                     # --- Extract Entities ---
                     summary = entities.get('summary')
                     date_str = entities.get('date')
                     time_str = entities.get('time')
                     start_iso_str = entities.get('start_datetime_iso') # Preferred if available
                     attendees_list = entities.get('attendees', [])
                     duration_minutes = 60 # Default duration
                     try:
                         extracted_duration = entities.get('duration_minutes')
                         if extracted_duration is not None:
                             duration_minutes = int(extracted_duration)
                     except (ValueError, TypeError):
                         logging.warning(f"Invalid duration '{entities.get('duration_minutes')}', using default {duration_minutes} min.")
                         duration_minutes = 60

                     # --- Validate Required Entities ---
                     if not summary:
                         action_reply = "I need a title or summary to schedule the meeting. What should it be called?"
                     elif not start_iso_str and (not date_str or not time_str):
                         action_reply = "I need the date and time for the meeting. When should it be?"
                     else:
                          # --- Parse Datetime ---
                          start_dt_utc = parse_datetime_entities(date_str, time_str, start_iso_str)
                          if not start_dt_utc:
                              action_reply = "I couldn't understand the date and time you provided. Please try specifying like 'tomorrow at 3 PM', 'August 20th 10:00', or '2024-08-20 15:00'."
                          else:
                               # --- Prepare Event Body ---
                               end_dt_utc = start_dt_utc + datetime.timedelta(minutes=duration_minutes)
                               start_rfc3339 = start_dt_utc.isoformat()
                               end_rfc3339 = end_dt_utc.isoformat()

                               # Validate and format attendees
                               attendees_api_format = []
                               invalid_emails = []
                               if attendees_list and isinstance(attendees_list, list):
                                    for email in attendees_list:
                                         if isinstance(email, str) and '@' in email and '.' in email.split('@')[-1]: # Basic validation
                                              attendees_api_format.append({'email': email.strip()})
                                         else:
                                              logging.warning(f"Invalid attendee format skipped: {email}")
                                              invalid_emails.append(str(email)) # Collect invalid ones for feedback

                               event_body = {
                                   'summary': summary,
                                   'start': {'dateTime': start_rfc3339, 'timeZone': 'UTC'}, # Always use UTC for API
                                   'end': {'dateTime': end_rfc3339, 'timeZone': 'UTC'},
                                   'attendees': attendees_api_format,
                                   'reminders': { # Add a default reminder
                                       'useDefault': False,
                                       'overrides': [
                                           {'method': 'popup', 'minutes': 15}, # 15 min popup reminder
                                           # {'method': 'email', 'minutes': 60}, # Optional email reminder
                                       ],
                                   },
                                   # Add description? Location? Conference data? - Future enhancements
                                   # 'description': f"Meeting scheduled via AI Assistant.\nUser query: {user_message}",
                               }
                               logging.info(f"Attempting to create event: {json.dumps(event_body)}") # Log the event body

                               # --- Call Google Calendar API ---
                               created_event = calendar_service.events().insert(
                                   calendarId='primary',
                                   body=event_body,
                                   sendNotifications=True # Send invites to attendees
                               ).execute()

                               # --- Format Success Reply ---
                               event_link = created_event.get('htmlLink', '#') # Link to the event in Google Calendar Web UI
                               # Format start time more readably for confirmation
                               start_time_formatted = start_dt_utc.strftime('%A, %B %d, %Y at %I:%M %p %Z') # e.g., Tuesday, August 20, 2024 at 3:00 PM UTC
                               action_reply = f"OK, I've scheduled '{summary}' for you on {start_time_formatted}. "
                               action_reply += f"<a href='{event_link}' target='_blank' class='event-link'>View Event</a>"
                               if attendees_api_format:
                                    action_reply += f"<br>Invites sent to {len(attendees_api_format)} attendee(s)."
                               if invalid_emails:
                                    action_reply += f"<br><small>(Note: I couldn't add these as attendees: {', '.join(invalid_emails)})</small>"

                 except HttpError as error:
                     status = error.resp.status
                     details = f"API error ({status}) scheduling meeting."
                     try: details = json.loads(error.content.decode('utf-8', errors='ignore')).get('error', {}).get('message', details)
                     except: pass
                     logging.error(f'Google Calendar HttpError during schedule_meeting: {status}, {details}, Response: {error.content.decode("utf-8", errors="ignore")}')
                     if status in [401, 403]: # Auth error
                          session.clear() # Clear bad creds
                          auth_url = url_for('authorize')
                          action_reply = f'Calendar Access Error ({details}). Your session was cleared. Please <a href="{auth_url}" class="auth-link">re-authenticate with Google</a>.'
                          needs_auth_flag = True
                     elif status == 400: # Bad request (e.g., invalid date format, invalid email)
                          action_reply = f"Couldn't schedule the meeting due to invalid data: {details}. Please check the event details (like time format or attendee emails) and try again."
                     else: # Other API errors
                          action_reply = f"Sorry, Google Calendar returned an error ({status}): {details}. Please try again later."
                 except Exception as e:
                      logging.exception(f"Unexpected error during 'schedule_meeting' execution: {e}")
                      action_reply = f"An unexpected internal error occurred while trying to schedule the meeting. Please try again later."

                 # Set the final reply for this intent
                 if action_reply: bot_reply = action_reply
            else: # Should not happen if Gemini adheres to prompt, but handle defensively
                 logging.warning(f"Received unhandled valid intent: {intent}. Entities: {entities}")
                 # Use Gemini's reply if available, otherwise a generic message
                 if not gemini_result.get('reply'):
                     bot_reply = f"I understood the intent '{intent}', but I don't have specific handling for it yet."

        # --- Update Chat History ---
        # Ensure bot_reply is a string before storing
        if not isinstance(bot_reply, str):
            logging.error(f"bot_reply is not a string ({type(bot_reply)}), converting. Value: {bot_reply}")
            bot_reply = str(bot_reply)

        # Add the current user message and the final bot reply to history
        # Use the Gemini expected format: list of dicts with "role" and "parts"
        chat_history.append({"role": "user", "parts": [{"text": user_message}]})
        chat_history.append({"role": "model", "parts": [{"text": bot_reply}]}) # Use "model" for the AI role

        # --- Trim History ---
        if len(chat_history) > MAX_HISTORY_TURNS * 2:
            logging.debug(f"History length ({len(chat_history)}) exceeds max ({MAX_HISTORY_TURNS*2}). Trimming oldest turns.")
            chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

        # --- Save Updated History to Session ---
        session['chat_history'] = chat_history
        logging.debug(f"Saved {len(chat_history)} history turns to session.")

        # 4. Send Final Reply to Frontend
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"Sending final reply (Intent: {intent}, Needs Auth: {needs_auth_flag}, Time: {processing_time:.2f}s): '{bot_reply[:150]}...'")
        return jsonify({'reply': bot_reply, 'needs_auth': needs_auth_flag})

    except Exception as e:
        # Catch-all for any unexpected errors in the main try block
        logging.exception(f"Critical unexpected error in /chat endpoint: {e}")
        # Avoid saving history if a critical error occurred mid-processing
        return jsonify({'reply': 'Sorry, an unexpected server error occurred. Please try again.'}), 500


# --- Route: /logout (MODIFIED to clear history) ---
@app.route('/logout')
def logout():
    """Clears the user's session, including credentials and chat history."""
    user_email = session.get('user_info', {}).get('email', 'Unknown User')
    logging.info(f"Logging out user: {user_email}. Clearing entire session.")
    # session.clear() handles removing 'credentials', 'user_info', 'chat_history', etc.
    session.clear()
    logging.info("Session cleared.")
    return redirect(url_for('index'))

# --- Run Application (Unchanged) ---
if __name__ == '__main__':
    # (Same implementation as previous version)
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', '0').lower() in ('true', '1', 't')

    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.getLogger().setLevel(log_level)
    logging.getLogger('werkzeug').setLevel(logging.INFO if not debug_mode else logging.DEBUG)
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR) # Reduce noise
    logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING) # Reduce noise
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO) # Reduce noise


    logging.info(f"Flask Debug Mode: {debug_mode}")
    logging.info(f"Application Root Log Level set to: {logging.getLevelName(log_level)}")

    # Environment checks for OAuth security
    if debug_mode and os.getenv('OAUTHLIB_INSECURE_TRANSPORT') != '1':
         logging.warning("Flask debug mode is ON, but OAUTHLIB_INSECURE_TRANSPORT is not set to '1'. Setting it now to allow HTTP callbacks for local development.")
         os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    elif not debug_mode and os.getenv('OAUTHLIB_INSECURE_TRANSPORT') == '1':
         logging.warning("Flask debug mode is OFF, but OAUTHLIB_INSECURE_TRANSPORT is set to '1'. This is insecure for production. Ensure your redirect URI uses HTTPS.")
    elif not debug_mode and not REDIRECT_URI.startswith('https'):
         logging.warning(f"Flask debug mode is OFF and the configured REDIRECT_URI ('{REDIRECT_URI}') does not start with 'https'. OAuth flow may fail or be insecure in production.")

    # Critical configuration checks
    if not app.secret_key or app.secret_key == 'a_default_fallback_secret_key_for_dev':
        logging.critical("FATAL: FLASK_SECRET_KEY is not set or is using the default insecure value. Sessions are not secure. SET THIS ENVIRONMENT VARIABLE.")
        # Consider exiting if critical config missing? Or just warn loudly.
    if not google_api_key:
        logging.error("GOOGLE_API_KEY environment variable not found. Gemini AI will not function.")
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        logging.error("GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET environment variable not found. Google OAuth will not function.")
    if gemini_model is None:
         logging.warning("Gemini model failed to initialize. AI features will be unavailable.")


    logging.info(f"Starting Flask server - Access at: http://{host}:{port}/")
    # Use waitress or gunicorn for production instead of app.run(debug=...)
    app.run(host=host, port=port, debug=debug_mode)