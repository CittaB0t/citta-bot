import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import BadRequest, GoogleAPIError

try:
    import requests
except Exception:
    requests = None


# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Citta Companion",
    page_icon="🧠",
    layout="centered"
)

st.title("Citta Companion")
st.caption(
    "AI-supported wellbeing discovery. "
    "Not a diagnosis, therapy, crisis service, or emergency support."
)


# ------------------------------------------------------------
# Helper: read Streamlit secrets safely
# ------------------------------------------------------------

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Reads from Streamlit Cloud secrets first.
    If not found, reads from environment variables.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


# ------------------------------------------------------------
# Read employee details from URL
# Example:
# https://your-app.streamlit.app?id=CITTA-001&sector=IT&lang=en
# ------------------------------------------------------------

query_params = st.query_params

EMPLOYEE_ID = query_params.get("id", "TEST-USER")
EMPLOYEE_SECTOR = query_params.get("sector", "General")
EMPLOYEE_LANG = query_params.get("lang", "en")


# ------------------------------------------------------------
# Gemini API key setup
# ------------------------------------------------------------

GOOGLE_API_KEY = (
    get_secret("GOOGLE_API_KEY")
    or get_secret("GEMINI_API_KEY")
    or get_secret("GOOGLE_GEMINI_API_KEY")
)

if GOOGLE_API_KEY:
    GOOGLE_API_KEY = str(GOOGLE_API_KEY).strip().strip('"').strip("'")

if not GOOGLE_API_KEY:
    st.error(
        "Missing Gemini API key. Please add GOOGLE_API_KEY in Streamlit Cloud secrets."
    )
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


# ------------------------------------------------------------
# Gemini model setup
# ------------------------------------------------------------

MODEL_NAME = get_secret("GEMINI_MODEL", "gemini-3.5-flash")

if MODEL_NAME:
    MODEL_NAME = str(MODEL_NAME).strip().strip('"').strip("'")
else:
    MODEL_NAME = "gemini-3.5-flash"


# ------------------------------------------------------------
# Optional Make webhook for risk alerts
# You can leave this blank for now.
# Later add in Streamlit secrets:
# MAKE_ALERT_WEBHOOK_URL = "your_make_webhook_url"
# ------------------------------------------------------------

MAKE_ALERT_WEBHOOK_URL = get_secret("MAKE_ALERT_WEBHOOK_URL", "")

if MAKE_ALERT_WEBHOOK_URL:
    MAKE_ALERT_WEBHOOK_URL = str(MAKE_ALERT_WEBHOOK_URL).strip().strip('"').strip("'")


# ------------------------------------------------------------
# Red-flag keyword detection
# This is a simple safety net for MVP.
# ------------------------------------------------------------

RED_FLAGS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "self-harm",
    "hurt myself",
    "can't go on",
    "cannot go on",
    "domestic violence",
    "abuse",
    "overdose",
    "addicted",
    "substance abuse",
    "i am not safe",
    "i don't feel safe",
    "i want to die",
    "no reason to live",
    "ending it all",
    "harm myself",
    "hurt someone",
    "kill someone",
]


def detect_red_flag(message: str) -> bool:
    """
    Checks whether the user message contains obvious safety red flags.
    This is not diagnosis. It is only a safety trigger.
    """

    if not message:
        return False

    text = message.lower()

    return any(flag in text for flag in RED_FLAGS)


def send_risk_alert(employee_id: str, sector: str, message: str, reason: str):
    """
    Optional: sends risk alert to Make webhook.
    If MAKE_ALERT_WEBHOOK_URL is not configured, this does nothing.
    """

    if not MAKE_ALERT_WEBHOOK_URL:
        return

    if requests is None:
        return

    payload = {
        "employee_id": employee_id,
        "sector": sector,
        "message": message,
        "risk_reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        requests.post(MAKE_ALERT_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Risk alert failed: {e}")


# ------------------------------------------------------------
# Gemini system instruction
# ------------------------------------------------------------

SYSTEM_INSTRUCTION = f"""
You are Citta Companion, a compassionate workplace mental health support companion.

Employee session ID: {EMPLOYEE_ID}
Sector: {EMPLOYEE_SECTOR}
Language: {EMPLOYEE_LANG}

Your purpose:
- Support early wellbeing discovery.
- Ask gentle questions about how the employee is feeling.
- Explore general wellbeing, stress, sleep, emotional load, workplace strain, and support needs.
- Be warm, non-judgmental, trauma-informed, and workplace-aware.

Important boundaries:
- You are not a therapist.
- You are not a doctor.
- You are not a diagnosis tool.
- You are not a crisis service.
- You are not an emergency service.
- Do not diagnose.
- Do not say the user has PTSD, depression, anxiety, addiction, trauma, ADHD, or any clinical condition.
- Do not provide medical advice.
- Do not provide legal advice.
- Do not make HR decisions.
- Do not tell the employee that their individual response will be shared with their employer.

Privacy statement:
- Individual responses are not shared with the employer.
- The employer may receive only de-identified and aggregated insights.
- If serious distress or safety concerns appear, Citta may recommend human support or alert the Citta intake team for review.

Conversation style:
- Greet the employee warmly.
- Ask how they are feeling today.
- Keep responses brief and clear.
- Ask one or two questions at a time.
- Use simple language.
- Be supportive but not clinical.
- If the employee seems distressed, gently suggest that speaking with a human Citta intake professional or counsellor may be helpful.

Safety handling:
- If there are signs of immediate danger, self-harm, harm to others, abuse, or serious safety risk, tell the user that Citta Companion is not emergency support.
- Advise them to contact local emergency services, a trusted person, or a human support professional immediately if they may be unsafe.
- Encourage human support for serious or ongoing distress.

Start the conversation by saying hello and asking how the employee is feeling today.
"""


# ------------------------------------------------------------
# Create Gemini model
# ------------------------------------------------------------

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "raw_history" not in st.session_state:
    st.session_state.raw_history = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# ------------------------------------------------------------
# Convert Streamlit messages into Gemini format
# ------------------------------------------------------------

def to_gemini_history(raw_history: List[Dict[str, Any]], max_messages: int = 30) -> List[Dict[str, Any]]:
    """
    Streamlit uses:
        {"role": "assistant", "content": "..."}

    Gemini expects:
        {"role": "model", "parts": ["..."]}

    This function converts the history safely.
    """

    converted: List[Dict[str, Any]] = []

    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            gemini_role = "user"
        elif role in ["assistant", "model"]:
            gemini_role = "model"
        else:
            continue

        if content is None:
            continue

        if isinstance(content, list):
            content = "\n".join(str(item) for item in content if item)
        else:
            content = str(content)

        content = content.strip()

        if not content:
            continue

        converted.append(
            {
                "role": gemini_role,
                "parts": [content]
            }
        )

    # Gemini should not receive history that starts with the model.
    while converted and converted[0]["role"] != "user":
        converted.pop(0)

    # Merge consecutive messages with the same role.
    normalised: List[Dict[str, Any]] = []

    for item in converted:
        if normalised and normalised[-1]["role"] == item["role"]:
            normalised[-1]["parts"][0] += "\n\n" + item["parts"][0]
        else:
            normalised.append(item)

    return normalised[-max_messages:]


# ------------------------------------------------------------
# Extract Gemini response text safely
# ------------------------------------------------------------

def extract_response_text(response: Any) -> str:
    """
    Safely extracts text from Gemini response.
    """

    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass

    try:
        candidates = getattr(response, "candidates", [])

        if candidates:
            parts = candidates[0].content.parts
            texts = []

            for part in parts:
                text = getattr(part, "text", "")

                if text:
                    texts.append(text)

            final_text = "\n".join(texts).strip()

            if final_text:
                return final_text

    except Exception:
        pass

    try:
        prompt_feedback = getattr(response, "prompt_feedback", None)

        if prompt_feedback:
            return (
                "I could not generate a response because the request was blocked "
                f"or rejected by the model safety system.\n\nDetails: {prompt_feedback}"
            )

    except Exception:
        pass

    return "I could not generate a response. Please try again."


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.subheader("Citta Session")

    st.write("Employee session ID:")
    st.code(EMPLOYEE_ID)

    st.write("Sector:")
    st.code(EMPLOYEE_SECTOR)

    st.write("Language:")
    st.code(EMPLOYEE_LANG)

    st.divider()

    st.write("Gemini model:")
    st.code(MODEL_NAME)

    if GOOGLE_API_KEY:
        masked_key = GOOGLE_API_KEY[:6] + "..." + GOOGLE_API_KEY[-4:]
        st.caption(f"API key loaded: {masked_key}")

    st.session_state.debug_mode = st.toggle(
        "Debug mode",
        value=st.session_state.debug_mode
    )

    if st.button("List available Gemini models"):
        try:
            available_models = []

            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)

            if available_models:
                st.write("Available generateContent models:")
                st.code("\n".join(available_models))
            else:
                st.warning("No generateContent models were returned.")

        except Exception as e:
            st.error("Could not list models.")
            st.code(str(e))

    if st.button("Clear chat"):
        st.session_state.raw_history = []
        st.rerun()

    st.divider()

    st.caption(
        "Citta Companion is not a diagnosis, therapy, crisis service, or emergency support."
    )


# ------------------------------------------------------------
# Opening message
# This is displayed only.
# It is NOT sent to Gemini history.
# ------------------------------------------------------------

OPENING_MESSAGE = """
Hello, I’m **Citta Companion**.

I’m here to support your wellbeing discovery in a gentle and confidential way.

Your individual responses are not shared with your employer. Your employer may receive only de-identified and aggregated wellbeing insights.

I’m not a therapist, diagnosis tool, crisis service, or emergency service.

How are you feeling today?
"""

if not st.session_state.raw_history:
    with st.chat_message("assistant"):
        st.markdown(OPENING_MESSAGE)


# ------------------------------------------------------------
# Display existing chat history
# ------------------------------------------------------------

for msg in st.session_state.raw_history:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    if role not in ["user", "assistant"]:
        continue

    with st.chat_message(role):
        st.markdown(content)


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    # Store user message
    st.session_state.raw_history.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Check red flag
    red_flag_detected = detect_red_flag(user_prompt)

    if red_flag_detected:
        st.warning(
            "It sounds like you may be experiencing serious distress or safety concerns. "
            "Citta Companion is not an emergency or crisis service. "
            "If you may be unsafe, please contact local emergency services, a trusted person, "
            "or a human support professional immediately."
        )

        send_risk_alert(
            employee_id=EMPLOYEE_ID,
            sector=EMPLOYEE_SECTOR,
            message=user_prompt,
            reason="Red-flag keyword detected"
        )

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                gemini_history = to_gemini_history(st.session_state.raw_history)

                if not gemini_history:
                    st.error("No valid user message was available for Gemini.")
                    st.stop()

                if gemini_history[-1]["role"] != "user":
                    st.error("The latest Gemini message must be from the user.")

                    if st.session_state.debug_mode:
                        st.json(gemini_history)

                    st.stop()

                response = model.generate_content(
                    gemini_history,
                    generation_config=genai.GenerationConfig(
                        temperature=0.4,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=900
                    )
                )

                assistant_reply = extract_response_text(response)

            except BadRequest as e:
                st.error("Gemini rejected the request payload.")

                if st.session_state.debug_mode:
                    st.subheader("BadRequest details")
                    st.code(str(e))

                    st.subheader("Gemini history sent to API")
                    st.json(to_gemini_history(st.session_state.raw_history))

                    st.subheader("Raw Streamlit history")
                    st.json(st.session_state.raw_history)

                st.stop()

            except GoogleAPIError as e:
                st.error("Google API error while generating the response.")

                if st.session_state.debug_mode:
                    st.code(str(e))

                    st.subheader("Active model")
                    st.code(MODEL_NAME)

                st.stop()

            except Exception:
                st.error("Unexpected app error.")

                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())

                st.stop()

        st.markdown(assistant_reply)

    # Store assistant message
    st.session_state.raw_history.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )
