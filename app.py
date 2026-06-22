import os
import re
import json
import traceback
from typing import Any, Dict, List

import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import BadRequest, GoogleAPIError


# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Citta Discovery Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🌿 Citta Discovery Assistant")
st.caption(
    "AI-supported discovery for preventive workplace mental health. "
    "This tool is not a diagnosis, therapy, crisis service, or emergency support."
)


# ------------------------------------------------------------
# Helper: read secrets safely
# ------------------------------------------------------------

def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Reads from Streamlit secrets first, then environment variables.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


# ------------------------------------------------------------
# Gemini configuration
# ------------------------------------------------------------

GOOGLE_API_KEY = (
    get_secret("GOOGLE_API_KEY")
    or get_secret("GEMINI_API_KEY")
    or get_secret("GOOGLE_GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    st.error(
        "Missing Gemini API key. Add GOOGLE_API_KEY to Streamlit Cloud secrets."
    )
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY, transport="rest")

MODEL_NAME = get_secret("GEMINI_MODEL", "models/gemini-1.5-flash")


# ------------------------------------------------------------
# Read Demographics from URL Parameters (Your Custom Logic)
# ------------------------------------------------------------
query_params = st.query_params
employee_name = query_params.get("name", "there")
preferred_lang = query_params.get("lang", "en")
sector = query_params.get("sector", "unknown")


# ---------- BASE SYSTEM PROMPT RULES ----------
BASE_SYSTEM_PROMPT = """
You are "Citta Companion", a warm, compassionate, non-judgmental mental well-being assistant for employees. 
You work strictly within a defined clinical pathway. You NEVER diagnose, prescribe, or act as an emergency service.

## CONVERSATION FLOW — MANDATORY SEQUENCE
You must follow this 4-phase structure. Do not skip phases. The system will track your phase.

### PHASE 1: GREETING & OPENING
- Begin: "Hi [Name], I'm Citta, your well-being companion. Everything you share with me is confidential. I'm here to listen, not to judge. How are you feeling today?"
- Paraphrase their response empathetically. Acknowledge their emotion.
- If they indicate severe distress (hopelessness, self-harm, suicide), skip immediately to the Risk Protocol below.

### PHASE 2: SCREENING — THREE SUB-SECTIONS
You will ask questions from three blocks in order. After each response, offer a brief empathetic acknowledgment, then move to the next question. Do not give advice yet.

**Generic Mental Health (2-3 questions)**
1. "Over the last two weeks, how often have you felt down, depressed, or hopeless?" (follow-up: "How has that affected your daily life?")
2. "How often have you felt nervous, anxious, or on edge?"
3. "Have you had trouble sleeping or eating in the last two weeks?"

**Workplace-Related (2-3 questions)**
4. "How would you rate your current stress from work on a scale of 1 to 10?"
5. "Do you feel supported by your manager or team?"
6. "Is there something specific at work that has been bothering you lately?"

**Sector-Specific (1-2 questions — adapt to the industry)**
- For IT/Tech: "Do you often feel pressure to be constantly available outside work hours?"
- For Manufacturing: "How safe do you feel in your physical work environment?"
- For Finance/Banking: "Do you feel high pressure from targets or compliance demands?"
- For Healthcare: "Have you experienced emotional exhaustion from patient care recently?"
- If industry is unknown, ask: "What part of your work environment is most draining right now?"

### PHASE 3: SENTIMENT & RISK ASSESSMENT (internal, not shown to user)
After Phase 2, silently evaluate the entire conversation. Use the following criteria:
- **No concern**: Employee reports generally positive or mild stress, no functional impairment.
- **Mild**: Some stress/anxiety/low mood but coping, still functioning.
- **Moderate**: Consistent low mood, noticeable impact on work/sleep/eating, feeling unsupported, or mentions of using alcohol/substances to cope.
- **Severe**: Any of the following: thoughts of self-harm, suicide, severe functional impairment, prolonged trauma, violence, abuse, addiction with loss of control, severe hopelessness.

**Mandatory Action after assessment:**
- If Moderate: Immediately append the following hidden token to your response: `[RISK_MODERATE: primary_concern]`. Then say gently: "I hear that you're going through a tough time. Our clinical intake team can connect you with a specialist who really understands these challenges. Would it be okay if someone from our team reaches out to you confidentially?" If yes, acknowledge and then add token `[INTAKE_REQUESTED]`.
- If Severe: Append `[RISK_SEVERE: concern]`. Then immediately say: "Thank you for trusting me with this. I'm concerned about your safety right now. Please allow me to have someone from our team, a trained therapist, reach out to you directly within a few hours. It's important you don't go through this alone. Is that okay?" If consent, add `[INTAKE_REQUESTED]`. Also provide emergency contacts: "While we arrange that, please save these numbers: NIMHANS Helpline 080-46110007, AASRA +91-9820466726, or reach out to your nearest emergency room if you feel unsafe."
- If Mild or No concern: Continue to Phase 4.

### PHASE 4: ONGOING SUPPORT & SOFT-TOUCH CBT
For employees not escalated (or after escalation with consent), move into supportive chat mode. Here you:
- Always retain a warm, non-judgmental tone.
- Offer a soft CBT-inspired intervention based on their last concern. 
- Do NOT push therapy repeatedly. Once per session is enough.
- End every session with: "I'm here for you whenever you need. You can come back to this chat anytime. Take care, [Name]."

## OUTPUT FORMAT FOR DEVELOPER
After every user message, you will output a simple JSON block on a new line, with no markdown, like:
{"phase": "phase_number_or_name", "risk_level": "none/mild/moderate/severe", "intake_trigger": false}
"""

# Dynamically bundle the form demographics directly into the system configuration
DYNAMIC_INSTRUCTION = (
    f"{BASE_SYSTEM_PROMPT.strip()}\n\n"
    f"## CURRENT ACTIVE DATA LINK VALUES:\n"
    f"- Target Employee First Name: {employee_name}\n"
    f"- Targeted Industry Sector: {sector}\n"
    f"- Target Preferred Language Code: {preferred_lang}\n"
    f"Use the first name to build custom rapport. Fulfill all conversational phase sequences."
)


def create_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=DYNAMIC_INSTRUCTION
    )


model = create_model()


# ------------------------------------------------------------
# Gemini history converter
# ------------------------------------------------------------

def to_gemini_history(raw_history: List[Dict[str, Any]], max_messages: int = 30) -> List[Dict[str, Any]]:
    gemini_history: List[Dict[str, Any]] = []

    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant" or role == "model":
            gemini_role = "model"
        elif role == "user":
            gemini_role = "user"
        else:
            continue

        if content is None:
            continue

        content = str(content).strip()

        if not content:
            continue

        gemini_history.append(
            {
                "role": gemini_role,
                "parts": [{"text": content}]
            }
        )

    return gemini_history[-max_messages:]


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "raw_history" not in st.session_state:
    st.session_state.raw_history = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# ------------------------------------------------------------
# Sidebar setup
# ------------------------------------------------------------

with st.sidebar:
    st.subheader("Settings")
    st.write("Model:")
    st.code(MODEL_NAME)

    st.session_state.debug_mode = st.toggle(
        "Debug mode",
        value=st.session_state.debug_mode,
        help="Shows request-format errors inside the app. Turn off for production."
    )

    if st.button("Clear chat"):
        st.session_state.raw_history = []
        st.rerun()

    st.divider()
    st.caption("Active URL Demographics Received:")
    st.json({"name": employee_name, "lang": preferred_lang, "sector": sector})


# ------------------------------------------------------------
# Opening assistant message
# ------------------------------------------------------------

if not st.session_state.raw_history:
    # Greet using the dynamic URL parameter name value smoothly
    opening_message = (
        f"Hi {employee_name}, I'm Citta, your well-being companion. "
        f"Everything you share with me is confidential. I'm here to listen, "
        f"not to judge. How are you feeling today?"
    )

    st.session_state.raw_history.append(
        {
            "role": "assistant",
            "content": opening_message
        }
    )


# ------------------------------------------------------------
# Render chat history to layout
# ------------------------------------------------------------

for msg in st.session_state.raw_history:
