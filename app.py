import os
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

st.title("Citta Discovery Assistant")
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

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = get_secret("GEMINI_MODEL", "gemini-1.5-flash")


SYSTEM_INSTRUCTION = """
You are Citta's AI Discovery Assistant.

Purpose:
- Support early workplace mental health discovery.
- Help employees reflect on stress, burnout, trauma-related language, workplace pressure, emotional load, and support needs.
- Use a whole-company, preventive mental health lens.
- Ask clear, gentle, structured questions.
- Keep the tone warm, professional, trauma-informed, and non-judgmental.

Important boundaries:
- Do not diagnose.
- Do not claim the user has PTSD, complex PTSD, depression, anxiety, trauma, ADHD, or any clinical condition.
- Do not provide therapy or crisis counselling.
- Do not replace a clinician, therapist, doctor, psychologist, psychiatrist, or emergency service.
- Do not make medical, legal, employment, or HR determinations.
- Do not ask for unnecessary identifying personal information.

Risk handling:
- If the user suggests immediate danger, self-harm, harm to others, abuse, or serious safety risk, respond supportively and advise them to contact local emergency services, a trusted person, or a crisis helpline immediately.
- For workplace risk, encourage the user to seek appropriate support through their employer, HR, EAP/FEAP contact, clinician, or emergency support depending on urgency.

Response style:
- Keep responses concise.
- Ask one or two questions at a time.
- Use simple language.
- Reflect the user's concern before asking the next question.
- Where appropriate, explain that the discovery is to understand support needs, not to judge performance.
"""


def create_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION
    )


model = create_model()


# ------------------------------------------------------------
# Gemini history converter
# ------------------------------------------------------------

def to_gemini_history(raw_history: List[Dict[str, Any]], max_messages: int = 30) -> List[Dict[str, Any]]:
    """
    Converts Streamlit/OpenAI-style history:
        {"role": "assistant", "content": "..."}
    into Gemini-style history:
        {"role": "model", "parts": ["..."]}

    Gemini expects role to be "user" or "model".
    It does not accept "assistant" as a role.
    """

    gemini_history: List[Dict[str, Any]] = []

    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            gemini_role = "model"
        elif role == "model":
            gemini_role = "model"
        elif role == "user":
            gemini_role = "user"
        else:
            # Skip system, tool, internal, or invalid messages.
            continue

        if content is None:
            continue

        if isinstance(content, list):
            content = "\n".join(str(item) for item in content if item)
        else:
            content = str(content)

        content = content.strip()

        # Gemini can reject empty parts.
        if not content:
            continue

        gemini_history.append(
            {
                "role": gemini_role,
                "parts": [content]
            }
        )

    return gemini_history[-max_messages:]


# ------------------------------------------------------------
# Response extractor
# ------------------------------------------------------------

def extract_response_text(response: Any) -> str:
    """
    Safely extracts text from a Gemini response.
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

    return "I could not generate a response. Please try again with a shorter or clearer message."


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "raw_history" not in st.session_state:
    st.session_state.raw_history = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# ------------------------------------------------------------
# Sidebar
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

    st.caption(
        "For Streamlit Cloud, add your API key under: "
        "Manage app → Settings → Secrets."
    )

    st.code(
        """
GOOGLE_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-1.5-flash"
        """.strip(),
        language="toml"
    )


# ------------------------------------------------------------
# Opening assistant message
# ------------------------------------------------------------

if not st.session_state.raw_history:
    opening_message = (
        "Hello, I’m Citta’s AI Discovery Assistant. "
        "I can help explore workplace stress, emotional load, burnout signals, "
        "and support needs in a structured and non-judgmental way.\n\n"
        "To begin, what would you like support with today?"
    )

    st.session_state.raw_history.append(
        {
            "role": "assistant",
            "content": opening_message
        }
    )


# ------------------------------------------------------------
# Render chat history
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
    # Add user message to Streamlit history.
    st.session_state.raw_history.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                gemini_history = to_gemini_history(st.session_state.raw_history)

                if not gemini_history:
                    st.error("No valid message history was available for Gemini.")
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
                assistant_reply = (
                    "The request sent to Gemini was rejected. "
                    "This usually means the message history format is invalid, "
                    "too large, empty, or contains a role Gemini does not accept."
                )

                st.error(assistant_reply)

                if st.session_state.debug_mode:
                    st.subheader("BadRequest details")
                    st.code(str(e))

                    st.subheader("Converted Gemini history")
                    st.json(to_gemini_history(st.session_state.raw_history))

                st.stop()

            except GoogleAPIError as e:
                assistant_reply = (
                    "There was a Google API error while generating the response."
                )

                st.error(assistant_reply)

                if st.session_state.debug_mode:
                    st.code(str(e))

                st.stop()

            except Exception as e:
                assistant_reply = (
                    "Something went wrong while generating the response."
                )

                st.error(assistant_reply)

                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())

                st.stop()

        st.markdown(assistant_reply)

    # Add assistant message after successful generation.
    st.session_state.raw_history.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )
