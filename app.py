import os
import traceback
from typing import Any, Dict, List

import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import BadRequest, GoogleAPIError


# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Citta Discovery Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("Citta Discovery Assistant")
st.caption(
    "AI-supported discovery for preventive workplace mental health. "
    "This is not diagnosis, therapy, crisis support, or emergency care."
)


# ------------------------------------------------------------
# Secrets helper
# ------------------------------------------------------------

def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


# ------------------------------------------------------------
# Gemini setup
# ------------------------------------------------------------

GOOGLE_API_KEY = (
    get_secret("GOOGLE_API_KEY")
    or get_secret("GEMINI_API_KEY")
    or get_secret("GOOGLE_GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    st.error("Missing Gemini API key. Add GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = get_secret("GEMINI_MODEL", "gemini-1.5-flash")

SYSTEM_INSTRUCTION = """
You are Citta's AI Discovery Assistant.

Purpose:
- Support early workplace mental health discovery.
- Help employees reflect on workplace stress, burnout signals, emotional load, trauma-related language, psychological safety, and support needs.
- Use a whole-company preventive mental health lens.
- Be warm, professional, trauma-informed, and non-judgmental.

Important boundaries:
- Do not diagnose.
- Do not say the user has PTSD, complex PTSD, depression, anxiety, ADHD, trauma, or any other condition.
- Do not provide therapy, counselling, medical advice, legal advice, or HR determinations.
- Do not replace a clinician, doctor, psychologist, psychiatrist, emergency service, or crisis service.
- Ask only one or two questions at a time.

Risk handling:
- If the user suggests immediate danger, self-harm, harm to others, abuse, or serious safety risk, advise them to contact local emergency services, a trusted person, or crisis support immediately.
- For workplace concerns, encourage appropriate support through HR, manager, EAP/FEAP, clinician, or emergency support depending on urgency.

Style:
- Keep responses concise.
- Use simple language.
- Validate the concern before asking the next question.
- Explain that discovery is to understand support needs, not to judge performance.
"""

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
    st.session_state.debug_mode = True


# ------------------------------------------------------------
# Gemini history converter
# ------------------------------------------------------------

def to_gemini_history(raw_history: List[Dict[str, Any]], max_messages: int = 30) -> List[Dict[str, Any]]:
    """
    Converts Streamlit-style messages into Gemini-compatible history.

    Streamlit style:
        {"role": "assistant", "content": "..."}

    Gemini style:
        {"role": "model", "parts": ["..."]}

    Gemini expects:
        role = "user" or "model"
        parts = [...]
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
            # Skip system/tool/internal messages.
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

    # Gemini should not receive a history that starts with the model.
    while converted and converted[0]["role"] != "user":
        converted.pop(0)

    # Merge consecutive messages with the same role.
    # This avoids invalid or messy history such as user-user or model-model.
    normalised: List[Dict[str, Any]] = []

    for item in converted:
        if normalised and normalised[-1]["role"] == item["role"]:
            normalised[-1]["parts"][0] += "\n\n" + item["parts"][0]
        else:
            normalised.append(item)

    return normalised[-max_messages:]


# ------------------------------------------------------------
# Response text extractor
# ------------------------------------------------------------

def extract_response_text(response: Any) -> str:
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
    st.subheader("App settings")

    st.write("Gemini model:")
    st.code(MODEL_NAME)

    st.session_state.debug_mode = st.toggle(
        "Debug mode",
        value=st.session_state.debug_mode
    )

    if st.button("Clear chat"):
        st.session_state.raw_history = []
        st.rerun()

    st.divider()

    st.caption("Streamlit Cloud secrets should include:")

    st.code(
        """
GOOGLE_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-1.5-flash"
        """.strip(),
        language="toml"
    )


# ------------------------------------------------------------
# Opening message
# IMPORTANT:
# This is displayed only.
# It is NOT stored in raw_history.
# ------------------------------------------------------------

OPENING_MESSAGE = """
Hello, I’m Citta’s AI Discovery Assistant.

I can help explore workplace stress, emotional load, burnout signals, psychological safety, and support needs in a structured and non-judgmental way.

To begin, what would you like support with today?
"""

if not st.session_state.raw_history:
    with st.chat_message("assistant"):
        st.markdown(OPENING_MESSAGE)


# ------------------------------------------------------------
# Render existing chat history
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

                st.stop()

            except Exception:
                st.error("Unexpected app error.")

                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())

                st.stop()

        st.markdown(assistant_reply)

    st.session_state.raw_history.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )
