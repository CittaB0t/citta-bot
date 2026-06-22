import os
import traceback
from typing import Any, Dict, List, Optional

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
    "This is not diagnosis, therapy, crisis support, or emergency care."
)


# ------------------------------------------------------------
# Secrets helper
# ------------------------------------------------------------

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
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
# API key setup
# ------------------------------------------------------------

GOOGLE_API_KEY = (
    get_secret("GOOGLE_API_KEY")
    or get_secret("GEMINI_API_KEY")
    or get_secret("GOOGLE_GEMINI_API_KEY")
)

if GOOGLE_API_KEY:
    GOOGLE_API_KEY = str(GOOGLE_API_KEY).strip().strip('"').strip("'")

if not GOOGLE_API_KEY:
    st.error("Missing Gemini API key. Add GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


# ------------------------------------------------------------
# Model selection
# ------------------------------------------------------------

CONFIGURED_MODEL_NAME = get_secret("GEMINI_MODEL", "gemini-3.5-flash")

if CONFIGURED_MODEL_NAME:
    CONFIGURED_MODEL_NAME = str(CONFIGURED_MODEL_NAME).strip().strip('"').strip("'")
else:
    CONFIGURED_MODEL_NAME = "gemini-3.5-flash"


def strip_models_prefix(model_name: str) -> str:
    """
    Converts models/gemini-3.5-flash to gemini-3.5-flash.
    """

    if not model_name:
        return ""

    return model_name.replace("models/", "").strip()


@st.cache_data(ttl=3600, show_spinner=False)
def list_available_generate_content_models() -> List[str]:
    """
    Lists Gemini models available to this API key that support generateContent.
    """

    available_models: List[str] = []

    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", [])

            if "generateContent" in methods:
                available_models.append(m.name)

    except Exception:
        return []

    return available_models


def choose_model(configured_model: str, available_models: List[str]) -> str:
    """
    Chooses a working model.

    Priority:
    1. Use configured model if it is available.
    2. Use preferred current models if available.
    3. Use any available Flash model.
    4. Use first available model.
    5. Fall back to configured model if model listing failed.
    """

    configured_clean = strip_models_prefix(configured_model)

    if available_models:
        for model_name in available_models:
            if strip_models_prefix(model_name) == configured_clean:
                return model_name

        preferred_models = [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
            "gemini-3-flash-preview",
        ]

        for preferred in preferred_models:
            for model_name in available_models:
                if strip_models_prefix(model_name) == preferred:
                    return model_name

        for model_name in available_models:
            if "flash" in strip_models_prefix(model_name):
                return model_name

        return available_models[0]

    return configured_model


AVAILABLE_MODELS = list_available_generate_content_models()
MODEL_NAME = choose_model(CONFIGURED_MODEL_NAME, AVAILABLE_MODELS)


# ------------------------------------------------------------
# Citta system instruction
# ------------------------------------------------------------

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
- Do not ask for unnecessary identifying personal information.

Risk handling:
- If the user suggests immediate danger, self-harm, harm to others, abuse, or serious safety risk, advise them to contact local emergency services, a trusted person, or crisis support immediately.
- For workplace concerns, encourage appropriate support through HR, manager, EAP/FEAP, clinician, or emergency support depending on urgency.

Style:
- Keep responses concise.
- Use simple language.
- Validate the concern before asking the next question.
- Explain that discovery is to understand support needs, not to judge performance.
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
    st.subheader("App settings")

    st.write("Configured model:")
    st.code(CONFIGURED_MODEL_NAME)

    st.write("Active model:")
    st.code(MODEL_NAME)

    if CONFIGURED_MODEL_NAME != MODEL_NAME:
        st.warning(
            "The configured model was not available, so the app selected an available model automatically."
        )

    if GOOGLE_API_KEY:
        masked_key = GOOGLE_API_KEY[:6] + "..." + GOOGLE_API_KEY[-4:]
        st.caption(f"API key loaded: {masked_key}")

    st.session_state.debug_mode = st.toggle(
        "Debug mode",
        value=st.session_state.debug_mode
    )

    if st.button("List available Gemini models"):
        try:
            available_models = list_available_generate_content_models()

            if available_models:
                st.write("Available generateContent models:")
                st.code("\n".join(available_models))
            else:
                st.warning("No generateContent models were returned for this API key.")

        except Exception as e:
            st.error("Could not list models.")
            st.code(str(e))

    if st.button("Clear chat"):
        st.session_state.raw_history = []
        st.rerun()

    st.divider()

    st.caption("Streamlit Cloud secrets should include:")

    st.code(
        """
GOOGLE_API_KEY = "your_google_ai_studio_api_key_here"
GEMINI_MODEL = "gemini-3.5-flash"
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

                    st.subheader("Active model")
                    st.code(MODEL_NAME)

                    st.subheader("Available models")
                    st.code("\n".join(AVAILABLE_MODELS) if AVAILABLE_MODELS else "No models listed.")

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
