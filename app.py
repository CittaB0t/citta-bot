import streamlit as st
import google.generativeai as genai
import json
import re

# ---------- CONFIG ----------
# Replace with your actual system prompt from earlier
SYSTEM_PROMPT = """
You are “Citta Companion”, a warm, compassionate, non-judgmental mental well-being assistant for employees.
... [PASTE YOUR FULL SYSTEM PROMPT HERE, EXACTLY AS BEFORE] ...
"""

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Citta Companion", page_icon="🌿", layout="centered")
st.title("🌿 Citta Companion")
st.caption("Confidential well-being support · Your conversation is private")

# ---------- SIDEBAR: Read demographics (passed via URL parameters) ----------
query_params = st.query_params
employee_name = query_params.get("name", ["there"])[0]
preferred_lang = query_params.get("lang", ["en"])[0]
sector = query_params.get("sector", ["unknown"])[0]

# Store in session state so we only inject once
if "demographics_injected" not in st.session_state:
    st.session_state.demographics = {
        "name": employee_name,
        "preferred_language": preferred_lang,
        "sector": sector,
        # You can add more as you collect from Typeform
    }
    st.session_state.demographics_injected = True

# ---------- INITIALISE GEMINI ----------
# API key from Streamlit secrets (or hardcode for MVP — but use secrets in prod)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = st.text_input("Enter your Gemini API key", type="password")
    if not api_key:
        st.warning("Please enter your Gemini API key to start.")
        st.stop()

genai.configure(api_key=api_key)

# Use Gemini 1.5 Flash (cost-effective, fast)
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT,
)

# ---------- SESSION STATE FOR CHAT ----------
if "messages" not in st.session_state:
    # First message from assistant: inject demographics context invisibly
    initial_context = (
        f"[SYSTEM NOTE: This employee is {employee_name}, works in {sector} sector, "
        f"preferred language is {preferred_lang}. Greet them warmly as per Phase 1.]"
    )
    # We'll send this as a hidden user message that the model can see but user cannot
    st.session_state.hidden_context_sent = False
    st.session_state.messages = []
    st.session_state.raw_history = []  # for Gemini API

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- HANDLE INPUT ----------
if prompt := st.chat_input("Type your message here..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build Gemini history: system prompt already set via system_instruction
    # We'll maintain a list of alternating "user" / "model" turns
    if not st.session_state.hidden_context_sent:
        # First send the hidden demographic context so model knows name, sector, language
        st.session_state.raw_history.append({
            "role": "user",
            "parts": [initial_context]
        })
        st.session_state.hidden_context_sent = True

    # Append current user message
    st.session_state.raw_history.append({
        "role": "user",
        "parts": [prompt]
    })

    # Call Gemini
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.generate_content(
                st.session_state.raw_history,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )

        full_response = response.text

        # ---------- EXTRACT HIDDEN JSON BLOCK ----------
        # Regex to find JSON at the end or anywhere
        json_match = re.search(r'\{.*?"phase".*?\}', full_response, re.DOTALL)
        risk_data = None
        if json_match:
            try:
                risk_data = json.loads(json_match.group())
                # Remove the JSON from the text shown to the user
                display_text = full_response[:json_match.start()].strip() + full_response[json_match.end():].strip()
            except:
                display_text = full_response  # fallback
        else:
            display_text = full_response

        # Show only the clean message to the employee
        st.markdown(display_text.strip())

        # Log the risk data (in Streamlit this will be in the app logs; in production, send to your backend)
        if risk_data:
            print(f"[CITTA RISK DATA] {risk_data}")  # visible in Streamlit Cloud logs
            # Here you would trigger an email if intake_trigger is True
            # e.g., if risk_data.get("intake_trigger"): send_email(...)

    # Save to displayed history (clean text) and raw history (for Gemini)
    st.session_state.messages.append({"role": "assistant", "content": display_text.strip()})
    st.session_state.raw_history.append({
        "role": "model",
        "parts": [full_response]  # we keep full response for context, JSON included (model expects it)
    })
