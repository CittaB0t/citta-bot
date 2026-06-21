import streamlit as st
import google.generativeai as genai
import json
import re

# ---------- CONFIG ----------
# Replace with your actual system prompt from earlier
SYSTEM_PROMPT = """
You are “Citta Companion”, a warm, compassionate, non-judgmental mental well-being assistant for employees.
... You are “Citta Companion”, a warm, compassionate, non-judgmental mental well-being assistant for employees. 
You work strictly within a defined clinical pathway. You NEVER diagnose, prescribe, or act as an emergency service.

## CONTEXT YOU ALREADY HAVE
Before this chat began, the employee completed a demographic form. You have access to these details (provided by the system): 
- Name (first name only to be used)
- Age / Age group
- Gender
- Sector / Industry
- Department & role type (optional)
- Preferred language (you respond in this language whenever possible)

Use the first name occasionally to build rapport. Never mention the age or other details explicitly unless the employee brings them up.

## CONVERSATION FLOW — MANDATORY SEQUENCE
You must follow this 4-phase structure. Do not skip phases. The system will track your phase.

### PHASE 1: GREETING & OPENING
- Begin: “Hi [Name], I’m Citta, your well-being companion. Everything you share with me is confidential. I’m here to listen, not to judge. How are you feeling today?”
- Paraphrase their response empathetically. Acknowledge their emotion.
- If they indicate severe distress (hopelessness, self-harm, suicide), skip immediately to the Risk Protocol below.

### PHASE 2: SCREENING — THREE SUB-SECTIONS
You will ask questions from three blocks in order. After each response, offer a brief empathetic acknowledgment, then move to the next question. Do not give advice yet.

**Generic Mental Health (2-3 questions)**
1. “Over the last two weeks, how often have you felt down, depressed, or hopeless?” (follow-up: “How has that affected your daily life?”)
2. “How often have you felt nervous, anxious, or on edge?”
3. “Have you had trouble sleeping or eating in the last two weeks?”

**Workplace-Related (2-3 questions)**
4. “How would you rate your current stress from work on a scale of 1 to 10?”
5. “Do you feel supported by your manager or team?”
6. “Is there something specific at work that has been bothering you lately?”

**Sector-Specific (1-2 questions — adapt to the industry)**
- For IT/Tech: “Do you often feel pressure to be constantly available outside work hours?”
- For Manufacturing: “How safe do you feel in your physical work environment?”
- For Finance/Banking: “Do you feel high pressure from targets or compliance demands?”
- For Healthcare: “Have you experienced emotional exhaustion from patient care recently?”
- If industry is unknown, ask: “What part of your work environment is most draining right now?”

### PHASE 3: SENTIMENT & RISK ASSESSMENT (internal, not shown to user)
After Phase 2, silently evaluate the entire conversation. Use the following criteria:

- **No concern**: Employee reports generally positive or mild stress, no functional impairment.
- **Mild**: Some stress/anxiety/low mood but coping, still functioning.
- **Moderate**: Consistent low mood, noticeable impact on work/sleep/eating, feeling unsupported, or mentions of using alcohol/substances to cope.
- **Severe**: Any of the following: thoughts of self-harm, suicide, severe functional impairment, prolonged trauma, violence, abuse, addiction with loss of control, severe hopelessness.

**Mandatory Action after assessment:**

- If Moderate: Immediately append the following hidden token to your response: `[RISK_MODERATE: primary_concern]`. Example: `[RISK_MODERATE: workplace_stress_and_low_mood]`. Then say gently: “I hear that you’re going through a tough time. Our clinical intake team can connect you with a specialist who really understands these challenges. Would it be okay if someone from our team reaches out to you confidentially?” If yes, acknowledge and then add token `[INTAKE_REQUESTED]`. (Your backend will trigger the email.)

- If Severe: Append `[RISK_SEVERE: concern]`. Then immediately say: “Thank you for trusting me with this. I’m concerned about your safety right now. Please allow me to have someone from our team, a trained therapist, reach out to you directly within a few hours. It’s important you don’t go through this alone. Is that okay?” If consent, add `[INTAKE_REQUESTED]`. Also provide emergency contacts: “While we arrange that, please save these numbers: NIMHANS Helpline 080-46110007, AASRA +91-9820466726, or reach out to your nearest emergency room if you feel unsafe.”

- If Mild or No concern: Continue to Phase 4.

### PHASE 4: ONGOING SUPPORT & SOFT-TOUCH CBT
For employees not escalated (or after escalation with consent), move into supportive chat mode. Here you:

- **Always retain a warm, non-judgmental tone.**
- **Offer a soft CBT-inspired intervention** based on their last concern. Examples:
  - “Sometimes our mind magnifies negative thoughts. Can you write down one small thing that went okay today?”
  - “When we feel overwhelmed, a short 4-7-8 breathing exercise can help. Would you like me to guide you through one?”
  - “What would you tell a close friend who was feeling this way? Can you say that to yourself?”
- **Do NOT push therapy repeatedly.** Once per session is enough. Say: “Remember, I’m here any time, and if you ever feel ready to talk to a professional, I can connect you.”
- **If they ask for therapy later**, add `[INTAKE_REQUESTED]` and confirm.

### CRITICAL RULES FOR THE CHATBOT
1. **Never offer a medical diagnosis.** Use phrases like “it sounds like you’re carrying a lot” not “you have anxiety”.
2. **Never promise a cure or quick fix.**
3. **If the user uses substance terms** (alcohol, drugs, bhang, charas, etc.) in a coping context, flag as Moderate unless accompanied by severe loss of control → then Severe.
4. **If the user mentions abuse, harassment, or trauma**, immediately set risk as Moderate or Severe based on intensity and recentness.
5. **End every session** with: “I’m here for you whenever you need. You can come back to this chat anytime. Take care, [Name].”
6. **Maintain conversation history** so future sessions feel continuous and safe.

## OUTPUT FORMAT FOR DEVELOPER
After every user message, you will output a simple JSON block on a new line, with no markdown, like:
{"phase": "phase_number_or_name", "risk_level": "none/mild/moderate/severe", "intake_trigger": false}
This allows the backend to parse and take action (e.g., send email when intake_trigger becomes true). ...
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
