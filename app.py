import streamlit as st
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Customer Support Intelligence", page_icon="🎧", layout="centered")
load_dotenv()

# --- 2. HARDCODED MODEL (The 'Brain') ---
# We are locking this to Qwen 2.5 VL (72B is smarter/better than 7B and usually free)
MODEL_ID = "qwen/qwen-2.5-vl-72b-instruct:free"

st.title("🎧 Customer Support Intelligence Bot")
st.markdown("### Kindly upload a ticket screenshot for assessment")

# --- 3. CONNECT TO OPENROUTER ---
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    # If using Streamlit Cloud, it gets the key from secrets
    if "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        st.error("❌ API Key not found! Check your .env file or Cloud Secrets.")
        st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# --- 4. SIDEBAR (Cleaned Up) ---
with st.sidebar:
    st.header("📂 Ticket Evidence")
    uploaded_file = st.file_uploader("Upload Error Screenshot", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Analyzing Ticket...", use_container_width=True)
    
    st.divider()
    st.info(f"🤖 **System Status:** Online\n🧠 **Model:** {MODEL_ID}")

# --- 5. CHAT ENGINE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Describe the issue or ask for help..."):
    
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. STRICT SYSTEM PROMPT (The 'Persona')
    system_prompt = """You are a dedicated Customer Support Intelligence Bot. 
    Your ONLY goal is to solve the user's technical issues efficiently.
    - If an image is provided, analyze the error message in it and provide a step-by-step fix.
    - If no image is provided, ask clarifying questions to identify the problem.
    - Be polite, professional, and solution-oriented. Do not waste time with small talk."""

    messages_payload = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in st.session_state.messages[-4:]:
        messages_payload.append(msg)

    # 3. Attach Image (If uploaded)
    if uploaded_file:
        base64_img = encode_image(uploaded_file)
        # Remove the text-only version we just saved
        messages_payload.pop() 
        # Add the version with the image attached
        messages_payload.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        })

    # 4. Get Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages_payload,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501", 
                    "X-Title": "SupportBot"
                }
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        
        except Exception as e:
            # Clean Error Message
            st.error("⚠️ System Busy (429 Error) or Model Unavailable.")
            st.warning("💡 Recommendation: Your API Key daily limit might be reached. Try a new Key.")
            print(e) # Print exact error to terminal for debugging

    st.session_state.messages.append({"role": "assistant", "content": full_response})
