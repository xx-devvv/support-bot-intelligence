import streamlit as st
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import base64

#page configuration
st.set_page_config(page_title="SupportBot", page_icon="🤖", layout="centered")
load_dotenv()

st.title("Customer Support Intelligence Bot")
st.markdown("### Kindly upload a ticket screenshot for assessment")

#connecting to open router
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ API Key not found! Check your .env file.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


#to fetch vision models only
@st.cache_data(ttl=3600)
def get_vision_models():
    """Scans OpenRouter for models that can SEE images"""
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        all_models = response.json()["data"]

        #targetting vision models
        vision_models = []
        for m in all_models:
            if m['id'].endswith(':free'):
                # Check for vision capability keywords
                if any(x in m['id'] for x in ['vision', 'vl', 'gemini', 'free']):
                    vision_models.append(m['id'])


        return sorted(vision_models, key=lambda x: ('qwen' not in x, 'llama' not in x))
    except:
        #Reliable backup models
        return [
            "qwen/qwen-2.5-vl-72b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "google/gemini-2.0-flash-exp:free"
        ]


#Sidebar controls
with st.sidebar:
    st.header("⚙️ Brain Settings")

    # The Dropdown to fix 429 Errors
    models = get_vision_models()
    selected_model = st.selectbox("Choose AI Model", models, index=0)

    st.divider()
    st.write("📷 **Upload Evidence:**")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Analyzing Screenshot...", use_container_width=True)

#chat engine
if "messages" not in st.session_state:
    st.session_state.messages = []


def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')


#displaying chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("What is the error here?"):

    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Build the Payload
    messages_payload = [{"role": "system",
                         "content": "You are a Level 2 Technical Support Agent. Analyze the image and text provided."}]

    # Add conversation history
    for msg in st.session_state.messages[-4:]:
        messages_payload.append(msg)

    # 3. Attach Image (Crucial Step)
    if uploaded_file:
        base64_img = encode_image(uploaded_file)
        # We must attach the image to the latest user message
        # Remove the text-only version we just saved to history
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
                model=selected_model,
                messages=messages_payload,
                stream=True,
                extra_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "SupportBot"}
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"⚠️ Error with {selected_model}: {e}")
            st.info("💡 Tip: Use the dropdown in the sidebar to try a different model (like Qwen or Llama)!")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
