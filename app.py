import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()
#google_api_key = os.getenv("GOOGLE_API_KEY")

google_api_key = os.getenv("GOOGLE_API_KEY")

# Use Streamlit secrets when deployed
if st.secrets.get("GOOGLE_API_KEY"):
    google_api_key = st.secrets["GOOGLE_API_KEY"]

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Check your environment or Streamlit Secrets.")

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
async def create_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key, temperature=1.0)


# Initialize the Gemini model using Langchain
#model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key, temperature=1.0)
model = loop.run_until_complete(create_model())

# Streamlit UI
st.title("Medical Diagnosis Assistant")
st.write("Describe your symptoms, and I'll diagnose the most likely diseases.")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input for symptoms
symptoms = st.text_input("Enter your symptoms (e.g., headache, nausea, fever):")

# Diagnose Button
if st.button("Diagnose"):
    if symptoms:
        # Add user message to session state
        st.session_state.messages.append(f"**You:** {symptoms}")

        # Create a direct prompt to diagnose top 3 possible diseases
        prompt = f"Based on the symptoms '{symptoms}', list the top 3 most likely diseases with a brief explanation for each."
        response = model.invoke([HumanMessage(content=prompt)])
        
        # Add model response to session state
        st.session_state.messages.append(f"**Bot:** {response.content}")

        # Display conversation
        for message in st.session_state.messages:
            st.write(message)
    else:
        st.warning("⚠️ Please enter symptoms before diagnosing.")

# **Download Button**
if st.session_state.messages:
    chat_history = "\n".join(st.session_state.messages)
    st.download_button(
        label="📥 Download Chat",
        data=chat_history,
        file_name="medical_diagnosis_chat.txt",
        mime="text/plain"
    )

# **Reset Button**
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []  # Clear session state
    st.experimental_rerun()



