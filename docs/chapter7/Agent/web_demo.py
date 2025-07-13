import streamlit as st
from src.core import Agent
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia, get_current_temperature
from openai import OpenAI

# --- Page configuration ---
st.set_page_config(
    page_title="Tiny Agent Demo",  # Page title
    page_icon="🤖",  # Page icon
    layout="centered",  # Page layout
    initial_sidebar_state="auto",  # Initial status of the sidebar
)

# --- OpenAI client initialization ---
client = OpenAI(
    api_key="sk-quovvfgjdmmrvwiljusggiwvxfiekzicwjgtdvpfqhpmbpqu",
    base_url="https://api.siliconflow.cn/v1",  
)

# --- Agent initialization ---
@st.cache_resource
def load_agent():
    """Create and cache Agent instances."""
    return Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",  # Models used
        tools=[get_current_datetime, search_wikipedia, get_current_temperature],  # Tools available to Agent
    )

agent = load_agent()  # Loading Agent

# --- UI components ---
st.title("🤖 Happy-LLM Tiny Agent")  # Set page title
st.markdown("""Welcome to the Tiny Agent web interface!

Enter your prompt below to see what the Agent does.""")  # 显示Markdown格式的欢迎信息

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show historical chat history when the app is rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Respond to user input
if prompt := st.chat_input("What can I do for you?"):
    # Display user messages in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user messages to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('Thinking...'):
        response = agent.get_completion(prompt)  # Get the Agent's response
        
    # Show assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add Assistant Response to Chat History
    st.session_state.messages.append({"role": "assistant", "content": response})