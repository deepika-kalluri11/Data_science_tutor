import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set Streamlit Page Configuration
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")

st.title("🤖 AI Data Science Tutor")
st.write("Ask any Data Science-related questions!")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Ensure memory persists in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = st.session_state.memory

# Define a Prompt Template for better responses
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="You are a Data Science tutor. Answer only Data Science-related queries. Keep the conversation context in mind.\n\nChat History: {chat_history}\nUser: {question}\nTutor:"
)

# Define the conversation chain with memory
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask your Data Science question...")
if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI response
    response = conversation.run({"question": user_input})

    # Add AI response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response)

# Add a reset button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()

# Display memory content in the sidebar (for debugging)
if st.sidebar.checkbox("Show Memory"):
    st.sidebar.write(st.session_state.memory.load_memory_variables({}))