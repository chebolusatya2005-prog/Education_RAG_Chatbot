import streamlit as st
from chatbot import get_response

st.set_page_config(
    page_title="Education RAG Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Education RAG Chatbot")
st.caption("Ask questions about the Student, Trainer & Platform Administration Policy Handbook.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user question
question = st.chat_input("Ask something about the policy handbook...")

if question:
    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Searching the policy handbook..."):
            try:
                answer = get_response(question)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
