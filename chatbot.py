
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from ingest import create_vectorstore

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


def get_response(question):

    # Get API key from local .env or Streamlit Cloud Secrets
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return "GROQ_API_KEY is missing. Please configure it in your .env file or Streamlit Secrets."

    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Automatically create FAISS database if it does not exist
    index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
    pickle_file = os.path.join(VECTORSTORE_PATH, "index.pkl")

    if not os.path.isfile(index_file) or not os.path.isfile(pickle_file):
        print("Vectorstore not found. Creating it now...")
        create_vectorstore()

    # Load FAISS database
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve relevant policy information
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create the prompt
    prompt = ChatPromptTemplate.from_template("""
You are an educational policy assistant for the
AI Student Learning & Career Success Platform.

Answer the user's question using ONLY the policy context below.

Rules:
1. Do not invent policies or information.
2. If the answer is not available in the context,
   clearly say: "This information is not specified
   in the sample policy handbook."
3. Give clear and beginner-friendly answers.
4. Mention that this is a fictional sample policy
   when relevant.
5. Do not claim that sample policies are official rules.

Policy Context:
{context}

User Question:
{question}

Answer:
""")

    # Connect to Groq
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=api_key
    )

    # Generate the answer
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content
