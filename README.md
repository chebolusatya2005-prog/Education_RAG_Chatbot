# Education RAG Chatbot

An AI-powered educational chatbot built using Retrieval-Augmented Generation (RAG). It answers questions based on the Student, Trainer & Platform Administration Policy Handbook.

## Project Overview

The Education RAG Chatbot helps users understand educational platform policies by retrieving relevant information from PDF and TXT documents and generating context-grounded answers using an LLM.

## Features

- PDF and TXT document processing
- Text splitting and chunking
- Semantic search using Hugging Face embeddings
- FAISS vector database for document retrieval
- Groq LLM integration
- Context-grounded answers using RAG
- Streamlit interactive chatbot interface
- Chat history during the current session
- Fallback response when information is not available in the policy context

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Groq API
- PyPDF
- python-dotenv

## Project Structure

```text
Education_RAG_Chatbot/
│
├── app.py
├── chatbot.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── Education_policy.txt
    └── education_rag_chatbot.pdf
```

The `vectorstore/` directory is generated locally during document ingestion and is excluded from GitHub.

## RAG Architecture

1. Load educational policy documents (PDF and TXT).
2. Split documents into smaller text chunks.
3. Generate embeddings using Hugging Face.
4. Store embeddings in a FAISS vector database.
5. Retrieve relevant chunks based on the user's question.
6. Send the retrieved context and question to the Groq LLM.
7. Generate a context-grounded response.
8. Display the answer through Streamlit.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Education_RAG_Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your `.env` file or expose your API key.

### 6. Generate the FAISS vector database

```bash
python ingest.py
```

### 7. Run the Streamlit application

```bash
streamlit run app.py
```

The chatbot will open in your browser.

## Environment Configuration

The application requires a valid Groq API key.

The embedding model used is:

`sentence-transformers/all-MiniLM-L6-v2`

The Groq model used is:

`openai/gpt-oss-120b`

## Project Purpose

This project demonstrates the practical implementation of Retrieval-Augmented Generation, document processing, vector databases, semantic search, and LLM-based question answering.

**Note:** The educational policy handbook used in this project is a fictional sample document created for learning and demonstration purposes. It does not represent official institutional policies.

## Author

Education RAG Chatbot Project