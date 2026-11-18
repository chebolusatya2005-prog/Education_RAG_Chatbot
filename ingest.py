from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_FILES = [
    "data/Education_policy.txt",
    "data/education_rag_chatbot.pdf"
]

VECTORSTORE_PATH = "vectorstore"


def create_vectorstore():

    print("Loading policy documents...")

    documents = []

    for file_path in DATA_FILES:

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents.extend(loader.load())
        print(f"Loaded: {file_path}")

    print(f"Total documents loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)

    print("Vectorstore created successfully!")


if __name__ == "__main__":
    create_vectorstore()
