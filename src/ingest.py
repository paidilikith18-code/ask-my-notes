from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


# --------------------------------
# Configuration
# --------------------------------

DATA_DIR = Path("data/notes")
FAISS_DIR = "faiss_index"


# --------------------------------
# Load documents
# --------------------------------
documents = []

for file_path in DATA_DIR.iterdir():

    # Skip directories
    if not file_path.is_file():
        continue
    # Skip empty files
    if file_path.stat().st_size == 0:
        print(f"Skipping empty file: {file_path.name}")
        continue
    if file_path.suffix == ".txt":
        loader = TextLoader(str(file_path))
    elif file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix == ".md":
        loader = TextLoader(str(file_path))
    else:
        print(f"Skipping unsupported file: {file_path.name}")
        continue
    print(f"Loading: {file_path.name}")
    documents.extend(loader.load())
print(f"\nLoaded documents: {len(documents)}")


# --------------------------------
# Split documents
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created chunks: {len(chunks)}")


# --------------------------------
# Create embeddings
# --------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------
# Build FAISS
# --------------------------------

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created.")


# --------------------------------
# Save FAISS
# --------------------------------

vector_store.save_local(FAISS_DIR)

print(f"FAISS index saved to: {FAISS_DIR}")