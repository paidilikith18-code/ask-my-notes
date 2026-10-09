from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()


# --------------------------------
# 1. Load the notes
# --------------------------------

loader = TextLoader("data/notes/ai.txt")

documents = loader.load()


# --------------------------------
# 2. Split the notes
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------
# 3. Create embeddings
# --------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------
# 4. Create FAISS vector store
# --------------------------------

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created!")


# --------------------------------
# 5. Save FAISS locally
# --------------------------------

vector_store.save_local("faiss_index")

print("FAISS index saved!")