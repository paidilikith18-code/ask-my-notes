from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Create the same embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load our existing FAISS database
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Ask a question
query = "What is deep learning?"

# Search for similar chunks
results = vector_store.similarity_search_with_relevance_scores(
    query,
    k=3
)
# Display results
print("\nSearch results:\n")

for i, (document, score) in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print("Score:", score)
    print("Content:")
    print(document.page_content)
    print("Metadata:")
    print(document.metadata)
    print()