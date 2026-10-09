from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


# --------------------------------
# Load embeddings
# --------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------
# Load FAISS
# --------------------------------

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------
# Test question
# --------------------------------

query = "What is retrieval augmented generation?"


# --------------------------------
# Similarity search with scores
# --------------------------------

results = vector_store.similarity_search_with_score(
    query,
    k=5
)


# --------------------------------
# Display results
# --------------------------------

for i, (document, score) in enumerate(results):

    print("\n" + "=" * 60)

    print(f"RESULT {i + 1}")

    print("Distance:", score)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)