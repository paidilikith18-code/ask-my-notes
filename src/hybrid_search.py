
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# 1. Load the existing FAISS index
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# 2. Load documents from the existing index.
# These are the chunks already stored in the FAISS docstore.
documents = list(vector_store.docstore._dict.values())

if not documents:
    raise ValueError("No documents found in the FAISS index.")

# 3. Create a keyword retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 4. Create a semantic retriever
semantic_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# 5. Ask a question
question = input("\nAsk your notes: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# 6. Retrieve using both methods
semantic_docs = semantic_retriever.invoke(question)
keyword_docs = bm25_retriever.invoke(question)

# 7. Combine rankings using Reciprocal Rank Fusion
rrf_scores = defaultdict(float)
documents_by_id = {}

RRF_K = 60


def document_id(document):
    # A stable identity for this demonstration.
    # Identical text with different metadata may need a
    # more specific identity in a production system.
    return (
        document.metadata.get("source", ""),
        document.metadata.get("page", None),
        document.page_content,
    )


for ranked_docs in (semantic_docs, keyword_docs):
    for rank, document in enumerate(ranked_docs, start=1):
        key = document_id(document)
        rrf_scores[key] += 1 / (RRF_K + rank)
        documents_by_id[key] = document

# 8. Sort by combined RRF score
fused_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True,
)[:5]

# 9. Display results
print("\nHYBRID SEARCH RESULTS")
print("=" * 60)

for rank, (key, score) in enumerate(fused_results, start=1):
    document = documents_by_id[key]
    source = Path(
        document.metadata.get("source", "Unknown")
    ).name

    print(f"\nRank: {rank}")
    print(f"Source: {source}")
    print(f"RRF score: {score:.5f}")
    print(f"Content: {document.page_content[:500]}")

print("\n" + "=" * 60)