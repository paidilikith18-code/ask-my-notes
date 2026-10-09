
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Load the existing vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 2. Small evaluation dataset.
# Each expected_source must actually contain the answer.
evaluation_data = [
    {
        "question": "What is retrieval augmented generation?",
        "expected_source": "rag.md",
    },
    {
        "question": "What is Python?",
        "expected_source": "python.txt",
    },
    {
        "question": "What is artificial intelligence?",
        "expected_source": "ai.txt",
    },
]

K = 3

total_recall = 0
total_reciprocal_rank = 0

print("\nRAG RETRIEVAL EVALUATION")
print("=" * 60)

for item in evaluation_data:
    question = item["question"]
    expected_source = item["expected_source"]

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=K
    )

    # Print retrieved sources and scores
    retrieved_sources = []

    print(f"\nQuestion: {question}")
    print(f"Expected source: {expected_source}")
    print("\nRetrieved results:")

    for rank, (document, score) in enumerate(results, start=1):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).name

        retrieved_sources.append(source)

        print(
            f"{rank}. {source} | "
            f"relevance score: {score:.3f}"
        )

    # Recall@K for one expected source
    found = expected_source in retrieved_sources

    if found:
        total_recall += 1
        rank = retrieved_sources.index(expected_source) + 1
        reciprocal_rank = 1 / rank
        total_reciprocal_rank += reciprocal_rank

        print(f"Result: FOUND at rank {rank}")
    else:
        print("Result: EXPECTED SOURCE NOT FOUND")

# 3. Calculate average metrics
count = len(evaluation_data)

recall_at_k = total_recall / count
mrr = total_reciprocal_rank / count

print("\n" + "=" * 60)
print(f"Recall@{K}: {recall_at_k:.2%}")
print(f"MRR:       {mrr:.3f}")
print("=" * 60)