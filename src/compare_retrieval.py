
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
    allow_dangerous_deserialization=True,
)

# 2. Evaluation questions
# Ensure each expected source really contains the answer.
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

strategies = {
    "similarity": vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K},
    ),
    "mmr": vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": K, "fetch_k": 10},
    ),
}

# 3. Evaluate each strategy
for strategy_name, retriever in strategies.items():
    hits = 0
    reciprocal_rank_sum = 0.0

    print(f"\n{'=' * 60}")
    print(f"STRATEGY: {strategy_name.upper()}")
    print("=" * 60)

    for item in evaluation_data:
        question = item["question"]
        expected_source = item["expected_source"]

        documents = retriever.invoke(question)

        retrieved_sources = [
            Path(
                doc.metadata.get("source", "Unknown")
            ).name
            for doc in documents
        ]

        print(f"\nQuestion: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved: {retrieved_sources}")

        if expected_source in retrieved_sources:
            rank = retrieved_sources.index(expected_source) + 1
            hits += 1
            reciprocal_rank_sum += 1 / rank

            print(f"Expected source rank: {rank}")
        else:
            print("Expected source not found in top results")

    count = len(evaluation_data)

    if count == 0:
        raise ValueError("Evaluation dataset is empty.")

    recall_at_k = hits / count
    mrr = reciprocal_rank_sum / count

    print(f"\nRecall@{K}: {recall_at_k:.2%}")
    print(f"MRR: {mrr:.3f}")