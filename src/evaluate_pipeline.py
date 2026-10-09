
import csv
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from evaluation_dataset import evaluation_dataset

load_dotenv()

# 1. Configuration
K = 3
OUTPUT_FILE = "evaluation_report.csv"

# 2. Load embeddings and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# 3. Store results and metrics
report_rows = []
total_hits = 0
total_reciprocal_rank = 0.0

print("\nAUTOMATED RAG EVALUATION")
print("=" * 60)

# 4. Evaluate each question
for item in evaluation_dataset:
    question = item["question"]
    expected_source = item["expected_source"]

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=K,
    )

    retrieved_sources = []
    result_details = []

    for rank, (document, score) in enumerate(results, start=1):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).name

        retrieved_sources.append(source)

        result_details.append(
            f"{rank}:{source} ({score:.3f})"
        )

    # 5. Calculate retrieval metrics for this question
    if expected_source in retrieved_sources:
        total_hits += 1

        first_rank = retrieved_sources.index(expected_source) + 1
        reciprocal_rank = 1 / first_rank
    else:
        first_rank = None
        reciprocal_rank = 0.0

    total_reciprocal_rank += reciprocal_rank

    print(f"\nQuestion: {question}")
    print(f"Expected: {expected_source}")
    print(f"Retrieved: {', '.join(result_details) or 'None'}")
    print(f"Expected source rank: {first_rank or 'Not found'}")

    report_rows.append({
        "question": question,
        "expected_source": expected_source or "",
        "retrieved_sources": "; ".join(retrieved_sources),
        "expected_source_rank": first_rank or "",
        f"recall_at_{K}": int(first_rank is not None),
        "reciprocal_rank": f"{reciprocal_rank:.3f}",
        "retrieval_scores": "; ".join(
            f"{score:.3f}" for _, score in results
        ),
    })

# 6. Calculate aggregate metrics
question_count = len(evaluation_dataset)

if question_count == 0:
    raise ValueError("Evaluation dataset is empty.")

recall_at_k = total_hits / question_count
mrr = total_reciprocal_rank / question_count

# 7. Save a CSV report
fieldnames = list(report_rows[0].keys())

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(report_rows)

print("\n" + "=" * 60)
print(f"Questions evaluated: {question_count}")
print(f"Recall@{K}: {recall_at_k:.2%}")
print(f"MRR: {mrr:.3f}")
print(f"Report saved to: {OUTPUT_FILE}")
print("=" * 60)