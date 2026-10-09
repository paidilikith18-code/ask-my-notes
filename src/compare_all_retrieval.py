
from pathlib import Path
from time import perf_counter
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from evaluation_dataset import evaluation_dataset

load_dotenv()

# 1. Load the existing vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

documents = list(vector_store.docstore._dict.values())

if not documents:
    raise ValueError("FAISS index contains no documents.")

# 2. Configure the retrieval strategies
K = 3
RRF_K = 60

similarity = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": K},
)

mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": K, "fetch_k": 10},
)

bm25 = BM25Retriever.from_documents(documents)
bm25.k = 5


def doc_key(doc):
    return (
        doc.metadata.get("source", ""),
        doc.metadata.get("page"),
        doc.page_content,
    )


def hybrid_search(question):
    semantic_docs = similarity.invoke(question)
    keyword_docs = bm25.invoke(question)

    scores = defaultdict(float)
    docs_by_key = {}

    for ranked_docs in (semantic_docs, keyword_docs):
        for rank, doc in enumerate(ranked_docs, start=1):
            key = doc_key(doc)
            scores[key] += 1 / (RRF_K + rank)
            docs_by_key[key] = doc

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:K]

    return [docs_by_key[key] for key in ranked_keys]


strategies = {
    "Similarity": similarity.invoke,
    "MMR": mmr.invoke,
    "Hybrid": hybrid_search,
}

# 3. Evaluate answerable questions only
test_cases = [
    item for item in evaluation_dataset
    if item.get(
        "answerable",
        item.get("expected_source") is not None,
    )
    and item.get("expected_source") is not None
]

if not test_cases:
    raise ValueError(
        "No answerable test questions with expected_source found."
    )

# 4. Run the same questions through every strategy
reports = {}

for name, retrieve in strategies.items():
    hits = 0
    reciprocal_rank_sum = 0.0
    elapsed_total = 0.0

    print(f"\n{'=' * 55}")
    print(f"STRATEGY: {name.upper()}")
    print("=" * 55)

    for item in test_cases:
        question = item["question"]
        expected = Path(item["expected_source"]).name

        start = perf_counter()
        results = retrieve(question)
        elapsed = perf_counter() - start
        elapsed_total += elapsed

        sources = [
            Path(doc.metadata.get("source", "Unknown")).name
            for doc in results
        ]

        print(f"\nQuestion: {question}")
        print(f"Expected source: {expected}")
        print(f"Retrieved sources: {sources}")

        if expected in sources:
            rank = sources.index(expected) + 1
            hits += 1
            reciprocal_rank_sum += 1 / rank
            print(f"Expected source rank: {rank}")
        else:
            print("Expected source not found.")

    total = len(test_cases)

    reports[name] = {
        "Recall@3": hits / total,
        "MRR": reciprocal_rank_sum / total,
        "Average latency (ms)": elapsed_total / total * 1000,
    }

# 5. Print a comparison table
print("\n\nFINAL COMPARISON")
print("=" * 72)
print(
    f"{'Strategy':<14}"
    f"{'Recall@3':>12}"
    f"{'MRR':>12}"
    f"{'Avg ms':>14}"
)
print("-" * 72)

for name, metrics in reports.items():
    print(
        f"{name:<14}"
        f"{metrics['Recall@3']:>11.1%}"
        f"{metrics['MRR']:>12.3f}"
        f"{metrics['Average latency (ms)']:>14.2f}"
    )

print("=" * 72)
print(f"Answerable questions evaluated: {len(test_cases)}")