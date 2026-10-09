
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 2. Load the existing FAISS index
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Configure the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 4. Configure the prompt
prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions about the user's notes.

Rules:
- Answer using ONLY the supplied context.
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not answer the question,
  say: "I couldn't find that information in your notes."
- Be clear and concise.

Context:
{context}

Question:
{question}

Answer:
""")

# 5. Relevance threshold
# This is an initial value to evaluate, not a universal constant.
RELEVANCE_THRESHOLD = 0.50


def format_source(document):
    source = document.metadata.get("source", "Unknown")
    filename = Path(source).name

    page = document.metadata.get("page")

    if page is not None:
        # PDF page metadata is commonly zero-indexed.
        return f"{filename} — Page {page + 1}"

    return filename


# 6. Ask a question
question = input("\nAsk your notes: ").strip()

if not question:
    print("Please enter a question.")
    raise SystemExit

# 7. Retrieve documents and relevance scores
results = vector_store.similarity_search_with_relevance_scores(
    question,
    k=3
)

# 8. Keep only results that pass the threshold
relevant_documents = [
    (document, score)
    for document, score in results
    if score >= RELEVANCE_THRESHOLD
]

print("\n" + "=" * 60)

if not relevant_documents:
    print("ANSWER:")
    print("I couldn't find that information in your notes.")
    print("\nSOURCES:")
    print("- None: no documents passed the relevance threshold.")

else:
    # 9. Build context only from accepted documents
    documents = [
        document for document, score in relevant_documents
    ]

    context = "\n\n".join(
        document.page_content for document in documents
    )

    # 10. Generate a grounded answer
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    print("ANSWER:")
    print(response.content)

    # 11. Display sources and scores
    print("\nSOURCES:")

    seen_sources = set()

    for document, score in relevant_documents:
        source = format_source(document)

        if source not in seen_sources:
            print(f"- {source} (relevance: {score:.3f})")
            seen_sources.add(source)

print("=" * 60)