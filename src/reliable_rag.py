
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

if not (INDEX_DIR / "index.faiss").is_file():
    raise RuntimeError(
        "FAISS index is missing. Run the ingestion script first."
    )


# 1. Load the existing vector index
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)


# 2. Configure the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# 3. Define instructions for grounded answers
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an assistant that answers questions about the user's notes.

Use only the supplied context.
Do not invent facts or use outside knowledge.
If the context does not contain the answer, respond exactly:
I couldn't find that information in your notes."""
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    ),
])

answer_chain = prompt | llm


def answer_question(question: str) -> dict:
    """Retrieve relevant notes and generate a grounded answer."""

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # 4. Retrieve the three most relevant chunks
    documents = vector_store.similarity_search(
        question,
        k=3,
    )

    # 5. Handle missing retrieval results
    if not documents:
        return {
            "answer": (
                "I couldn't find that information in your notes."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # 6. Generate an answer using the retrieved context
    response = answer_chain.invoke({
        "context": context,
        "question": question,
    })

    # 7. Collect source names and page numbers where available
    sources = []

    for doc in documents:
        source = Path(
            doc.metadata.get("source", "Unknown source")
        ).name

        page = doc.metadata.get("page")

        citation = (
            f"{source}, page {page + 1}"
            if isinstance(page, int)
            else source
        )

        if citation not in sources:
            sources.append(citation)

    # 8. Return structured output
    return {
        "answer": response.content,
        "sources": sources,
    }


if __name__ == "__main__":
    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            result = answer_question(question)

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCES:")
            if result["sources"]:
                for source in result["sources"]:
                    print(f"- {source}")
            else:
                print("- No sources found")

        except Exception as error:
            # Show the error for debugging during development.
            print(f"\nPipeline error: {error}")