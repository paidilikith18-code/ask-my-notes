from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "What is retrieval augmented generation?"

results = vector_store.similarity_search(
    query,
    k=3
)

for i, document in enumerate(results):

    print("\n" + "=" * 60)

    print(f"RESULT {i + 1}")

    print("\nCONTENT:")
    print(document.page_content)

    print("\nMETADATA:")
    print(document.metadata)