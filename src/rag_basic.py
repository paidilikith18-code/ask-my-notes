from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


load_dotenv()


# 1. Load embeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 2. Load FAISS

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Create LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 4. User question

question = "What are large language models?"

# 5. Retrieve relevant documents

documents = vector_store.similarity_search(
    question,
    k=3
)

# 6. Create context

context = "\n\n".join(
    document.page_content
    for document in documents
)

# 7. Create prompt


prompt = f"""
You are "Ask My Notes", an assistant that answers
questions using ONLY the provided context.

Rules:

1. Answer only using information from the context.
2. Do not use your outside knowledge.
3. Do not make up information.
4. If the context does not contain the answer,say exactly:
"I couldn't find that information in your notes."

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

# 8. Ask the LLM

response = llm.invoke(prompt)


# 9. Print answer

print("\nAnswer:")
print(response.content)