
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from evaluation_dataset import evaluation_dataset

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

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# 2. Configure the answer-generating LLM
answer_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer the question using only the supplied context.
If the context does not contain the answer, say:
I couldn't find that information in your notes.
Do not invent facts."""
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    ),
])

# 3. Define the evaluator's output structure
class QualityScores(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    explanation: str


judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).with_structured_output(QualityScores)

judge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evaluate an answer against the supplied evidence.

Score each dimension from 1 to 5:
- faithfulness: Are claims supported by the retrieved context?
- relevance: Does the answer address the question?
- correctness: Does it agree with the reference answer?

Use these anchors:
1 = very poor, 2 = poor, 3 = mixed, 4 = good, 5 = excellent.

Do not award high correctness merely because the answer sounds confident.
If the context conflicts with the reference answer, flag that conflict
in your explanation. Treat the reference as an evaluation target,
not proof that the reference is factually correct."""
    ),
    (
        "human",
        """Question:
{question}

Retrieved context:
{context}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}"""
    ),
])

answer_chain = answer_prompt | answer_llm
judge_chain = judge_prompt | judge_llm

# 4. Evaluate answerable questions with reference answers
test_cases = [
    item for item in evaluation_dataset
    if item.get("answerable", True)
    and item.get("reference_answer")
]

if not test_cases:
    raise ValueError(
        "No answerable questions with reference answers found."
    )

report = []

for item in test_cases:
    question = item["question"]
    reference = item["reference_answer"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    generated = answer_chain.invoke({
        "question": question,
        "context": context,
    }).content

    scores = judge_chain.invoke({
        "question": question,
        "context": context,
        "reference_answer": reference,
        "generated_answer": generated,
    })

    sources = sorted({
        Path(doc.metadata.get("source", "Unknown")).name
        for doc in docs
    })

    row = {
        "question": question,
        "reference_answer": reference,
        "generated_answer": generated,
        "sources": ", ".join(sources),
        "faithfulness": scores.faithfulness,
        "relevance": scores.relevance,
        "correctness": scores.correctness,
        "explanation": scores.explanation,
    }

    report.append(row)

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("GENERATED ANSWER:", generated)
    print("SOURCES:", ", ".join(sources))
    print(
        "SCORES:",
        f"Faithfulness={scores.faithfulness}/5,",
        f"Relevance={scores.relevance}/5,",
        f"Correctness={scores.correctness}/5",
    )
    print("EVALUATOR NOTES:", scores.explanation)

# 5. Save detailed results
output_path = Path("answer_quality_report.json")
output_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# 6. Print average scores
count = len(report)

print("\n" + "=" * 60)
print("AVERAGE ANSWER QUALITY")
print("=" * 60)

for metric in ("faithfulness", "relevance", "correctness"):
    average = sum(row[metric] for row in report) / count
    print(f"{metric.title()}: {average:.2f}/5")

print(f"\nEvaluated {count} questions.")
print(f"Detailed report saved to: {output_path}")