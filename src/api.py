from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from reliable_rag import answer_question
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ask My Notes API",
    description="Ask questions about your personal notes using RAG.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value.strip()

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question to ask about your notes",
    )


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health_check():
    return {"status": "healthy"}



@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        result = answer_question(question)

        # Add the logging line here
        logger.info("RAG question completed successfully")

        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception:
        logger.exception("Failed to process RAG question")

        raise HTTPException(
            status_code=500,
            detail="Unable to answer your question right now.",
        )
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app.mount(
    "/",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="frontend",
)