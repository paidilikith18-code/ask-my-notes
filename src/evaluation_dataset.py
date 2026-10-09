
evaluation_dataset = [
    {
        "question": "What is retrieval augmented generation?",
        "expected_source": "rag.md",
        "reference_answer": (
            "RAG retrieves relevant information from external "
            "documents and uses it to generate an answer."
        ),
        "answerable": True,
    },
    {
        "question": "What is Python?",
        "expected_source": "python.txt",
        "reference_answer": (
            "Python is a programming language."
        ),
        "answerable": True,
    },
    {
        "question": "What is artificial intelligence?",
        "expected_source": "ai.txt",
        "reference_answer": (
            "AI refers to systems designed to perform tasks "
            "associated with human intelligence."
        ),
        "answerable": True,
    },
    {
        "question": "What is the capital of Mars?",
        "expected_source": None,
        "reference_answer": None,
        "answerable": False,
    },
]