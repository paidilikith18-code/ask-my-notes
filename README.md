# Ask My Notes — Retrieval-Augmented Generation (RAG)

Ask My Notes is a document question-answering application that helps users retrieve information from personal notes using semantic search and large language models.

## Features

* Document ingestion from supported text, Markdown, and PDF files
* Text chunking for retrieval
* Embeddings using OpenAI's embedding models
* FAISS vector search
* Experiments with similarity search, MMR, and hybrid retrieval
* RAG-based answer generation using LangChain
* Source attribution for retrieved documents
* Retrieval evaluation using Recall@K and Mean Reciprocal Rank (MRR)
* Answer-quality evaluation for faithfulness, relevance, and correctness
* FastAPI backend with request validation
* Web-based chat interface

## Tech Stack

* Python
* LangChain
* OpenAI API
* FAISS
* FastAPI
* Pydantic
* HTML, CSS, and JavaScript

## Architecture

1. **Ingestion:** Load documents and split them into chunks.
2. **Embedding:** Convert chunks into vector representations.
3. **Indexing:** Store vectors and metadata in FAISS.
4. **Retrieval:** Retrieve relevant chunks for a user question.
5. **Generation:** Provide the retrieved context to the language model.
6. **Response:** Return the generated answer and source information through FastAPI.
7. **Evaluation:** Measure retrieval performance and assess answer quality.

## Local Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Ask-my-notes
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Never commit your actual API key.

### 5. Prepare the vector index

Follow the project's ingestion instructions to create the FAISS index from your documents.

### 6. Start the API

```bash
python -m uvicorn api:app --app-dir src --reload
```

Open `http://127.0.0.1:8000/` for the interface and `http://127.0.0.1:8000/docs` for API documentation.

## Evaluation

The project includes experiments for:

* Recall@3
* Mean Reciprocal Rank (MRR)
* Comparison of similarity, MMR, and hybrid retrieval
* LLM-based assessment of faithfulness, relevance, and correctness

Add your actual measured results here after running the evaluation scripts.

## Limitations

* Answer quality depends on the quality and coverage of the source documents.
* LLM-based evaluation can be inconsistent and should be checked against human-reviewed examples.
* API usage may incur costs.
* The project requires additional security and operational work before production use.

## Future Improvements

* Automated tests and continuous integration
* Authentication and rate limiting
* More robust retrieval and reranking
* Improved evaluation with a larger test dataset
* Monitoring, deployment automation, and usage analytics

## Security

Do not commit API keys, private documents, or sensitive evaluation data. Configure secrets through environment variables in the deployment environment.
