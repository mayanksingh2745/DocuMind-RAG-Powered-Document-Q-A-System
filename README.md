# DocuMind — RAG-Powered Document Q&A System

DocuMind is an AI-powered system designed to process, index, and query PDF documents using Retrieval-Augmented Generation (RAG). By combining dense vector search (FAISS) and sparse keyword search (BM25), it provides highly accurate, citation-backed answers while minimizing LLM hallucinations.

## Features

- **Hybrid Retrieval System**: Uses an ensemble of FAISS (dense embeddings via OpenAI) and BM25 (sparse retrieval) to significantly reduce hallucination rates compared to vanilla LLM baselines.
- **Robust OCR Processing**: Extracts text and processes images within complex PDFs using the `unstructured` library, powered by Tesseract and Poppler.
- **Citation-Backed Answers**: The LLM streams responses back in real-time, strictly grounding its answers in the provided context and citing the source document.
- **Streaming FastAPI Backend**: Efficiently streams tokens back to the client for a responsive user experience.
- **One-Command Deployment**: Fully containerized with Docker and Docker Compose for seamless setup.

## Tech Stack

- **Backend**: FastAPI (Python)
- **RAG Framework**: LangChain, LangChain-OpenAI
- **Vector Database**: FAISS
- **Sparse Retrieval**: `rank_bm25`
- **Embeddings**: OpenAI `text-embedding-3-small`
- **LLM**: OpenAI `gpt-4o-mini` (or `gpt-3.5-turbo`)
- **Document Parsing**: `unstructured[pdf]`, `pdf2image`, `tesseract-ocr`
- **Deployment**: Docker

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed on your machine.
- An [OpenAI API Key](https://platform.openai.com/).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mayanksingh2745/DocuMind-RAG-Powered-Document-Q-A-System.git
   cd DocuMind-RAG-Powered-Document-Q-A-System
   ```

2. **Configure Environment Variables:**
   Rename `.env.example` to `.env` and add your OpenAI API Key:
   ```bash
   cp .env.example .env
   # Edit .env and insert: OPENAI_API_KEY=sk-your_key_here
   ```

3. **Start the Application:**
   Run the following command to build the Docker image and start the server:
   ```bash
   docker-compose up --build
   ```
   *Note: The initial build might take a few minutes as it installs Tesseract, Poppler, and Python dependencies.*

## API Usage

Once the server is running, you can access the interactive Swagger UI at:
[http://localhost:8000/docs](http://localhost:8000/docs)

### 1. Upload a Document
- **Endpoint**: `POST /api/upload`
- **Description**: Upload a PDF file to be processed, chunked, and indexed into the vector store.
- **Request**: `multipart/form-data` with the file.

### 2. Query the Document
- **Endpoint**: `POST /api/query`
- **Description**: Ask questions about the uploaded documents. The response will be a server-sent event (SSE) stream containing the answer and citations.
- **Request Body**:
  ```json
  {
    "question": "What is the main topic of the uploaded document?"
  }
  ```

## Local Persistence
The FAISS vector index and BM25 sparse index are automatically saved to the `./data/index` directory on your local machine via Docker volumes. This means you won't lose your indexed documents when the container restarts. Uploaded PDFs are temporarily stored in `./data/uploads`.
