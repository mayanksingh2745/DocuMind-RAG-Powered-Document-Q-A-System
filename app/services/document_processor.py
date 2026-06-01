import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def process_document(file_path: str) -> List[Document]:
    """
    Extracts text from a PDF document. Tries fast digital extraction using PyPDFLoader first.
    If the extracted text is empty or extremely short (e.g. scanned PDF), falls back to
    UnstructuredPDFLoader with hi-res OCR.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    docs = []
    
    # 1. Attempt digital text extraction (fast)
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        # Fall back if loading fails entirely
        docs = []

    # Check if the extracted content is empty or extremely short (e.g. image-only PDF)
    total_text_len = sum(len(doc.page_content.strip()) for doc in docs)
    
    if total_text_len < 100:
        # 2. Fall back to OCR-based extraction (slower but accurate for scanned docs)
        try:
            loader = UnstructuredPDFLoader(file_path, strategy="hi_res")
            docs = loader.load()
        except Exception as e:
            # If everything fails, raise a descriptive exception
            raise RuntimeError(f"Failed to extract text from PDF via digital and OCR loaders: {str(e)}")

    # 3. Standardize metadata (1-indexed page number and source filename)
    for doc in docs:
        doc.metadata["source"] = filename
        
        # PyPDFLoader uses 0-indexed "page". Standardize to 1-indexed.
        if "page" in doc.metadata:
            doc.metadata["page"] = int(doc.metadata["page"]) + 1
        elif "page_number" in doc.metadata:
            doc.metadata["page"] = int(doc.metadata["page_number"])
        else:
            doc.metadata["page"] = 1

    # Split documents into chunks while preserving metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(docs)
    return chunks
