import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def process_document(file_path: str) -> List[Document]:
    """
    Extracts text from a PDF document, utilizing OCR for images within the PDF,
    and splits the text into chunks for indexing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use UnstructuredPDFLoader which has built-in OCR support via Tesseract
    # strategy="hi_res" ensures it attempts OCR on images and complex layouts
    loader = UnstructuredPDFLoader(file_path, strategy="hi_res")
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    # Add the source filename to metadata for citations
    filename = os.path.basename(file_path)
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = filename
            
    chunks = text_splitter.split_documents(docs)
    return chunks
