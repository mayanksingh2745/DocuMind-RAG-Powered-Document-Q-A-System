import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import DocumentInfo, QueryRequest
from app.services.document_processor import process_document
from app.services.rag_engine import rag_engine
from app.core.config import settings

router = APIRouter()

@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract text (using OCR if necessary), 
    and index it into the vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Process and index the document
        chunks = process_document(file_path)
        rag_engine.add_documents(chunks)
        
        return DocumentInfo(
            filename=file.filename,
            message=f"Successfully processed and indexed {len(chunks)} chunks from the document."
        )
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_document(request: QueryRequest):
    """
    Query the indexed documents. Returns a streaming response in standard Server-Sent Events (SSE) format.
    """
    if not rag_engine.vectorstore:
        raise HTTPException(status_code=400, detail="No documents indexed. Please upload a document first.")
        
    return StreamingResponse(
        rag_engine.astream_answer(request.question), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
