from pydantic import BaseModel
from typing import List

class DocumentInfo(BaseModel):
    filename: str
    message: str

class QueryRequest(BaseModel):
    question: str

class Citation(BaseModel):
    source: str
    content: str
    page: int | None = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
