import os
import pickle
import json
from typing import List, AsyncGenerator
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from app.core.config import settings

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            streaming=True
        )
        
        self.faiss_index_path = os.path.join(settings.INDEX_DIR, "faiss_index")
        self.bm25_index_path = os.path.join(settings.INDEX_DIR, "bm25_index.pkl")
        
        self.vectorstore = None
        self.bm25_retriever = None
        self._ranker = None
        
        self._load_indices()

    @property
    def ranker(self) -> Ranker:
        """Lazily initialize the FlashRank ranker to optimize startup time."""
        if self._ranker is None:
            self._ranker = Ranker()
        return self._ranker

    def _load_indices(self):
        """Load indices from disk if they exist."""
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.bm25_index_path):
            self.vectorstore = FAISS.load_local(
                self.faiss_index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            with open(self.bm25_index_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)
        else:
            self.vectorstore = None
            self.bm25_retriever = None

    def add_documents(self, documents: List[Document]):
        """Add documents to the indices and save them to disk."""
        if not documents:
            return

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.bm25_retriever = BM25Retriever.from_documents(documents)
        else:
            self.vectorstore.add_documents(documents)
            # BM25 doesn't easily support adding docs, so we recreate it with all docs
            all_docs = list(self.vectorstore.docstore._dict.values())
            self.bm25_retriever = BM25Retriever.from_documents(all_docs)

        # Save to disk
        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        self.vectorstore.save_local(self.faiss_index_path)
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

    def get_retriever(self, k: int = 4):
        """Get the ensemble retriever combining FAISS and BM25."""
        if not self.vectorstore or not self.bm25_retriever:
            raise ValueError("No documents have been indexed yet.")
        
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.bm25_retriever.k = k
        
        # Combine dense and sparse retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

    def rerank_documents(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        """Rerank hybrid-retrieved documents using FlashRank."""
        if not docs:
            return []
        
        passages = []
        for idx, doc in enumerate(docs):
            passages.append({
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata
            })
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Convert top_n results back into LangChain Document objects
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(Document(
                page_content=res["text"],
                metadata=res["meta"]
            ))
        return reranked_docs

    def get_qa_prompt(self, question: str, docs: List[Document]) -> str:
        """Construct a prompt that enforces STRICT grounding and inline citations."""
        context_parts = []
        for idx, doc in enumerate(docs):
            doc_idx = idx + 1
            context_parts.append(
                f"[{doc_idx}] (Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 1)})\n"
                f"Content: {doc.page_content}"
            )
        
        context_str = "\n\n".join(context_parts)
        
        prompt_template = (
            "You are DocuMind, an AI assistant for answering questions based on provided document context.\n"
            "Your answers must be grounded STRICTLY in the provided context. If the answer cannot be found in the context, state that clearly.\n"
            "You must cite your sources by referencing their bracketed number (e.g. [1], [2]) right after the statement that relies on it. "
            "Keep the answer concise and well-grounded in the given sources.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt_template

    async def astream_answer(self, question: str) -> AsyncGenerator[str, None]:
        """Stream the answer in standardized SSE format."""
        if not self.vectorstore:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Please upload a document first.'})}\n\n"
            return

        try:
            # 1. Retrieve top candidates (k=15) from sparse + dense ensemble
            retriever = self.get_retriever(k=15)
            raw_docs = retriever.invoke(question)
            
            if not raw_docs:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant context found.'})}\n\n"
                return

            # 2. Rerank using FlashRank to select the top 5
            reranked_docs = self.rerank_documents(question, raw_docs, top_n=5)
            
            # 3. Stream citations to the client first
            citations = []
            for idx, doc in enumerate(reranked_docs):
                citations.append({
                    "index": idx + 1,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1),
                    "content": doc.page_content
                })
            
            yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
            
            # 4. Generate & stream prompt response token-by-token
            prompt = self.get_qa_prompt(question, reranked_docs)
            
            async for chunk in self.llm.astream(prompt):
                token = chunk.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# Global instance
rag_engine = RAGEngine()
