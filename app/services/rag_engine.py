import os
import pickle
from typing import List, AsyncGenerator
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
        
        self._load_indices()

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
            # Note: For large scale, you might need a different sparse retriever like ElasticSearch
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

    def get_qa_chain(self):
        template = """You are DocuMind, an AI assistant for answering questions based on provided document context.
Your answers must be grounded STRICTLY in the provided context. If the answer cannot be found in the context, state that clearly.
Always cite your sources by referencing the document name at the end of your answer.

Context: {context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

        retriever = self.get_retriever()
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    async def astream_answer(self, question: str) -> AsyncGenerator[str, None]:
        """Stream the answer to the user."""
        if not self.vectorstore:
            yield "Please upload a document first."
            return

        chain = self.get_qa_chain()
        async for chunk in chain.astream(question):
            yield chunk

# Global instance
rag_engine = RAGEngine()
