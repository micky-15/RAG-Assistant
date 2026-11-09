"""Dify External Knowledge API over the local Chroma store. Run: uvicorn app:app --port 8000"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from ingest import EMBEDDING_MODEL, PERSIST_DIR
from query import RERANK_MODEL, retrieve

load_dotenv()
API_KEY = os.environ["RETRIEVAL_API_KEY"]  # Dify sends this as "Authorization: Bearer <key>"

app = FastAPI()
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
reranker = CrossEncoder(RERANK_MODEL)


class RetrievalRequest(BaseModel):
    knowledge_id: str = ""
    query: str
    retrieval_setting: dict = {}


@app.post("/retrieval")
def retrieval(req: RetrievalRequest, authorization: str = Header("")):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail={"error_code": 1002, "error_msg": "Authorization failed"})
    k = req.retrieval_setting.get("top_k", 5)
    docs = retrieve(vectorstore, req.query, k=k, reranker=reranker)
    # Same "[Source: ...]" label as query.py's format_docs, so the LLM can tell which document each chunk is from.
    # Dify sorts by score; strictly decreasing scores keep the reranker's order.
    return {"records": [
        {
            "content": f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}",
            "score": 1.0 - i * 0.01,
            "title": f"{d.metadata.get('source', 'unknown')} (page {d.metadata.get('page', '?')})",
            "metadata": d.metadata,
        }
        for i, d in enumerate(docs)
    ]}
