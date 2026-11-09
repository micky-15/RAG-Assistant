"""Ask questions against the Chroma vector store built by ingest.py."""

import argparse
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from ingest import PERSIST_DIR, EMBEDDING_MODEL

load_dotenv()

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # chunks fetched by vector search before reranking
TOP_K = 5  # chunks passed to the LLM

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer isn't in the context, say I dont know.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs) -> str:
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)


def retrieve(vectorstore, question: str, k: int = TOP_K, reranker=None, candidates: int = CANDIDATES):
    """Vector search, then (optionally) re-sort the candidates with a cross-encoder and keep the best k."""
    if reranker is None:
        return vectorstore.similarity_search(question, k=k)
    docs = vectorstore.similarity_search(question, k=max(candidates, k))
    scores = reranker.predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: -pair[0])
    return [d for _, d in ranked[:k]]


def build_rag(persist_dir: str = PERSIST_DIR, model: str = "openai/gpt-oss-120b", k: int = TOP_K, rerank: bool = True):
    """Returns ask(question) -> (answer, retrieved_docs). Retrieval runs once; the same chunks feed the LLM and the sources."""
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it via $env:GROQ_API_KEY or a .env file."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    reranker = CrossEncoder(RERANK_MODEL) if rerank else None
    llm = ChatGroq(model=model, temperature=0)  # same as eval_ragas.py, so answers are repeatable

    def ask(question: str):
        docs = retrieve(vectorstore, question, k=k, reranker=reranker)
        prompt = PROMPT_TEMPLATE.format(context=format_docs(docs), question=question)
        return llm.invoke(prompt).content, docs

    return ask


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("question", nargs="?", help="Question to ask. If omitted, starts an interactive loop.")
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Groq model name")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of chunks passed to the LLM")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder reranking step")
    args = parser.parse_args()

    ask_rag = build_rag(args.persist_dir, args.model, args.k, rerank=not args.no_rerank)

    def ask(question: str):
        answer, sources = ask_rag(question)
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        for d in sources:
            print(f"- {d.metadata.get('source', 'unknown')} (page {d.metadata.get('page', '?')})")

    if args.question:
        ask(args.question)
    else:
        print("RAG query loop. Type a question, or 'exit' to quit.")
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                ask(q)


if __name__ == "__main__":
    main()
