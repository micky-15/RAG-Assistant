"""Load PDFs from docs/, chunk them, embed, and persist to a Chroma vector store."""

import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "docs"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1500/300 chosen over 1000/200 by the retrieval sweep 
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def build_index(docs_dir: str = DOCS_DIR, persist_dir: str = PERSIST_DIR) -> None:
    docs_path = Path(docs_dir)
    docs_path.mkdir(exist_ok=True)

    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {docs_path.resolve()}. Add some and re-run.")

    loader = PyPDFDirectoryLoader(str(docs_path))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    persist_path = Path(persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        print(f"Cleared existing vector store at {persist_path.resolve()}")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Vector store persisted to {Path(persist_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into a Chroma vector store.")
    parser.add_argument("--docs-dir", default=DOCS_DIR)
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    args = parser.parse_args()
    build_index(args.docs_dir, args.persist_dir)
