"""Evaluate retrieval quality of the RAG pipeline against eval_set_v2.json.

Builds a throwaway Chroma index (the real chroma_db/ is untouched) from the PDFs
in --docs-dir with the chosen embedding model and chunking, then measures how
often (and how high) a chunk containing the expected answer is retrieved.

Examples:
    python eval_retrieval.py --out results/baseline.json
    python eval_retrieval.py --embedding-model BAAI/bge-base-en-v1.5 --out results/bge.json
    python eval_retrieval.py --compare results/baseline.json results/bge.json
"""

import argparse
import json
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import EMBEDDING_MODEL

KS = (1, 3, 5, 10)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def is_relevant(doc, req: dict) -> bool:
    if Path(doc.metadata.get("source", "")).name != req["source"]:
        return False
    text = norm(doc.page_content)
    return any(norm(s) in text for s in req["any_of"])


def requirements(case: dict) -> list[dict]:
    """Evidence a question needs. Multi-section / cross-document questions list several;
    unanswerable ones list none ("requirements": []) and are excluded from the metrics."""
    if "requirements" in case:
        return case["requirements"]
    return [{"source": case["source"], "any_of": case["any_of"]}]


def case_rank(docs, reqs):
    """Smallest k at which every requirement has a relevant chunk in the top k, else None."""
    ranks = [next((i + 1 for i, d in enumerate(docs) if is_relevant(d, r)), None) for r in reqs]
    return None if None in ranks else max(ranks)


def build_index(docs_dirs, embeddings, chunk_size, chunk_overlap, persist_dir):
    documents = []
    for d in docs_dirs:
        documents += PyPDFDirectoryLoader(d).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return len(documents), len(chunks)


def run(args):
    cases = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    tmp = tempfile.mkdtemp(prefix="rag_eval_")
    try:
        t0 = time.perf_counter()
        n_pages, n_chunks = build_index(args.docs_dir, embeddings, args.chunk_size, args.chunk_overlap, tmp)
        index_s = time.perf_counter() - t0
        store = Chroma(persist_directory=tmp, embedding_function=embeddings)

        results, latencies = [], []
        for case in cases:
            t = time.perf_counter()
            docs = store.similarity_search(case["question"], k=max(KS))
            latencies.append(time.perf_counter() - t)
            reqs = requirements(case)
            if not reqs:
                results.append({"question": case["question"], "source": "unanswerable", "rank": None, "scored": False})
                continue
            sources = {r["source"] for r in reqs}
            results.append({"question": case["question"], "source": sources.pop() if len(sources) == 1 else "multiple",
                            "rank": case_rank(docs, reqs), "scored": True})
    finally:
        # Chroma keeps the sqlite file open on Windows; ignore if it can't be removed.
        shutil.rmtree(tmp, ignore_errors=True)

    scored = [r for r in results if r.get("scored", True)]
    summary = {
        "n_questions": len(scored),
        **{f"hit@{k}": sum(r["rank"] is not None and r["rank"] <= k for r in scored) / len(scored) for k in KS},
        "mrr@10": sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored),
        "avg_query_ms": statistics.mean(latencies) * 1000,
        "index_seconds": index_s,
    }
    for src in sorted({r["source"] for r in scored}):
        sub = [r for r in scored if r["source"] == src]
        summary[f"hit@{args.k}[{src}]"] = sum(r["rank"] is not None and r["rank"] <= args.k for r in sub) / len(sub)

    config = {
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "pages": n_pages,
        "chunks": n_chunks,
    }
    return {"config": config, "summary": summary, "results": results}


def print_report(report, k):
    print("\nConfig:", json.dumps(report["config"]))
    print(f"\n{'rank':>5}  question")
    for r in report["results"]:
        label = "n/a" if not r.get("scored", True) else (r["rank"] or "MISS")
        print(f"{label:>5}  [{r['source'][:12]}] {r['question']}")
    print("\nSummary:")
    for key, val in report["summary"].items():
        print(f"  {key:<34} {val:.3f}" if isinstance(val, float) else f"  {key:<34} {val}")


def compare(path_a, path_b):
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    print(f"A = {path_a}: {json.dumps(a['config'])}")
    print(f"B = {path_b}: {json.dumps(b['config'])}\n")
    print(f"{'metric':<34}{'A':>9}{'B':>9}{'B-A':>9}")
    for key in a["summary"]:
        if key in b["summary"] and isinstance(a["summary"][key], float):
            va, vb = a["summary"][key], b["summary"][key]
            print(f"{key:<34}{va:>9.3f}{vb:>9.3f}{vb - va:>+9.3f}")
    print("\nQuestions whose result changed:")
    for ra, rb in zip(a["results"], b["results"]):
        if ra["rank"] != rb["rank"]:
            print(f"  {ra['rank'] or 'MISS'} -> {rb['rank'] or 'MISS'}  {ra['question']}")


def main():
    p = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    p.add_argument("--eval-set", default="eval_set_v2.json")
    p.add_argument("--docs-dir", nargs="+", default=["docs", "sample_docs"])
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--k", type=int, default=4, help="k used for the per-document breakdown")
    p.add_argument("--out", help="Write the full report to this JSON file")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="Compare two saved reports and exit")
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    report = run(args)
    print_report(report, args.k)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
