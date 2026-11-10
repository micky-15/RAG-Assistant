"""End-to-end RAG evaluation with RAGAS (LLM-judged), using Groq and local embeddings.

For each question in eval_set_v2.json it retrieves chunks from a throwaway
index, generates an answer with the same prompt as query.py, then scores:
  context_precision  - are the relevant chunks ranked high among those retrieved?
  context_recall     - do the retrieved chunks cover the reference answer?
  faithfulness       - is the answer supported by the retrieved chunks?
  answer_relevancy   - does the answer address the question?

Examples:
    python eval_ragas.py --limit 3                      # smoke test
    python eval_ragas.py --out results/ragas_baseline.json
    python eval_ragas.py --embedding-model BAAI/bge-base-en-v1.5 --out results/ragas_bge.json
    python eval_ragas.py --compare results/ragas_baseline.json results/ragas_bge.json
"""

import argparse
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

from dotenv import load_dotenv

# ragas 0.4.3 imports a module that newer langchain-community removed; it is only
# used for isinstance checks on Vertex models, so a stub is enough.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

from eval_retrieval import build_index
from sentence_transformers import CrossEncoder

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from query import PROMPT_TEMPLATE, RERANK_MODEL, TOP_K, format_docs, retrieve

load_dotenv()

METRIC_COLUMNS = ["llm_context_precision_with_reference", "context_recall", "faithfulness", "answer_relevancy"]


def generate(args, cases, embeddings):
    """Retrieve + answer every question. Returns ragas samples and chunk counts."""
    tmp = tempfile.mkdtemp(prefix="rag_eval_")
    try:
        n_pages, n_chunks = build_index(args.docs_dir, embeddings, args.chunk_size, args.chunk_overlap, tmp)
        store = Chroma(persist_directory=tmp, embedding_function=embeddings)
        reranker = None if args.no_rerank else CrossEncoder(RERANK_MODEL)
        llm = ChatGroq(model=args.model, temperature=0, max_retries=6)

        samples = []
        for i, case in enumerate(cases, 1):
            docs = retrieve(store, case["question"], k=args.k, reranker=reranker)  # same code path as query.py
            prompt = PROMPT_TEMPLATE.format(context=format_docs(docs), question=case["question"])
            answer = llm.invoke(prompt).content
            print(f"[{i}/{len(cases)}] generated: {case['question'][:70]}")
            samples.append({
                "user_input": case["question"],
                "retrieved_contexts": [d.page_content for d in docs],
                "response": answer,
                "reference": case["reference"],
            })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return samples, n_pages, n_chunks


def score(samples, args, embeddings):
    """Judge samples with RAGAS. Returns rows: each sample plus one score per metric (None if the judge call failed)."""
    judge = LangchainLLMWrapper(
        ChatGroq(model=args.judge_model, temperature=0, max_retries=6, max_tokens=args.judge_max_tokens)
    )
    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(strictness=1),  # Groq rejects n>1 completions
    ]
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=judge,
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(max_workers=args.workers, timeout=240, max_retries=6),
        raise_exceptions=False,
    )
    df = result.to_pandas()
    present = [c for c in METRIC_COLUMNS if c in df.columns]
    return [
        {**s, **{c: (None if r[c] != r[c] else float(r[c])) for c in present}}
        for s, (_, r) in zip(samples, df.iterrows())
    ]


def summarize(rows):
    """Mean per metric over the questions that were scored (failed judge calls are skipped)."""
    summary = {}
    for c in METRIC_COLUMNS:
        vals = [r[c] for r in rows if r.get(c) is not None]
        if vals:
            summary[c] = sum(vals) / len(vals)
            summary[f"n_scored[{c}]"] = len(vals)
    summary["n_questions"] = len(rows)
    return summary


def rerun_missing(args):
    """Re-judge only the questions that have a missing score in a saved report; keep every score already obtained."""
    report = json.loads(Path(args.rerun_missing).read_text(encoding="utf-8"))
    rows = report["results"]
    todo = [i for i, r in enumerate(rows) if any(r.get(c) is None for c in METRIC_COLUMNS)]
    print(f"Re-judging {len(todo)} of {len(rows)} questions with missing scores: {[i + 1 for i in todo]}")
    if todo:
        samples = [{k: rows[i][k] for k in ("user_input", "retrieved_contexts", "response", "reference")} for i in todo]
        embeddings = HuggingFaceEmbeddings(model_name=report["config"]["embedding_model"])
        for i, new in zip(todo, score(samples, args, embeddings)):
            for c in METRIC_COLUMNS:
                if rows[i].get(c) is None and new.get(c) is not None:
                    rows[i][c] = new[c]
    report["summary"] = summarize(rows)
    return report


def run(args):
    cases = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))[: args.limit]
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    samples, n_pages, n_chunks = generate(args, cases, embeddings)

    rows = score(samples, args, embeddings)
    summary = summarize(rows)
    config = {
        "embedding_model": args.embedding_model, "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap, "k": args.k, "rerank": not args.no_rerank, "generator": args.model,
        "judge": args.judge_model, "pages": n_pages, "chunks": n_chunks,
    }
    return {"config": config, "summary": summary, "results": rows}


def compare(path_a, path_b):
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    print(f"A = {path_a}: {json.dumps(a['config'])}")
    print(f"B = {path_b}: {json.dumps(b['config'])}\n")
    print(f"{"metric":<38}{'A':>9}{'B':>9}{'B-A':>9}")
    for c in METRIC_COLUMNS:
        if c in a["summary"] and c in b["summary"]:
            va, vb = a["summary"][c], b["summary"][c]
            print(f"{c:<38}{va:>9.3f}{vb:>9.3f}{vb - va:>+9.3f}")


def main():
    p = argparse.ArgumentParser(description="RAGAS evaluation of the RAG pipeline.")
    p.add_argument("--eval-set", default="eval_set_v2.json")
    p.add_argument("--docs-dir", nargs="+", default=["docs"])
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    p.add_argument("--k", type=int, default=TOP_K, help="Chunks passed to the LLM")
    p.add_argument("--no-rerank", action="store_true", help="Plain vector search, no cross-encoder (the old pipeline)")
    p.add_argument("--model", default="openai/gpt-oss-120b", help="Groq model that answers questions")
    p.add_argument("--judge-model", default="openai/gpt-oss-120b", help="Groq model that scores answers")
    p.add_argument("--workers", type=int, default=2, help="Parallel judge calls (keep low for Groq rate limits)")
    p.add_argument("--judge-max-tokens", type=int, default=4096,
                   help="Output-token cap for the judge; reasoning models need room or the call is cut off")
    p.add_argument("--rerun-missing", metavar="REPORT.json",
                   help="Re-judge only questions with missing scores in this saved report, then merge")
    p.add_argument("--limit", type=int, default=None, help="Only use the first N questions")
    p.add_argument("--out", help="Write the full report to this JSON file")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="Compare two saved reports and exit")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.compare:
        compare(*args.compare)
        return

    report = rerun_missing(args) if args.rerun_missing else run(args)
    print("\nConfig:", json.dumps(report["config"]))
    print("\nSummary:")
    for key, val in report["summary"].items():
        print(f"  {key:<30} {val:.3f}" if isinstance(val, float) else f"  {key:<30} {val}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
