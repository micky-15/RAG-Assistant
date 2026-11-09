# RAG Chatbot: LangChain + Chroma + Groq + Dify



Ask questions about PDF reports and get answers with sources. Search runs in my own
Python pipeline; [Dify](https://dify.ai) provides the chatbot and its public web page.

How it works, in pictures: **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**


## What's in the repo

| File | Purpose |
|---|---|
| `ingest.py` | Reads PDFs in `docs/`, splits them into chunks, embeds them, stores them in `chroma_db/`. |
| `query.py` | Asks a question in the terminal (search, rerank, answer with Groq). |
| `app.py` | Web service Dify calls to search the index (`POST /retrieval`). |
| `push_to_dify.py` | Optional: uploads PDFs into Dify's own knowledge base (not used by the final chatbot). |

## 1. Setup

```
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (never commit `.env`) and fill in:

| Key | What it is for |
|---|---|
| `GROQ_API_KEY` | Free key from [console.groq.com](https://console.groq.com). Lets `query.py` call the LLM. |
| `RETRIEVAL_API_KEY` | A password for `app.py`, any long random string you choose. Dify sends it with every search; requests without it are refused. |

Put your PDFs in `docs/`.

## 2. Build the index

```
python ingest.py
```

Run it again whenever the PDFs change. It rebuilds `chroma_db/` from scratch.

## 3. Ask questions in the terminal

```
python query.py "What do the Bland-Altman plots show for the hip and knee versus the ankle?"
```

Run `python query.py` with no question for an interactive loop.

## 4. Run the chatbot through Dify

You need a free [Dify](https://cloud.dify.ai) account and [ngrok](https://ngrok.com) (free account).

**Keep two terminals open while the chatbot is in use.**

Terminal 1, the search service:

```
python -m uvicorn app:app --port 8000
```

Terminal 2, the public tunnel:

```
ngrok http 8000
```

Copy the `https://...ngrok-free.dev` address that ngrok prints.

Then, once, in Dify:

1. **Settings → Model Provider → GroqCloud**: add your Groq key.
2. **Knowledge → External Knowledge API → Add**
   - API Endpoint: the ngrok address
   - API Key: your `RETRIEVAL_API_KEY`
3. **Knowledge → Create → Connect to an External Knowledge Base**
   - Pick the API above, any Knowledge ID, **Top K = 5**, score threshold off.
4. **Studio → Create from Blank → More basic app types → Chatbot**
   - Add the external knowledge base. In its Retrieval Setting use **Top K = 5** and leave **Rerank Model off** (the pipeline already reranks).
   - Model: **GroqCloud → openai/gpt-oss-120b**, with **Temperature switched on and set to 0**.
   - Instructions:
     ```
     Answer the question using only the context below.
     If the answer isn't in the context, say I dont know.

     Context:
     {{#context#}}
     ```
   - Features: switch on **Citations and Attributions**.
5. **Publish → Run App** gives the public chat page.

> The free ngrok address can change when ngrok restarts. If it does, update the API
> Endpoint in Dify. If either terminal is closed, the chatbot cannot reach your documents.

## Troubleshooting

| Problem | Fix |
|---|---|
| `query.py` "is not recognized" | Start with `python`: `python query.py "..."`. |
| `No PDF files found` | Put PDFs in `docs/` and run `python ingest.py`. |
| Dify answers without your documents | Check both terminals are running and the ngrok address in Dify is current. |
| `403 Authorization failed` | The key in Dify does not match `RETRIEVAL_API_KEY` in `.env`. |
| Changed the PDFs but answers are old | Run `python ingest.py`, then restart the uvicorn terminal. |
| Broad questions ("what are the documents about?") cover only one PDF | Search returns the 5 best chunks overall, not 5 per document. Ask a specific question. |

## Notes

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, run locally.
- Chunking: 1500 characters with 300 overlap.
- Search: top 20 by vector similarity, reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 sent to the LLM.
- LLM: Groq `openai/gpt-oss-120b`, temperature 0.
