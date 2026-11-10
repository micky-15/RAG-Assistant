# Project Overview

A chatbot that answers questions from PDF reports and shows where each answer came from.

## 1. The big picture

```mermaid
flowchart LR
    A["📄 PDF reports"] --> B["ingest.py<br/>chunk + embed"]
    B --> C[("Chroma<br/>vector database")]

    U["👤 User<br/>public chat page"] --> D["Dify chatbot"]
    D -->|question| N["ngrok tunnel"]
    N --> E["app.py<br/>search + rerank"]
    C --> E
    E -->|5 best chunks| D
    D -->|question + chunks| L["🧠 Groq LLM<br/>gpt-oss-120b"]
    L -->|answer| D
    D -->|answer + sources| U

    classDef store fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef mine fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef ext fill:#f3e5f5,stroke:#6a1b9a,color:#000
    class A,C store
    class B,E mine
    class D,N,L ext
```

🟩 data &nbsp; 🟦 my code (runs on my laptop) &nbsp; 🟪 external services

## 2. Phase A: preparing the documents (once)

```mermaid
flowchart LR
    A["📄 PDF"] --> B["Split into chunks<br/>1500 chars, 300 overlap"]
    B --> C["Embedding model<br/>MiniLM: text → numbers"]
    C --> D[("Chroma database")]
```

## 3. Phase B: answering one question

```mermaid
sequenceDiagram
    actor User
    participant Dify
    participant App as app.py (laptop)
    participant LLM as Groq LLM

    User->>Dify: asks a question
    Dify->>App: search request (with secret key)
    Note over App: 1. top 20 chunks by meaning<br/>2. reranker re-sorts them<br/>3. keep the best 5
    App-->>Dify: 5 chunks, labelled with source file
    Dify->>LLM: instructions + 5 chunks + question
    LLM-->>Dify: answer using only those chunks
    Dify-->>User: answer + citations (file, page)
```

## 4. Demo



https://github.com/user-attachments/assets/f1189cd9-beb6-4b49-972e-f288ecf679e8



## 5. Evaluation


### Retrieval on 100 reference questions

How often the right chunk appears in the top results, and how high it ranks:

| Search method | Right chunk first (hit@1) | In top 3 | In top 5 |
|---|---|---|---|
| Vector search only | 0.57 | 0.72 | 0.79 |
| **Vector search + reranker (used in this project)** | **0.72** | **0.87** | **0.91** 


### Generation on 10 hand-written questions with reference answers

| RAGAS metric | Score |
|---|---|
| Context precision | 0.63 |
| Context recall | 0.75 |
| Faithfulness | 0.95 |
| Answer relevancy | 0.70 |

