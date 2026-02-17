# InsightRAG (UPS Case Study) – Document Q&A with RAG + FAISS + Streaming UI

A Retrieval-Augmented Generation (RAG) system for answering questions over PDF documents (e.g., the UPS 2024 GRI Report).  
Includes: ingestion → chunking → embeddings → FAISS vector DB → retrieval (Similarity/MMR) → rerank → grounded answer with citations → Streamlit UI with streaming + feedback loop.  
Author: Karan Kadam

### Goal
----
Provide fast, grounded Q&A over one or more PDF documents by:
- Ingesting PDFs -> chunking -> embedding -> storing vectors in FAISS
- Retrieving relevant chunks at query time (Similarity/MMR)
- Reranking for better relevance
- Generative answering with strict citations from retrieved context
- Streamlit UI with streaming responses + source transparency + feedback loop
- Evaluation + tracing (DeepEval + Langfuse)

### Architecture Diagram
```bash
--------------------------
                   +---------------------------+
                   |      PDFs in /data        |
                   |  (UPS GRI report + more)  |
                   +-------------+-------------+
                                 |
                                 | (offline)
                                 v
+---------------------+   +---------------------+   +---------------------+
|  PyPDFLoader        |-->| Chunking            |-->| Embeddings          |
|  Extract per page   |   | Recursive splitter  |   | (Open-source / OA)  |
+---------------------+   +---------------------+   +----------+----------+
                                                                  |
                                                                  v
                                                       +---------------------+
                                                       | FAISS Vector Index  |
                                                       | vectorstore/...     |
                                                       +----------+----------+
                                                                  |
                                                                  | (online)
                                                                  v
+---------------------+      +---------------------+      +---------------------+
| Streamlit UI        |----->| Retrieval           |----->| Rerank   |
| - threads           |      | Similarity/MMR      |      | LLM JSON indices    |
| - streaming         |      | top_k/fetch_k       |      | fallback keyword    |
| - sources/previews  |      +----------+----------+      +----------+----------+
| - feedback          |                 |                            |
+----------+----------+                 +------------+---------------+
           |                                        |
           v                                        v
+---------------------+                     +------------------------+
| Feedback Store      |                     | Prompted Answering     |
| feedback.jsonl      |                     | Grounded RAG prompt    |
| + thread overrides  |                     | citations from tags    |
+---------------------+                     +-----------+------------+
                                                        |
                                                        v
                                            +------------------------+
                                            | LLM (stream/non-stream)|
                                            | gpt-4o-mini (current)  |
                                            +-----------+------------+
                                                        |
                                                        v
                                            +------------------------+
                                            | Eval/Tracing  |
                                            | DeepEval + Langfuse    |
                                            +------------------------+
```

## Features
- PDF ingestion using `PyPDFLoader`
- Intelligent chunking (RecursiveCharacterTextSplitter with overlap)
- Vector DB: FAISS (persisted locally)
- Retrieval modes:
  - Similarity (top-k)
  - MMR (top-k + fetch-k + diversity lambda)
- reranking:
  - LLM-based rerank returning JSON indices (with safe fallback rerank)
- Grounded answering:
  - Answers only from retrieved context
  - Citations using `[Source: ... | Page ...]` tags
- Streamlit UI:
  - Multi-thread chat
  - Streaming responses
  - Shows sources/pages + chunk previews
- Feedback loop:
  - 👍/👎 rating + comment saved to JSONL
  - Downvote strengthens retrieval for the thread (fetch_k increase + rerank ON)
- Evaluation & tracing:
  - DeepEval (AnswerRelevancy, Faithfulness, GEval-based correctness/completeness)
  - Langfuse tracing + metric scoring

### Key Design Decisions
--------------------
- FAISS: simple local vector DB, fast, easy to persist, good for case study
- Chunking: recursive splitter with overlap balances recall + coherence
- MMR: reduces redundancy in retrieved chunks, improves answer diversity
- Rerank: improves relevance when retrieval returns mixed chunks (with safe fallback)
- Strict citation tags: ensures claims are grounded and verifiable
- Feedback loop: thread-level adaptation improves retrieval after poor answers
- Async eval: avoids blocking UI response streaming

### Limitations
-----------
- LLM rerank may be slower than cross-encoder rerank
- PDF extraction quality depends on document structure (tables/images may lose fidelity)
- No BM25 hybrid retrieval unless added (easy extension)
- Faithfulness metrics are proxies without gold answers

### Suggested Next Improvements
---------------------------
- Add true hybrid retrieval (BM25 + vectors)
- Add cross-encoder reranker (bge-reranker / ms-marco MiniLM)
- Add structured citation formatting (inline claim-level citations)
- Add caching for embeddings and retrieval results
- Add document upload in UI and re-index trigger

## Repo Structure
```bash
├── app.py
├── main.py
├── faiss_ingest.py
├── prompt_manager.py
├── eval_langfuse.py
├── data/
│ └── 2024-UPS-GRI-Report.pdf
├── prompts/
│ ├── rag/v1/prompt.yaml
│ └── rerank/v1/prompt.yaml
├── vectorstore/
│ └── faiss_index/ (generated)
├── feedback/
│ └── feedback.jsonl (generated)
└── README.md
```

---

## Setup

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
```
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### 2) Install dependencies
```bash
pip install -r requirements.txt
```
### 3) Add the PDF(s)
Place the UPS PDF here:

- data/Report.pdf

### Configure Environment Variables -

Create a .env file in project root.
```bash
OPENAI_API_KEY= "YOUR API KEY"
LANGFUSE_SECRET_KEY = "YOUR KEY"
LANGFUSE_PUBLIC_KEY = "YOUR KEY"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
LANGFUSE_ENABLED=true
```

### Run: Build the Vector Index (Ingestion)

Run ingestion to create the FAISS index:
```bash
python faiss_ingest.py
```

This will:
- Load PDF pages
- Chunk text
- Embed chunks
- Save FAISS index to vectorstore/faiss_index

### Run: Start the Streamlit App
```bash
streamlit run app.py
```
Open the browser URL shown in terminal and ask questions.

## How It Works (Design Summary)

### Ingestion

1. `PyPDFLoader` extracts per-page text into LangChain `Document` objects.
2. `RecursiveCharacterTextSplitter` splits text into overlapping chunks.
3. Each chunk is embedded and stored in FAISS along with metadata:

   * `source`: PDF filename
   * `page`: page number

### Retrieval

* `main.py` loads the FAISS index and creates a retriever per request:

  * Similarity search OR MMR for diversity
* Retrieves top chunks and  reranks them.

### Reranking 

* Uses a strict rerank prompt (`prompts/rerank/v1`) to return JSON indices.
* If invalid output occurs, falls back to keyword-based reranking.

### Answering

* Builds context with tags:

  * `[Source: <filename> | Page <page>] ...`
* Uses a grounded prompt (`prompts/rag/v1`) to:

  * Answer only from context
  * Return citations using existing tags

### UI + Feedback Loop

* Streamlit streams tokens in real time.
* Shows sources/pages/chunk previews for transparency.
* Stores feedback in `feedback/feedback.jsonl`.
* If downvoted, retrieval is strengthened for that thread:

  * higher `fetch_k`
  * rerank forced ON

### Configuration Knobs

In `main.py`:

* `TOP_K`: number of chunks used for answer context
* `FETCH_K`: candidate pool size for MMR
* `USE_MMR` + `MMR_LAMBDA`: diversity control
* `USE_RERANK` + `RERANK_TOP_K`: reranking controls

### Demo Script (Quick)

Try:

* "What are the key sustainability reporting entities included?"
* "Summarize the main environmental commitments and cite pages."
* "Which sections talk about emissions reduction targets?"

You should see:

* Answer with citations like `[Source: 2024-UPS-GRI-Report.pdf | Page X]`
* Retrieval info with chunk previews
* Feedback controls at the bottom

