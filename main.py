"""
main.py

Purpose:
Core RAG runtime that retrieves relevant chunks from a FAISS vectorstore, optionally
re-ranks them using an LLM, builds an evidence-backed context, and generates an
answer using a chat model. It also supports streaming responses, feedback logging,
and optional DeepEval + Langfuse scoring/tracing.

High-level flow:
1) Load prompts from YAML (system + user templates)
2) Load FAISS index and initialize LLM client
3) Retrieve top chunks (MMR optional) and optionally rerank
4) Build context and call LLM to answer
5) Log metrics/traces (sync for non-stream, async for stream)
6) Capture user feedback and adapt retrieval settings on downvote

Author:
Karan Kadam
"""


import os
import json
import re
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Tuple, Iterator, Any, Optional, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from prompt_manager import get_prompts

from eval_langfuse import (
    evaluate_and_log,
    get_langfuse_handler,
)

import threading

# ----------------------------
# DeepEval (for auto-evaluation of answers)
# ----------------------------
try:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_query")

# ----------------------------
# Environment & Config
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment (.env)")

VECTORSTORE_PATH = "vectorstore/faiss_index"
PROMPT_YAML_PATH = "prompts/rag/v1/prompt.yaml"
RERANK_PROMPT_YAML_PATH = "prompts/rerank/v1/prompt.yaml"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 10
FETCH_K = 40
USE_MMR = True
MMR_LAMBDA = 0.7

DEBUG_RETRIEVAL = True
DEBUG_CHUNK_PREVIEW_CHARS = 350

# Re-rank Configs
USE_RERANK = True
RERANK_TOP_K = TOP_K

# Feedback adaptation caps/defaults
FETCH_K_MAX = 100
FETCH_K_STEP_ON_DOWNVOTE = 20

# Auto-eval toggles
AUTO_EVAL_ENABLED = os.getenv("AUTO_EVAL_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
EVAL_MAX_CHARS_PER_CHUNK = int(os.getenv("EVAL_MAX_CHARS_PER_CHUNK", "1500"))

# ----------------------------
# Prompt Manager (YAML via prompt_manager.py)
# ----------------------------
SYSTEM_PROMPT, USER_TEMPLATE, PROMPT_SPEC = get_prompts(PROMPT_YAML_PATH)
RERANK_SYSTEM_PROMPT, RERANK_USER_TEMPLATE, RERANK_PROMPT_SPEC = get_prompts(
    RERANK_PROMPT_YAML_PATH
)

def _cli_log_eval(turn_id: str, trace_id: str, eval_payload: Dict[str, Any]) -> None:
    """
    Log evaluation results in a compact, CLI-friendly format.

    Args:
        turn_id: Unique identifier for this conversation turn.
        trace_id: Langfuse trace id (if available).
        eval_payload: Output dict returned by evaluate_and_log().
    """
    
    ev = (eval_payload or {}).get("eval", {}) or {}
    if not ev.get("enabled"):
        logger.info(f"[EVAL] turn_id={turn_id} trace_id={trace_id} disabled: {ev.get('reason')}")
        return

    metrics = ev.get("metrics", {}) or {}
    msg = " | ".join(
        [f"{k}={float(v.get('score', 0.0)):.3f}" for k, v in metrics.items()]
    )
    logger.info(f"[EVAL] turn_id={turn_id} trace_id={trace_id} {msg}")

# ----------------------------
# Initialize Embeddings + Vector Store
# ----------------------------
def load_vectorstore(path: str) -> FAISS:
    """
    Load a persisted FAISS vectorstore from disk.

    Args:
        path: Folder path where the FAISS index was saved via save_local().

    Returns:
        A loaded FAISS vectorstore instance.

    Raises:
        FileNotFoundError: If the FAISS path does not exist.
        RuntimeError: If loading fails.
    """
    
    logger.info(f"Loading FAISS index from: {path}")

    if not os.path.exists(path):
        logger.error(f"FAISS index path not found: {path}")
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. Run faiss_ingest.py first."
        )

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vs = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception as e:
        logger.exception("Failed to load FAISS vectorstore")
        raise RuntimeError("Vectorstore load failed") from e


vectorstore = load_vectorstore(VECTORSTORE_PATH)

# ----------------------------
# LLM
# ----------------------------
def init_llm() -> ChatOpenAI:
    """
    Initialize the chat model used for both answering and optional reranking.

    Returns:
        ChatOpenAI client instance.

    Raises:
        RuntimeError: If initialization fails.
    """
    
    logger.info(f"Initializing LLM: {LLM_MODEL}")
    try:
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=400,
            frequency_penalty=0.3,
            presence_penalty=0.0,
            top_p=1.0,
            stop=None,
        )
    except Exception as e:
        logger.exception("Failed to initialize LLM client")
        raise RuntimeError("LLM init failed") from e


llm = init_llm()

# ----------------------------
# Feedback storage (JSONL)
# ----------------------------
def _utc_now_iso() -> str:
    """
    Get the current UTC time in ISO-8601 format.

    Returns:
        ISO timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def save_feedback_jsonl(record: Dict[str, Any], path: str = "feedback/feedback.jsonl") -> None:
    """
    Append a feedback record to a JSONL file on disk.

    Args:
        record: Feedback payload to store.
        path: Output JSONL file path.

    Raises:
        OSError: If writing fails.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ----------------------------
# Helpers
# ----------------------------
def _chunk_id(source: str, page: str, text: str) -> str:
    """
    Build a stable chunk identifier from source + page + text preview.

    Args:
        source: Document source name.
        page: Page number or identifier.
        text: Text content (preview used for hashing).

    Returns:
        Deterministic chunk id string.
    """
    base = f"{source}|{page}|{text[:300]}"
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{source}|p{page}|{h}"


def build_context(retrieved_docs) -> str:
    """
    Convert retrieved documents into a single context string for the LLM.

    Each chunk is prefixed with a tag including its source and page if available.

    Args:
        retrieved_docs: Iterable of LangChain Document objects.

    Returns:
        A context string formed by joining tagged chunks.
    """
    parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", doc.metadata.get("filename", ""))
        text = (doc.page_content or "").strip()
        if not text:
            continue

        if source:
            tag = f"[Source: {source} | Page {page}]"
        else:
            tag = f"[Page {page}]"

        parts.append(f"{tag} {text}")
    return "\n\n".join(parts)


def format_prompt(question: str, context: str) -> str:
    """
    Fill the user prompt template with the question and retrieved context.

    Args:
        question: User question.
        context: Retrieved evidence context.

    Returns:
        Formatted prompt string.
    """
    return USER_TEMPLATE.format(question=question, context=context)


def log_retrieval_preview(retrieved_docs, k: int) -> None:
    """
    Log a short snippet preview of retrieved chunks for debugging.

    Args:
        retrieved_docs: Retrieved documents list.
        k: Number of chunks to preview in logs.
    """
    for i, d in enumerate(retrieved_docs[:k], start=1):
        page = d.metadata.get("page", "N/A")
        snippet = (d.page_content or "")[:DEBUG_CHUNK_PREVIEW_CHARS].replace("\n", " ").strip()
        logger.info(f"#{i} | Page {page} | {snippet}")


def parse_json_int_list(text: str):
    """
    Extract a JSON list of integers from a model output.

    Supports:
    - Pure JSON: [1,2,3]
    - Fenced JSON: ```json [1,2,3] ```
    - Embedded list: ... [1,2,3] ...

    Args:
        text: Raw model output.

    Returns:
        Parsed list if found, otherwise None.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return None


def fallback_keyword_rerank(q: str, candidate_docs, k: int):
    """
    Keyword-based rerank fallback when the LLM reranker output is invalid.

    Scores chunks by the number of query terms present in the chunk text.

    Args:
        q: User query.
        candidate_docs: Candidate document chunks.
        k: Number of chunks to return.

    Returns:
        Top-k documents by keyword match score.
    """
    q_terms = set((q or "").lower().split())
    scored = []
    for d in candidate_docs:
        text = (d.page_content or "").lower()
        score = sum(1 for t in q_terms if t in text)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def rerank_with_llm(question: str, docs, top_k: int):
    """
    Rerank retrieved chunks using an LLM reranker prompt.

    Expects the reranker to return a JSON list of 1-based indices representing
    the preferred ranking order.

    Args:
        question: User question.
        docs: Candidate retrieved documents.
        top_k: Number of chunks to return after reranking.

    Returns:
        Reranked list of documents limited to top_k.
    """
    question = (question or "").strip()
    if not question:
        logger.warning("Rerank skipped: empty question")
        return docs[:top_k]

    if not docs:
        logger.warning("Rerank skipped: no candidate docs")
        return []

    items = []
    for idx, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "N/A")
        snippet = (d.page_content or "").replace("\n", " ").strip()[:600]
        items.append(f"{idx}. [Page {page}] {snippet}")

    chunks_text = "\n".join(items)

    rerank_prompt_text = RERANK_USER_TEMPLATE.format(
        question=question,
        top_k=top_k,
        chunks=chunks_text,
    )

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": rerank_prompt_text},
            ]
        ).content.strip()
    except Exception:
        logger.exception("Reranker LLM call failed; falling back to original retrieval order")
        return docs[:top_k]

    ranked_idxs = parse_json_int_list(resp)
    if ranked_idxs is None:
        logger.warning("Reranker returned non-JSON. Using keyword fallback rerank.")
        logger.warning(f"Raw reranker output (first 800 chars): {resp[:800]}")
        return fallback_keyword_rerank(question, docs, top_k)

    ranked_idxs = [i for i in ranked_idxs if isinstance(i, int) and 1 <= i <= len(docs)]
    if not ranked_idxs:
        logger.warning("Reranker returned empty/invalid indices. Falling back to original order.")
        return docs[:top_k]

    seen = set()
    final = []
    for i in ranked_idxs:
        if i in seen:
            continue
        final.append(docs[i - 1])
        seen.add(i)
        if len(final) >= top_k:
            break

    if len(final) < top_k:
        for d in docs:
            if d not in final:
                final.append(d)
            if len(final) >= top_k:
                break

    return final[:top_k]


def _docs_metadata(docs, top_k: int) -> Dict[str, Any]:
    """
    Build retrieval metadata for UI display and evaluation.

    Includes:
    - unique pages and sources
    - chunk ids and short previews
    - retrieval_context list used by faithfulness metrics

    Args:
        docs: Retrieved documents (already reranked if enabled).
        top_k: Number of chunks considered as "used".

    Returns:
        Metadata dict to accompany a generated answer.
    """
    pages: List[Any] = []
    sources: List[str] = []
    previews: List[Dict[str, Any]] = []
    chunk_ids: List[str] = []
    retrieval_context: List[str] = []

    for d in docs[:top_k]:
        page = d.metadata.get("page", "N/A")
        source = d.metadata.get("source", d.metadata.get("filename", ""))
        text = (d.page_content or "").strip()
        preview = text[:200].replace("\n", " ").strip()

        pages.append(page)
        if source:
            sources.append(source)

        cid = _chunk_id(source or "unknown", str(page), preview)
        chunk_ids.append(cid)

        previews.append(
            {
                "chunk_id": cid,
                "page": page,
                "source": source,
                "preview": preview,
            }
        )

        # faithfulness uses this retrieved evidence
        tag = f"[Source: {source} | Page {page}]" if source else f"[Page {page}]"
        if text:
            retrieval_context.append(f"{tag} {text[:EVAL_MAX_CHARS_PER_CHUNK]}")

    unique_pages = []
    for p in pages:
        if p not in unique_pages:
            unique_pages.append(p)

    unique_sources = []
    for s in sources:
        if s and s not in unique_sources:
            unique_sources.append(s)

    return {
        "pages": unique_pages,
        "sources": unique_sources,
        "chunks_used": min(len(docs), top_k),
        "chunk_ids": chunk_ids,
        "previews": previews,
        "retrieval_context": retrieval_context,  # <-- added
    }


def _effective_cfg(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge retrieval overrides with defaults and apply safety caps.

    Args:
        overrides: Optional per-thread overrides (e.g., from downvote adaptation).

    Returns:
        Effective retrieval configuration dict.
    """
    overrides = overrides or {}
    cfg = {
        "top_k": int(overrides.get("top_k", TOP_K)),
        "fetch_k": int(overrides.get("fetch_k", FETCH_K)),
        "use_mmr": bool(overrides.get("use_mmr", USE_MMR)),
        "mmr_lambda": float(overrides.get("mmr_lambda", MMR_LAMBDA)),
        "use_rerank": bool(overrides.get("use_rerank", USE_RERANK)),
        "rerank_top_k": int(overrides.get("rerank_top_k", RERANK_TOP_K)),
    }
    if cfg["fetch_k"] < cfg["top_k"]:
        cfg["fetch_k"] = cfg["top_k"]
    cfg["fetch_k"] = min(cfg["fetch_k"], FETCH_K_MAX)
    return cfg


def _make_retriever(cfg: Dict[str, Any]):
    """
    Construct a LangChain retriever from the FAISS vectorstore based on cfg.

    Args:
        cfg: Effective retrieval configuration.

    Returns:
        A retriever instance ready to invoke(question).
    """
    if cfg["use_mmr"]:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": cfg["top_k"],
                "fetch_k": cfg["fetch_k"],
                "lambda_mult": cfg["mmr_lambda"],
            },
        )
    return vectorstore.as_retriever(search_kwargs={"k": cfg["top_k"]})


# ----------------------------
# DeepEval: turn evaluation
# ----------------------------
def evaluate_turn(
    question: str,
    answer: str,
    retrieval_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
    
    """
    Run local DeepEval metrics for a single Q/A turn.

    Note:
    This function is kept for compatibility/testing. In the main pipeline,
    evaluate_and_log() is used to run eval and optionally push scores to Langfuse.

    Args:
        question: User question.
        answer: Model answer.
        retrieval_context: Evidence chunks used for faithfulness.

    Returns:
        DeepEval result dict including enabled flag and metric scores/reasons.
    """
    if not (AUTO_EVAL_ENABLED and DEEPEVAL_AVAILABLE):
        return {
            "enabled": False,
            "reason": "DeepEval not installed or AUTO_EVAL_ENABLED is false.",
        }

    q = (question or "").strip()
    a = (answer or "").strip()
    ctx = retrieval_context or []

    if not q or not a:
        return {
            "enabled": False,
            "reason": "Empty question or answer; skipping evaluation.",
        }

    def _measure(metric_obj, tc: LLMTestCase) -> Tuple[float, str]:
        metric_obj.measure(tc)
        score = getattr(metric_obj, "score", None)
        reason = getattr(metric_obj, "reason", "") or ""
        return float(score) if score is not None else 0.0, reason

    results: Dict[str, Any] = {"enabled": True, "metrics": {}}

    # Relevancy (proxy for "retrieval usefulness")
    rel_metric = AnswerRelevancyMetric(threshold=0.5, include_reason=True)
    rel_tc = LLMTestCase(input=q, actual_output=a)
    rel_score, rel_reason = _measure(rel_metric, rel_tc)
    results["metrics"]["answer_relevancy"] = {"score": rel_score, "reason": rel_reason}

    # Faithfulness (groundedness to retrieved chunks)
    faith_metric = FaithfulnessMetric(threshold=0.5, include_reason=True)
    faith_tc = LLMTestCase(input=q, actual_output=a, retrieval_context=ctx)
    faith_score, faith_reason = _measure(faith_metric, faith_tc)
    results["metrics"]["faithfulness"] = {"score": faith_score, "reason": faith_reason}

    # Completeness (GEval)
    comp_metric = GEval(
        name="Completeness",
        criteria="Evaluate whether the answer sufficiently addresses the user's question without missing key details.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )
    comp_tc = LLMTestCase(input=q, actual_output=a)
    comp_score, comp_reason = _measure(comp_metric, comp_tc)
    results["metrics"]["completeness"] = {"score": comp_score, "reason": comp_reason}

    return results


# ----------------------------
# Public API for Frontend
# ----------------------------
def retrieve_context(
    question: str,
    overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    
    """
    Retrieve relevant document chunks for a question and build context.

    Applies:
    - effective retrieval config (defaults + overrides)
    - MMR retrieval (optional)
    - LLM reranking (optional)
    - metadata packaging for UI + evaluation

    Args:
        question: User question to retrieve against.
        overrides: Optional per-thread retrieval overrides.

    Returns:
        Tuple of:
        - context string
        - metadata dict
        - effective config dict

    Raises:
        ValueError: If question is empty.
        RuntimeError: If retrieval fails.
    """
    
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty")

    cfg = _effective_cfg(overrides)
    retriever = _make_retriever(cfg)

    logger.info(f"Question received: {question}")
    logger.info(f"Effective retrieval cfg: {cfg}")

    try:
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            logger.warning("No documents retrieved (index may be empty or query mismatch)")
            meta = {
                "pages": [],
                "sources": [],
                "chunks_used": 0,
                "chunk_ids": [],
                "previews": [],
                "retrieval_context": [],
            }
            return "", meta, cfg

        final_docs = retrieved_docs
        if cfg["use_rerank"]:
            final_docs = rerank_with_llm(question, retrieved_docs, top_k=cfg["rerank_top_k"])

        if DEBUG_RETRIEVAL:
            log_retrieval_preview(final_docs, k=cfg["top_k"])

        context = build_context(final_docs)
        meta = _docs_metadata(final_docs, top_k=cfg["top_k"])
        return context, meta, cfg

    except Exception as e:
        logger.exception("Retrieval failed")
        raise RuntimeError("Retriever invoke failed") from e


def answer_question(
    question: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    thread_id: str = "default-thread",
    turn_id: str = "turn-unknown",
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Non-streaming RAG call: retrieve context, generate an answer, then evaluate/log.

    Args:
        question: User question.
        overrides: Optional retrieval overrides.
        thread_id: Conversation thread id for tracing/feedback.
        turn_id: Turn id for tracing/feedback.

    Returns:
        Tuple of (answer, meta, cfg).
    """
    context, meta, cfg = retrieve_context(question, overrides=overrides)
    if not context.strip():
        answer = "Not found in provided documents."
        return answer, meta, cfg

    prompt_text = format_prompt(question, context)

    try:
        # Optional: trace LLM calls via langfuse callback handler (if enabled/installed)
        handler = get_langfuse_handler()
        llm_kwargs = {}
        if handler is not None:
            llm_kwargs = {"config": {"callbacks": [handler]}}

        response = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            **llm_kwargs
        )
    except Exception as e:
        logger.exception("LLM invoke failed")
        raise RuntimeError("LLM call failed") from e

    answer = getattr(response, "content", None)
    if not answer:
        logger.error("LLM returned empty response content")
        raise RuntimeError("Empty LLM response")

    answer = answer.strip()

    # Auto-eval + Langfuse scoring (CLI-only output)
    try:
        eval_payload = evaluate_and_log(
            question=question,
            answer=answer,
            meta=meta,
            cfg=cfg,
            thread_id=thread_id,
            turn_id=turn_id,
        )
        _cli_log_eval(turn_id, eval_payload.get("trace_id", ""), eval_payload)
    except Exception:
        logger.exception("Evaluation/Langfuse logging failed (non-blocking)")

    return answer, meta, cfg


def stream_answer(
    question: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    thread_id: str = "default-thread",
    turn_id: str = "turn-unknown",
    ) -> Tuple[Iterator[str], Dict[str, Any], Dict[str, Any]]:
    """
    Streaming RAG call that yields tokens while generating the answer.

    Behavior:
    - Routes very short/greeting inputs to a smalltalk mode (no retrieval)
    - For RAG mode: retrieves context, streams LLM output
    - Runs eval + Langfuse logging asynchronously after streaming completes

    Args:
        question: User input.
        overrides: Optional retrieval overrides.
        thread_id: Conversation thread id for tracing.
        turn_id: Turn id for tracing.

    Returns:
        Tuple of (token generator, meta, cfg).
    """

    def _empty_meta() -> Dict[str, Any]:
        """
        Build an empty metadata object for cases where retrieval is skipped/empty.
        """
        return {
            "pages": [],
            "sources": [],
            "chunks_used": 0,
            "chunk_ids": [],
            "previews": [],
            "retrieval_context": [],
        }

    def _is_smalltalk(text: str) -> bool:
        """
        Heuristic detector for smalltalk messages (greetings/acknowledgements).

        Args:
            text: Raw user input.

        Returns:
            True if text looks like smalltalk, otherwise False.
        """
        t = (text or "").strip().lower()
        if not t:
            return True

        direct = {
            "hi", "hello", "hey", "hii", "hiii",
            "good morning", "good afternoon", "good evening",
            "how are you", "how r you", "whats up", "what's up", "wassup",
            "thanks", "thank you", "thx",
            "ok", "okay", "cool", "done", "k", "kk", "alright",
        }
        if t in direct:
            return True

        patterns = [
            r"^\s*(hi|hello|hey|hii|hiii)\s*[!.]?\s*$",
            r"^\s*(good\s+morning|good\s+afternoon|good\s+evening)\s*[!.]?\s*$",
            r"^\s*how\s+are\s+you\s*[?.!]?\s*$",
            r"^\s*(thanks|thank\s+you)\s*[!.]?\s*$",
            r"^\s*done\s*[!.]?\s*$",
        ]
        for p in patterns:
            if re.match(p, t):
                return True

        if len(t.split()) <= 3 and "?" not in t:
            return True

        return False

    def _smalltalk_gen() -> Iterator[str]:
        """
        Token generator for smalltalk mode (no retrieval).
        """
        try:
            handler = get_langfuse_handler()
            llm_kwargs = {}
            if handler is not None:
                llm_kwargs = {"config": {"callbacks": [handler]}}

            for chunk in llm.stream(
                [
                    {
                        "role": "system",
                        "content": "You are a friendly helpful assistant. Reply naturally and briefly.",
                    },
                    {"role": "user", "content": (question or "").strip()},
                ],
                **llm_kwargs
            ):
                content = getattr(chunk, "content", "")
                if content:
                    yield content
        except Exception:
            logger.exception("Smalltalk LLM stream failed")
            yield "Hello! How can I help you today?"

    def _run_eval_async(q: str, a: str, m: Dict[str, Any], c: Dict[str, Any]) -> None:
        """
        Run evaluation and Langfuse scoring in a background thread.
        """
        try:
            eval_payload = evaluate_and_log(
                question=q,
                answer=a,
                meta=m,
                cfg=c,
                thread_id=thread_id,
                turn_id=turn_id,
            )
            _cli_log_eval(turn_id, eval_payload.get("trace_id", ""), eval_payload)
        except Exception:
            logger.exception("Async Evaluation/Langfuse logging failed (non-blocking)")

    question_clean = (question or "").strip()
    if _is_smalltalk(question_clean):
        return _smalltalk_gen(), _empty_meta(), {"mode": "smalltalk"}

    context, meta, cfg = retrieve_context(question_clean, overrides=overrides)
    if not context.strip():

        def _empty() -> Iterator[str]:
            yield "Not found in provided documents."

        cfg = dict(cfg or {})
        cfg["mode"] = "rag"
        cfg["retrieval_result"] = "empty_context"
        return _empty(), meta, cfg

    prompt_text = format_prompt(question_clean, context)

    cfg = dict(cfg or {})
    cfg["mode"] = "rag"
    cfg["retrieval_result"] = "ok"

    def _gen_with_async_eval() -> Iterator[str]:
        """
        Token generator for RAG mode with asynchronous evaluation after completion.
        """
        full: List[str] = []
        try:
            handler = get_langfuse_handler()
            llm_kwargs = {}
            if handler is not None:
                llm_kwargs = {"config": {"callbacks": [handler]}}

            for chunk in llm.stream(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                **llm_kwargs
            ):
                content = getattr(chunk, "content", "")
                if content:
                    full.append(content)
                    yield content
        except Exception:
            logger.exception("LLM stream failed")
            yield "\n\n[Error] LLM streaming failed."
        finally:
            # IMPORTANT: do not block Streamlit's write_stream() with eval/langfuse
            answer_text = "".join(full).strip()
            if answer_text:
                t = threading.Thread(
                    target=_run_eval_async,
                    args=(question_clean, answer_text, meta, cfg),
                    daemon=True,
                )
                t.start()

    return _gen_with_async_eval(), meta, cfg

def apply_downvote_adaptation(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Increase retrieval strength after a downvote (thread-specific).

    Strategy:
    - Increase fetch_k by a fixed step up to FETCH_K_MAX
    - Force rerank on
    - Ensure rerank_top_k is set

    Args:
        overrides: Current per-thread overrides.

    Returns:
        Updated overrides dict.
    """
    current_fetch_k = int(overrides.get("fetch_k", FETCH_K))
    new_fetch_k = min(current_fetch_k + FETCH_K_STEP_ON_DOWNVOTE, FETCH_K_MAX)

    new_overrides = dict(overrides)
    new_overrides["fetch_k"] = new_fetch_k
    new_overrides["use_rerank"] = True
    if "rerank_top_k" not in new_overrides:
        new_overrides["rerank_top_k"] = TOP_K

    return new_overrides


def log_feedback(
    thread_id: str,
    turn_id: str,
    question: str,
    answer: str,
    rating: int,
    comment: str,
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    ) -> None:
    """
    Persist user feedback for a turn to JSONL for later analysis.

    Args:
        thread_id: Conversation thread id.
        turn_id: Turn id.
        question: User question.
        answer: Assistant answer.
        rating: 1 for helpful, -1 for not helpful.
        comment: Optional free-text feedback.
        meta: Retrieval metadata associated with the turn.
        cfg: Retrieval configuration used for the turn.
    """
    record = {
        "created_at": _utc_now_iso(),
        "thread_id": thread_id,
        "turn_id": turn_id,
        "question": question,
        "answer": answer,
        "rating": int(rating),
        "comment": (comment or "").strip(),
        "retrieval_cfg": cfg,
        "pages": meta.get("pages", []),
        "sources": meta.get("sources", []),
        "chunk_ids": meta.get("chunk_ids", []),
        "chunk_previews": meta.get("previews", []),
    }
    save_feedback_jsonl(record)


# if __name__ == "__main__":
#     logger.info("===== RAG QUERY STARTED =====")
#     q = "Entities included in the organization's sustainability reporting ?"
#     ans, meta, cfg = answer_question(q)
#     print("\n===== ANSWER =====")
#     print(ans)
#     print("\n===== META =====")
#     print(meta)
#     print("\n===== CFG =====")
#     print(cfg)
#     logger.info("===== RAG QUERY FINISHED =====")