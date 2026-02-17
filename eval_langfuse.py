# eval_langfuse.py
import os
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("eval_langfuse")

# ----------------------------
# DeepEval (optional)
# ----------------------------
try:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False

# ----------------------------
# Langfuse (optional)
# ----------------------------
try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except Exception:
    LANGFUSE_AVAILABLE = False


def _env_true(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y"}


AUTO_EVAL_ENABLED = _env_true("AUTO_EVAL_ENABLED", "true")
LANGFUSE_ENABLED = _env_true("LANGFUSE_ENABLED", "true")

EVAL_MAX_CTX_ITEMS = int(os.getenv("EVAL_MAX_CTX_ITEMS", "6"))
EVAL_MAX_CTX_CHARS = int(os.getenv("EVAL_MAX_CTX_CHARS", "1500"))


# ----------------------------
# DeepEval: metric runner
# ----------------------------
def _measure(metric_obj, tc: "LLMTestCase") -> Tuple[float, str]:
    metric_obj.measure(tc)
    score = getattr(metric_obj, "score", None)
    reason = getattr(metric_obj, "reason", "") or ""
    return float(score) if score is not None else 0.0, reason


def run_deepeval_turn(
    *,
    question: str,
    answer: str,
    retrieval_context: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not (AUTO_EVAL_ENABLED and DEEPEVAL_AVAILABLE):
        return {"enabled": False, "reason": "DeepEval disabled or not installed."}

    q = (question or "").strip()
    a = (answer or "").strip()
    ctx = retrieval_context or []

    if not q or not a:
        return {"enabled": False, "reason": "Empty question/answer; skipping."}

    # trim context to keep eval latency reasonable
    ctx_trimmed = []
    for x in ctx[:EVAL_MAX_CTX_ITEMS]:
        x = (x or "")[:EVAL_MAX_CTX_CHARS]
        if x.strip():
            ctx_trimmed.append(x)

    results: Dict[str, Any] = {"enabled": True, "metrics": {}}

    # 1) Retrieval relevance proxy (is answer relevant to question?)
    rel_metric = AnswerRelevancyMetric(threshold=0.5, include_reason=True)
    rel_tc = LLMTestCase(input=q, actual_output=a)
    s, r = _measure(rel_metric, rel_tc)
    results["metrics"]["answer_relevancy"] = {"score": s, "reason": r}

    # 2) Grounded correctness proxy (faithfulness to retrieved context)
    faith_metric = FaithfulnessMetric(threshold=0.5, include_reason=True)
    faith_tc = LLMTestCase(input=q, actual_output=a, retrieval_context=ctx_trimmed)
    s, r = _measure(faith_metric, faith_tc)
    results["metrics"]["faithfulness"] = {"score": s, "reason": r}

    # 3) Correctness (GEval) - judged against question + retrieved evidence
    # NOTE: This is not "gold-answer" correctness (no ground truth available).
    correctness_metric = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the answer is correct given the user's question "
            "and the provided retrieval context. Penalize hallucinations or unsupported claims."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.5,
    )
    corr_tc = LLMTestCase(input=q, actual_output=a, retrieval_context=ctx_trimmed)
    s, r = _measure(correctness_metric, corr_tc)
    results["metrics"]["correctness_geval"] = {"score": s, "reason": r}

    # 4) Completeness (GEval)
    completeness_metric = GEval(
        name="Completeness",
        criteria="Evaluate whether the answer sufficiently addresses the user's question without missing key details.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )
    comp_tc = LLMTestCase(input=q, actual_output=a)
    s, r = _measure(completeness_metric, comp_tc)
    results["metrics"]["completeness_geval"] = {"score": s, "reason": r}

    return results


# ----------------------------
# Langfuse: tracing + scoring
# ----------------------------
_langfuse_client = None
_langfuse_handler = None


def get_langfuse_client():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not LANGFUSE_AVAILABLE:
        return None
    try:
        _langfuse_client = get_client()
        return _langfuse_client
    except Exception:
        logger.exception("Langfuse client init failed")
        return None


def get_langfuse_handler():
    global _langfuse_handler
    if _langfuse_handler is not None:
        return _langfuse_handler
    if not LANGFUSE_AVAILABLE:
        return None
    try:
        _langfuse_handler = CallbackHandler()
        return _langfuse_handler
    except Exception:
        logger.exception("Langfuse CallbackHandler init failed")
        return None


def start_generation_observation(
    *,
    name: str,
    trace_id: str,
    input_payload: Any,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Creates a Langfuse generation observation context manager.
    """
    if not (LANGFUSE_ENABLED and LANGFUSE_AVAILABLE):
        return None

    lf = get_langfuse_client()
    if lf is None:
        return None

    try:
        # trace_id is important to group everything under one trace
        return lf.start_as_current_observation(
            as_type="generation",
            name=name,
            input=input_payload,
            trace_id=trace_id,
            metadata=metadata or {},
        )
    except TypeError:
        # SDK compatibility fallback (older clients may not accept trace_id/metadata)
        try:
            return lf.start_as_current_observation(
                as_type="generation",
                name=name,
                input=input_payload,
            )
        except Exception:
            logger.exception("Langfuse observation start failed")
            return None
    except Exception:
        logger.exception("Langfuse observation start failed")
        return None


def push_scores_to_langfuse(obs, scores: Dict[str, Any]) -> None:
    """
    scores format:
      {"metrics": {"metric_name": {"score": float, "reason": str}, ...}}
    """
    if obs is None:
        return

    metrics = (scores or {}).get("metrics", {}) or {}
    for metric_name, payload in metrics.items():
        val = float(payload.get("score", 0.0) or 0.0)
        reason = payload.get("reason", "") or ""

        try:
            obs.score(name=metric_name, value=val, data_type="NUMERIC", comment=reason)
        except Exception:
            # some SDK versions differ; ignore safely
            pass

        try:
            obs.score_trace(name=metric_name, value=val, data_type="NUMERIC", comment=reason)
        except Exception:
            pass


def flush_langfuse() -> None:
    lf = get_langfuse_client()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception:
        pass


def evaluate_and_log(
    *,
    question: str,
    answer: str,
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    thread_id: str,
    turn_id: str,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs DeepEval + logs to Langfuse as one generation observation.
    Returns eval results (also useful for CLI logs).
    """
    trace_id = trace_id or str(uuid.uuid4())

    input_payload = {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "question": question,
        "retrieval_cfg": cfg,
        "sources": meta.get("sources", []),
        "pages": meta.get("pages", []),
        "chunk_ids": meta.get("chunk_ids", []),
    }

    obs_ctx = start_generation_observation(
        name="RAG Turn Eval",
        trace_id=trace_id,
        input_payload=input_payload,
        metadata={
            "mode": cfg.get("mode", "rag"),
            "retrieval_result": cfg.get("retrieval_result", "unknown"),
        },
    )

    eval_result: Dict[str, Any] = {}
    if obs_ctx is None:
        # run eval anyway (no Langfuse)
        eval_result = run_deepeval_turn(
            question=question,
            answer=answer,
            retrieval_context=meta.get("retrieval_context", []),
        )
        return {"trace_id": trace_id, "eval": eval_result}

    with obs_ctx as obs:
        # Ensure Langfuse shows clean output in UI
        try:
            obs.update(output=answer)
        except Exception:
            pass

        eval_result = run_deepeval_turn(
            question=question,
            answer=answer,
            retrieval_context=meta.get("retrieval_context", []),
        )

        push_scores_to_langfuse(obs, eval_result)

    flush_langfuse()
    return {"trace_id": trace_id, "eval": eval_result}