"""
app.py

Purpose:
Streamlit UI for InsightRAG that supports multi-threaded chat, streaming answers,
retrieval transparency (sources/pages/chunk previews), and per-turn feedback
that can adapt retrieval settings for the current thread.

Workflow:
1) Maintain thread-based chat history in Streamlit session_state
2) Stream answers from the RAG backend (stream_answer)
3) Show optional retrieval metadata for the last turn
4) Collect thumbs up/down feedback and persist it (log_feedback)
5) On downvote, strengthen retrieval for the next questions in the same thread

Author:
Karan Kadam
"""

import re
import uuid
import streamlit as st

from main import (
    stream_answer,
    log_feedback,
    apply_downvote_adaptation,
    evaluate_and_log,   # NEW
)

#Helper function to detect if user input is likely just small talk (not a real question)
def is_smalltalk_ui(text: str) -> bool:
    """
    Detect whether a user message is likely small talk rather than a document question.

    This is a lightweight UI-side heuristic used to avoid unnecessary retrieval for
    greetings, acknowledgements, and very short non-question messages.

    Args:
        text: Raw user input from the chat box.

    Returns:
        True if the input looks like small talk, otherwise False.
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


# =========================== Utilities ===========================
def generate_thread_id() -> str:
    """
    Generate a unique thread identifier for each new conversation.

    Returns:
        A UUID string.
    """
    return str(uuid.uuid4())

def init_state():
    """
    Initialize Streamlit session state keys used by the chat UI.

    Keys maintained:
    - thread_id: active conversation id
    - threads: list of all conversation ids
    - messages_by_thread: thread_id -> list of message dicts
    - active_meta: last-turn retrieval metadata
    - active_cfg: last-turn retrieval configuration
    - retrieval_overrides: thread_id -> adaptive retrieval overrides
    - last_turn_id: last assistant turn id (used for feedback linkage)
    """
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()

    if "threads" not in st.session_state:
        st.session_state["threads"] = [st.session_state["thread_id"]]

    if "messages_by_thread" not in st.session_state:
        st.session_state["messages_by_thread"] = {st.session_state["thread_id"]: []}

    if "active_meta" not in st.session_state:
        st.session_state["active_meta"] = {}

    if "active_cfg" not in st.session_state:
        st.session_state["active_cfg"] = {}

    if "retrieval_overrides" not in st.session_state:
        st.session_state["retrieval_overrides"] = {}

    if "last_turn_id" not in st.session_state:
        st.session_state["last_turn_id"] = None

def add_thread(thread_id: str):
    """
    Ensure a thread exists in session state.

    Args:
        thread_id: Conversation id to register.
    """
    if thread_id not in st.session_state["threads"]:
        st.session_state["threads"].append(thread_id)
    st.session_state["messages_by_thread"].setdefault(thread_id, [])
    st.session_state["retrieval_overrides"].setdefault(thread_id, {})

def reset_chat():
    """
    Start a new chat by generating a new thread_id and resetting
    last-turn metadata in the UI.
    """
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)
    st.session_state["active_meta"] = {}
    st.session_state["active_cfg"] = {}
    st.session_state["last_turn_id"] = None
    st.rerun()

def switch_thread(thread_id: str):
    """
    Switch the active chat thread.

    Args:
        thread_id: Existing conversation id to activate.
    """
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["active_meta"] = {}
    st.session_state["active_cfg"] = {}
    st.session_state["last_turn_id"] = None
    st.rerun()

# ======================= Session Initialization ===================
init_state()
thread_id = st.session_state["thread_id"]
add_thread(thread_id)
thread_messages = st.session_state["messages_by_thread"][thread_id]

# ============================ Sidebar ============================
st.sidebar.title("RAG Chatbot")
st.sidebar.markdown(f"Thread ID: {thread_id}")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()

st.sidebar.subheader("Past conversations")
threads = list(reversed(st.session_state["threads"]))
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for t in threads:
        if st.sidebar.button(t, key=f"thread-{t}"):
            switch_thread(t)

st.sidebar.divider()

over = st.session_state["retrieval_overrides"].get(thread_id, {})
if over:
    st.sidebar.subheader("Adaptive retrieval (this thread)")
    st.sidebar.write(over)

# ============================ Main Layout ========================
st.title("InsightRAG - Document Intelligence Assistant")
st.caption("Ask questions over the already-indexed documents in vectorstore/faiss_index")

for m in thread_messages:
    with st.chat_message(m["role"]):
        st.text(m["content"])

user_input = st.chat_input("Ask a question")

if user_input:
    thread_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    overrides = st.session_state["retrieval_overrides"].get(thread_id, {})

    turn_id = str(uuid.uuid4())
    st.session_state["last_turn_id"] = turn_id

    with st.chat_message("assistant"):
        status = st.status("Retrieving relevant chunks…", expanded=True)

        token_gen, meta, cfg = stream_answer(
            user_input,
            overrides=overrides,
            thread_id=thread_id,
            turn_id=turn_id
        )

        status.update(label="Generating answer…", state="running", expanded=True)
        assistant_text = st.write_stream(token_gen)

        status.update(label="Done", state="complete", expanded=False)

    thread_messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "turn_id": turn_id,
            "question": user_input,
            "meta": meta,
            "cfg": cfg,
        }
    )
    st.session_state["active_meta"] = meta
    st.session_state["active_cfg"] = cfg

# Optional: retrieval info still ok to show (you can remove if you want)
meta = st.session_state.get("active_meta") or {}
if meta.get("pages"):
    st.divider()
    st.subheader("Retrieval info (last question)")
    st.write(
        f"Sources: {', '.join([str(s) for s in meta.get('sources', [])])}"
        if meta.get("sources")
        else "Sources: (missing in metadata)"
    )
    st.write(f"Pages used: {', '.join([str(p) for p in meta.get('pages', [])])}")
    st.write(f"Chunks used: {meta.get('chunks_used', 0)}")

    with st.expander("Chunk previews"):
        for p in meta.get("previews", []):
            st.write(f"Chunk ID: {p.get('chunk_id')}")
            st.write(f"Page: {p.get('page')} | Source: {p.get('source')}")
            st.write(p.get("preview", ""))
            st.write("---")

# ========================= Feedback UI ===========================
st.divider()

last_assistant = None
for m in reversed(thread_messages):
    if m.get("role") == "assistant" and m.get("turn_id"):
        last_assistant = m
        break

if not last_assistant:
    st.info("Ask a question to enable feedback.")
else:
    col1, col2 = st.columns(2)
    with col1:
        up = st.button("👍 Helpful", use_container_width=True)
    with col2:
        down = st.button("👎 Not helpful", use_container_width=True)

    comment = st.text_area(
        "Optional: what was wrong / what did you expect?",
        key=f"fb-comment-{last_assistant['turn_id']}",
    )

    if up or down:
        rating = 1 if up else -1
        try:
            log_feedback(
                thread_id=thread_id,
                turn_id=last_assistant["turn_id"],
                question=last_assistant.get("question", ""),
                answer=last_assistant.get("content", ""),
                rating=rating,
                comment=comment,
                meta=last_assistant.get("meta", {}),
                cfg=last_assistant.get("cfg", {}),
            )
            st.success("Feedback saved.")

            if rating == -1:
                current_over = st.session_state["retrieval_overrides"].get(thread_id, {})
                updated_over = apply_downvote_adaptation(current_over)
                st.session_state["retrieval_overrides"][thread_id] = updated_over
                st.warning(
                    "Got it — I will use stronger retrieval for the next questions in this thread (higher fetch_k + rerank ON)."
                )
        except Exception as e:
            st.error(f"Failed to save feedback: {e}")