"""
Streamlit UI for the local RAG chatbot.

Talks only to the local FastAPI backend (http://localhost:8000) - no
external/cloud calls are made from this file.

Run with:
    streamlit run streamlit_app.py
"""

import queue
import threading
import time

import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🔒 Local RAG Chatbot")
st.caption("Fully local: Ollama LLM + HuggingFace embeddings + Chroma + local Whisper STT + local TTS.")

# ---------------- session bootstrap ----------------
if "session_id" not in st.session_state:
    try:
        resp = requests.post(f"{BACKEND_URL}/session/new", timeout=10)
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["session_id"]
    except requests.exceptions.RequestException:
        st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]

if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# Tracks an in-flight /chat call so we can show a Cancel button and abort it.
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # the question currently being answered
if "chat_result_queue" not in st.session_state:
    st.session_state.chat_result_queue = None
if "chat_http_session" not in st.session_state:
    st.session_state.chat_http_session = None  # requests.Session, closed on cancel

if st.session_state.session_id is None:
    st.error("Can't reach the backend at http://localhost:8000. Start it first (see README).")
    st.stop()


def start_chat_request(question: str):
    """Fire the /chat call on a background thread so the UI stays responsive
    and a Cancel button can abort it mid-flight."""
    result_q = queue.Queue()
    http_session = requests.Session()

    # IMPORTANT: st.session_state is tied to the main script thread and is
    # NOT safely readable from a background thread. Capture everything the
    # worker needs as plain local variables up front, and never touch
    # st.session_state inside worker().
    session_id = st.session_state.session_id

    def worker():
        try:
            r = http_session.post(
                f"{BACKEND_URL}/chat",
                json={"session_id": session_id, "question": question},
                timeout=300,
            )
            result_q.put(("done", r))
        except requests.exceptions.RequestException as e:
            # Also fires when the session is closed out from under the
            # request by cancel_chat_request() below.
            result_q.put(("cancelled_or_error", e))

    threading.Thread(target=worker, daemon=True).start()

    st.session_state.pending_question = question
    st.session_state.chat_result_queue = result_q
    st.session_state.chat_http_session = http_session


def cancel_chat_request():
    """Abort the in-flight request by closing its underlying connection."""
    if st.session_state.chat_http_session is not None:
        try:
            st.session_state.chat_http_session.close()
        except Exception:
            pass
    st.session_state.pending_question = None
    st.session_state.chat_result_queue = None
    st.session_state.chat_http_session = None

# ---------------- sidebar: document upload ----------------
with st.sidebar:
    st.header("📄 Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload documents to add to the vector store",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button(
        f"Ingest {len(uploaded_files)} document(s)", use_container_width=True
    ):
        progress = st.progress(0.0, text="Starting...")
        results = []  # (filename, ok, message)

        for i, uf in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Ingesting '{uf.name}'...")
            files = {"file": (uf.name, uf.getvalue())}
            try:
                r = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
            except requests.exceptions.RequestException as e:
                results.append((uf.name, False, str(e)))
                continue

            if r.status_code == 200:
                n_chunks = r.json()["chunks_added"]
                results.append((uf.name, True, f"{n_chunks} chunks added"))
            else:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                results.append((uf.name, False, detail))

        progress.progress(1.0, text="Done")

        for name, ok, message in results:
            if ok:
                st.success(f"✅ {name}: {message}")
            else:
                st.error(f"❌ {name}: {message}")

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        resp = requests.post(f"{BACKEND_URL}/session/new")
        st.session_state.session_id = resp.json()["session_id"]
        st.session_state.messages = []
        st.rerun()

# ---------------- render chat history ----------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            if st.button("🔊 Play answer", key=f"play_{i}"):
                with st.spinner("Synthesizing speech locally..."):
                    tr = requests.post(f"{BACKEND_URL}/tts", data={"text": msg["content"]})
                if tr.status_code == 200:
                    st.audio(tr.content, format="audio/wav")
                else:
                    st.error("TTS failed")

# ---------------- voice input ----------------
st.write("🎙️ Or record your question:")
audio_bytes = audio_recorder(text="", icon_size="2x", key="recorder")

if audio_bytes:
    audio_id = hash(audio_bytes)
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        with st.spinner("Transcribing locally with Whisper..."):
            files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
            r = requests.post(f"{BACKEND_URL}/transcribe", files=files)
        if r.status_code == 200:
            st.session_state.pending_input = r.json()["text"]
            st.info(f"Transcribed: {st.session_state.pending_input}")
        else:
            st.error("Transcription failed")

# ---------------- text input ----------------
# Disable the box while a request is in flight so a new question can't be
# submitted on top of one that's still running.
typed = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.pending_question is not None,
)
question = typed or (st.session_state.pending_input or None)

if question and st.session_state.pending_question is None:
    st.session_state.pending_input = ""
    st.session_state.messages.append({"role": "user", "content": question})
    start_chat_request(question)
    st.rerun()

# ---------------- in-flight request: show spinner + Cancel button ----------------
if st.session_state.pending_question is not None:
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.write("🤔 Thinking...")
        cancelled = st.button("✖️ Cancel", key="cancel_chat")

    if cancelled:
        cancel_chat_request()
        st.session_state.messages.append(
            {"role": "assistant", "content": "_Cancelled by user._"}
        )
        st.rerun()
    else:
        # Non-blocking poll: check if the background thread has a result yet.
        try:
            status, payload = st.session_state.chat_result_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.4)
            st.rerun()  # keep polling without freezing the UI
        else:
            if status == "done" and payload.status_code == 200:
                answer = payload.json()["answer"]
            elif status == "done":
                answer = f"Error: {payload.text}"
            else:
                # Request was aborted via cancel_chat_request(), or a real
                # connection error occurred.
                answer = None

            st.session_state.pending_question = None
            st.session_state.chat_result_queue = None
            st.session_state.chat_http_session = None

            if answer is not None:
                st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()