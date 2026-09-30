"""
FastAPI backend for the local RAG chatbot.

Endpoints
---------
POST /session/new     -> create a new chat session (for memory tracking)
POST /chat             -> ask a question within a session, returns the answer
POST /upload            -> upload a document (.pdf/.txt/.csv/.docx), chunk + embed it
POST /transcribe       -> upload audio, get back transcribed text (local Whisper)
POST /tts                -> upload text, get back a synthesized .wav file

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
import uuid
from typing import Dict, List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import run_agent
from stt import transcribe_audio
from tts import synthesize_speech
from vector_search import ingest_file

app = FastAPI(title="Local RAG Chat API")

# Frontend (Streamlit) runs on a different port, so allow local cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store: session_id -> list of (question, answer) turns.
# Fine for a single-user local app; swap for Redis/DB if you need persistence
# across restarts or multiple concurrent users.
SESSIONS: Dict[str, List[Tuple[str, str]]] = {}
MAX_TURNS = 5


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/session/new")
def new_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = []
    return {"session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "question cannot be empty")

    history = SESSIONS.setdefault(req.session_id, [])
    answer = run_agent(req.question, chat_history=history)

    history.append((req.question, answer))
    SESSIONS[req.session_id] = history[-MAX_TURNS:]

    return ChatResponse(answer=answer, session_id=req.session_id)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt", ".csv", ".docx"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        n_chunks = ingest_file(dest_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to ingest file: {e}")

    return {"filename": file.filename, "chunks_added": n_chunks}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes, filename_hint=file.filename)
    return {"text": text}


@app.post("/tts")
def tts(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(400, "text cannot be empty")
    filepath = synthesize_speech(text)
    return FileResponse(filepath, media_type="audio/wav", filename=os.path.basename(filepath))


@app.get("/health")
def health():
    return {"status": "ok"}
