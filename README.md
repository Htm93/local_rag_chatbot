# Local RAG Chatbot (Web App)

Fully local RAG chatbot: FastAPI backend + Streamlit frontend, Ollama for
generation, HuggingFace sentence-transformers for embeddings, Chroma as the
vector store, local Whisper for speech-to-text, and pyttsx3 for
text-to-speech. No cloud APIs are called anywhere in this app.

```
local_rag_app/
├── backend/
│   ├── main.py            # FastAPI app (/chat, /upload, /transcribe, /tts)
│   ├── agent.py            # LangGraph pipeline + 5-turn conversation memory
│   ├── vector_search.py    # Chroma vector store + document ingestion
│   ├── stt.py                # Local Whisper transcription
│   └── tts.py                # Local pyttsx3 speech synthesis
├── frontend/
│   └── streamlit_app.py    # Chat UI, file upload, mic button, play button
└── requirements.txt
```

## 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally: https://ollama.com
- **ffmpeg** on your PATH (required by Whisper for audio decoding)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `winget install ffmpeg` (or download and add to PATH)
- On Linux, pyttsx3 needs `espeak`: `sudo apt install espeak`

## 2. Pull the LLM used by the agent

```bash
ollama pull llama3:8b
ollama serve   # if it isn't already running as a background service
```

(To use a different model, change `MODEL_NAME` in `backend/agent.py`.)

## 3. Install Python dependencies

It's strongly recommended to use a virtual environment:

```bash
cd local_rag_app
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will download the HuggingFace embedding model
(`all-MiniLM-L6-v2`) and the Whisper model (`base`) — both are cached
locally afterward, so this only happens once.

## 4. Add your seed data (optional)

If you want to keep using the original CSV-based seed data, place a
`data.csv` file (with `Title`, `data`, `Date` columns) inside `backend/`
before first run. If it's absent, the app just starts with an empty vector
store — you can then use the "Upload document" button to populate it.

## 5. Run the backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

This exposes:
- `POST /chat` — ask a question
- `POST /upload` — add a document (.pdf / .txt / .csv / .docx) to the vector store
- `POST /transcribe` — audio → text (local Whisper)
- `POST /tts` — text → speech (local pyttsx3, returns a .wav)
- `POST /session/new` — start a new conversation (resets memory)

Test it's alive: open http://localhost:8000/docs

## 6. Run the frontend (Streamlit)

In a **second terminal**, with the same venv active:

```bash
cd frontend
streamlit run streamlit_app.py
```

This opens the UI at **http://localhost:8501**.

## 7. Using the app

- Type or speak (🎙️ record button) your question — recorded audio is
  transcribed locally and dropped into the chat.
- Answers are grounded strictly in your ingested documents (the prompt
  forces the model to say "I don't know" rather than hallucinate).
- Click **🔊 Play answer** under any AI response to hear it read aloud.
- Use the sidebar to upload new PDFs/TXT/CSV/DOCX files — they're chunked,
  embedded, and added to the same Chroma collection the chatbot searches.
- **New conversation** resets the 5-turn memory window for a fresh session.

## Notes & possible upgrades

- **Memory** is in-process (a Python dict keyed by session id) — fine for a
  single local user, but it resets if you restart the backend. Swap in
  SQLite/Redis if you need it to survive restarts.
- **TTS quality**: `pyttsx3` uses your OS's built-in voices (robotic but
  100% offline, no downloads). For much more natural-sounding local speech,
  swap `backend/tts.py` for [Coqui TTS](https://github.com/coqui-ai/TTS)
  (`pip install TTS`) — the `synthesize_speech(text) -> filepath` function
  signature can stay identical.
- **STT accuracy**: the `base` Whisper model is fast but modest. Bump
  `WHISPER_MODEL_SIZE` in `backend/stt.py` to `"small"` or `"medium"` for
  better accuracy at the cost of speed/RAM.
- Everything (LLM calls, embeddings, STT, TTS) happens on your machine —
  the only "network" traffic is `localhost` calls between the Streamlit UI
  and the FastAPI backend.
