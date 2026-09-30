"""
Local Speech-to-Text using open-source Whisper (openai-whisper).

Runs fully offline once the model weights are downloaded the first time
(they are cached locally afterwards - no per-request network calls).
"""

import os
import tempfile

import whisper

_model = None
WHISPER_MODEL_SIZE = "base"  # tiny / base / small / medium / large - bigger = more accurate, slower


def get_model():
    global _model
    if _model is None:
        print(f"[STT] Loading local Whisper model '{WHISPER_MODEL_SIZE}'...")
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _model


def transcribe_audio(file_bytes: bytes, filename_hint: str = "audio.wav") -> str:
    """Transcribe raw audio bytes to text using the local Whisper model."""
    model = get_model()
    suffix = os.path.splitext(filename_hint)[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        return result.get("text", "").strip()
    finally:
        os.remove(tmp_path)
