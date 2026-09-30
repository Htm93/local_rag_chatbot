"""
Local Text-to-Speech using pyttsx3 (wraps SAPI5 / NSSpeechSynthesizer / espeak
depending on OS - all fully offline, no cloud calls).

If you want higher-quality, more natural voices and don't mind a heavier
dependency, swap this module's internals for Coqui TTS (`pip install TTS`)
- the synthesize_speech() signature can stay identical so nothing else in
the app needs to change.
"""

import os
import uuid

import pyttsx3

OUTPUT_DIR = "tts_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def synthesize_speech(text: str) -> str:
    """Synthesize text to a local .wav file and return its filepath."""
    engine = pyttsx3.init()
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    engine.save_to_file(text, filepath)
    engine.runAndWait()
    engine.stop()

    return filepath
