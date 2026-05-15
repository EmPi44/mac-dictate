#!/usr/bin/env python3
"""
Push-to-talk Diktat für macOS (Apple Silicon optimiert).

Halte eine Taste gedrückt → sprich → lasse los → Text erscheint im aktiven Textfeld.

Setup:
    pip install pynput sounddevice mlx-whisper numpy scipy
    brew install portaudio

Berechtigungen (einmalig, macOS fragt automatisch):
    - Systemeinstellungen → Datenschutz → Eingabehilfen → Terminal erlauben
    - Systemeinstellungen → Datenschutz → Mikrofon → Terminal erlauben

Starten:
    python3 dictate.py

Konfiguration: siehe CONFIG-Block unten
"""

import threading
import tempfile
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import mlx_whisper
from pynput import keyboard
from pynput.keyboard import Controller

# ─────────────────────────────────────────────
# KONFIGURATION - hier anpassen
# ─────────────────────────────────────────────

# Taste die gehalten werden muss zum Aufnehmen.
# Optionen: keyboard.Key.ctrl_r, keyboard.Key.alt_r, keyboard.Key.cmd_r
# Oder ein normaler Buchstabe: keyboard.KeyCode.from_char('f19')
HOTKEY = keyboard.Key.ctrl_r          # Standard: rechte Ctrl-Taste

# Sprache: None = automatisch erkennen, "de" = Deutsch, "en" = Englisch
LANGUAGE = None

# Modell: large-v3-turbo = beste Balance aus Speed und Qualität auf M-Chips
MODEL = "mlx-community/whisper-large-v3-turbo"

# Leerzeichen nach dem transkribierten Text einfügen?
ADD_TRAILING_SPACE = True

# Minimale Aufnahmedauer in Sekunden (kürzere Aufnahmen werden ignoriert)
MIN_DURATION_SECONDS = 0.5

# ─────────────────────────────────────────────

SAMPLE_RATE = 16000  # Whisper erwartet 16kHz
recording = False
audio_chunks = []
_lock = threading.Lock()
_typer = Controller()


def _audio_callback(indata, frames, time_info, status):
    """Läuft kontinuierlich im Hintergrund - nimmt nur auf wenn recording=True."""
    with _lock:
        if recording:
            audio_chunks.append(indata.copy())


def _transcribe_and_type():
    """Wird in einem eigenen Thread ausgeführt nach dem Loslassen der Taste."""
    with _lock:
        chunks = list(audio_chunks)

    if not chunks:
        return

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE

    if duration < MIN_DURATION_SECONDS:
        print(f"  (zu kurz: {duration:.1f}s - ignoriert)", flush=True)
        return

    print(f"  Aufnahme: {duration:.1f}s - transkribiere...", flush=True)

    # Temp-Datei für Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        # Auf int16 konvertieren für WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(tmp_path, SAMPLE_RATE, audio_int16)

    try:
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=MODEL,
            language=LANGUAGE,
            verbose=False,
            temperature=0.0,       # Deterministisch = konsistenter
            no_speech_threshold=0.4,
            condition_on_previous_text=False,
        )
        text = result.get("text", "").strip()

        if text:
            if ADD_TRAILING_SPACE:
                text += " "
            print(f"  → {text}", flush=True)
            _typer.type(text)
        else:
            print("  (kein Text erkannt)", flush=True)

    finally:
        os.unlink(tmp_path)


def on_press(key):
    global recording, audio_chunks
    if key == HOTKEY:
        with _lock:
            if not recording:
                recording = True
                audio_chunks = []
        print("🎙 Aufnahme...", flush=True)


def on_release(key):
    global recording
    if key == HOTKEY:
        with _lock:
            if not recording:
                return
            recording = False
        print("⏹ Stopp.", flush=True)
        threading.Thread(target=_transcribe_and_type, daemon=True).start()


def main():
    print("=" * 50)
    print("🎤 Diktat bereit")
    print(f"   Taste:   {HOTKEY}")
    print(f"   Modell:  {MODEL}")
    print(f"   Sprache: {LANGUAGE or 'auto'}")
    print("   Halte die Taste → sprich → loslassen")
    print("   Ctrl+C zum Beenden")
    print("=" * 50)

    # Audio-Stream läuft immer im Hintergrund (kaum CPU/RAM)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
        blocksize=1024,
    )

    with stream:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nBeendet.")


if __name__ == "__main__":
    main()
