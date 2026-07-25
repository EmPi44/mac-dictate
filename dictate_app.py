#!/usr/bin/env python3
"""
mac-dictate Menu Bar App.

Mikrofon-Icon in der Menuleiste - klicken zum An/Aus schalten.
Rechte Option-Taste halten → sprechen → loslassen → Text erscheint.
"""

import threading
import os
import queue
import numpy as np
import sounddevice as sd
import mlx_whisper
import rumps
from pynput import keyboard
from pynput.keyboard import Controller
from ApplicationServices import AXIsProcessTrustedWithOptions
from CoreFoundation import kCFBooleanTrue

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────
HOTKEY          = keyboard.Key.alt_r
LANGUAGE        = None           # None = auto (Deutsch + Englisch)
MODEL           = "mlx-community/whisper-small-mlx"
ADD_TRAILING_SPACE = True
MIN_DURATION_SECONDS = 0.5
SAMPLE_RATE     = 16000

ICON_IDLE       = "Dic"
ICON_RECORDING  = "Dic ●"
ICON_THINKING   = "Dic ..."
# ─────────────────────────────────────────────

recording = False
audio_chunks = []
_lock = threading.Lock()
_typer = Controller()
_listener = None
_transcription_queue = queue.Queue()


def load_model():
    """Load and warm the MLX model before the listener accepts recordings."""
    mlx_whisper.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        path_or_hf_repo=MODEL,
        language="de",
        temperature=0.0,
        no_speech_threshold=0.4,
        condition_on_previous_text=False,
        fp16=True,
        verbose=None,
    )


def audio_callback(indata, frames, time_info, status):
    with _lock:
        if recording:
            audio_chunks.append(indata.copy())


def _log_error(msg):
    """Fail loudly: surface transcription failures to the log instead of swallowing them."""
    import time
    try:
        with open(os.path.expanduser("~/Library/Logs/mac-dictate.log"), "a") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except Exception:
        pass


def transcribe_and_type(app, chunks):
    if not chunks:
        app.title = ICON_IDLE
        return

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE

    if duration < MIN_DURATION_SECONDS:
        app.title = ICON_IDLE
        return

    app.title = ICON_THINKING

    try:
        # Pass the float32 16kHz audio array straight to MLX Whisper. This avoids
        # the file-loading path and its ffmpeg dependency under launchd.
        # See CLAUDE.md "Gotcha: ffmpeg / launchd PATH".
        result = mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=MODEL,
            language=LANGUAGE,
            temperature=0.0,
            no_speech_threshold=0.4,
            condition_on_previous_text=False,
            fp16=True,
            verbose=None,
        )
        text = result.get("text", "").strip()
        if text:
            if ADD_TRAILING_SPACE:
                text += " "
            _typer.type(text)
    except Exception as e:
        _log_error("transcription failed: %r" % e)
    finally:
        app.title = ICON_IDLE


def transcription_worker(app):
    """Process dictations in capture order so results cannot overtake each other."""
    while True:
        chunks = _transcription_queue.get()
        try:
            transcribe_and_type(app, chunks)
        finally:
            _transcription_queue.task_done()


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__(ICON_IDLE, quit_button=None)
        self.menu = [
            rumps.MenuItem("Diktat: EIN", callback=self.toggle),
            None,
            rumps.MenuItem("Beenden", callback=self.quit_app),
        ]
        self.active = True
        self._stream = None
        self._listener = None

        # Modell im Hintergrund laden
        self.title = "Dic ..."
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        load_model()
        self.title = ICON_IDLE
        threading.Thread(
            target=transcription_worker, args=(self,), daemon=True
        ).start()
        self._start_listener()

    def toggle(self, sender):
        self.active = not self.active
        if self.active:
            sender.title = "Diktat: AUS"
            self._start_listener()
        else:
            sender.title = "Diktat: EIN"
            self._stop_listener()

    def _start_listener(self):
        global recording, audio_chunks

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=1024,
        )
        self._stream.start()

        app = self

        def on_press(key):
            global recording, audio_chunks
            if not app.active:
                return
            if key == HOTKEY:
                with _lock:
                    if not recording:
                        recording = True
                        audio_chunks = []
                app.title = ICON_RECORDING

        def on_release(key):
            global recording, audio_chunks
            if not app.active:
                return
            if key == HOTKEY:
                with _lock:
                    if not recording:
                        return
                    recording = False
                    chunks = audio_chunks
                    audio_chunks = []
                app.title = ICON_THINKING
                _transcription_queue.put(chunks)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def _stop_listener(self):
        if self._listener:
            self._listener.stop()
            self._listener = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.title = ICON_IDLE

    def quit_app(self, _):
        self._stop_listener()
        rumps.quit_application()


if __name__ == "__main__":
    # Trigger the macOS Accessibility permission dialog if not yet granted
    AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": kCFBooleanTrue})
    DictateApp().run()
