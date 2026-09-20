#!/usr/bin/env python3
"""
mac-dictate Menu Bar App.

Mikrofon-Icon in der Menuleiste - klicken zum An/Aus schalten.
Rechte Option-Taste drücken → sprechen → erneut drücken → Text erscheint.
"""

import threading
import os
import queue
import time
from dataclasses import dataclass, field
import numpy as np
import sounddevice as sd
import mlx_whisper
import rumps
from live_caption import LiveCaptionOverlay
from live_preview import LatestAudioWorker
from prompt_package import format_prompt
from review_panel import ReviewPanel
from screenshot_context import (
    capture_region, capture_screen, discard, display_under_pointer,
    front_window, prune_old_captures,
)
from pynput import keyboard
from pynput.keyboard import Controller
from ApplicationServices import AXIsProcessTrustedWithOptions
from CoreFoundation import kCFBooleanTrue
from AppKit import NSPasteboard, NSPasteboardTypeString

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────
HOTKEY          = keyboard.Key.alt_r
SCREENSHOT_HOTKEY = keyboard.Key.f13  # Map the upper mouse side button to F13.
LANGUAGE        = None           # None = auto (Deutsch + Englisch)
MODEL           = "mlx-community/whisper-small-mlx"
ADD_TRAILING_SPACE = True
MIN_DURATION_SECONDS = 0.5
SAMPLE_RATE     = 16000
PREVIEW_START_SECONDS = 2.0
PREVIEW_INTERVAL_SECONDS = 1.5
HALLUCINATION_NO_SPEECH_THRESHOLD = 0.8
HALLUCINATION_COMPRESSION_RATIO = 2.4

ICON_IDLE       = "Dic"
ICON_RECORDING  = "Dic ●"
ICON_THINKING   = "Dic ..."
# ─────────────────────────────────────────────

recording = False
audio_chunks = []
_lock = threading.Lock()
_inference_lock = threading.Lock()
_typer = Controller()
_listener = None
_transcription_queue = queue.Queue()


@dataclass
class CaptureSession:
    requested: bool = False
    mode: str | None = None
    context: dict = field(default_factory=dict)
    image_path: str | None = None
    error: str | None = None
    finished: threading.Event = field(default_factory=threading.Event)
    discarded: bool = False
    review_queued: bool = False
    text_ready: threading.Event = field(default_factory=threading.Event)
    text: str = ""
    text_error: str | None = None
    text_applied: bool = False
    copy_when_ready: bool = False
    last_preview_text: str = ""


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


def transcribe_audio(audio):
    # One MLX inference at a time, including adjacent dictations. The model is
    # already warm before the listener accepts the first key press.
    with _inference_lock:
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
    segments = result["segments"]
    # MLX Whisper can keep text with a high no-speech probability when its
    # token confidence is high. A fixed temperature also cannot retry a
    # repetition loop, so reject those segments before previewing or typing.
    accepted = [
        segment["text"] for segment in segments
        if segment["no_speech_prob"] < HALLUCINATION_NO_SPEECH_THRESHOLD
        and segment["compression_ratio"] <= HALLUCINATION_COMPRESSION_RATIO
    ]
    return "".join(accepted).strip()


def transcribe_and_type(app, chunks, capture=None):
    if not chunks:
        if capture is not None and capture.requested:
            app._queue_review("", capture)
        app.title = ICON_IDLE
        return

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE

    if duration < MIN_DURATION_SECONDS:
        if capture is not None and capture.requested:
            app._queue_review("", capture)
        app.title = ICON_IDLE
        return

    app.title = ICON_THINKING

    try:
        # Pass the float32 16kHz audio array straight to MLX Whisper. This avoids
        # the file-loading path and its ffmpeg dependency under launchd.
        # See CLAUDE.md "Gotcha: ffmpeg / launchd PATH".
        text = transcribe_audio(audio)
        if capture is not None and capture.requested:
            app._queue_review(text, capture)
        elif text:
            if ADD_TRAILING_SPACE:
                text += " "
            _typer.type(text)
    except Exception as e:
        _log_error("transcription failed: %r" % e)
        if capture is not None and capture.requested:
            capture.text_error = "Transkription fehlgeschlagen. Du kannst den Prompt selbst eingeben."
            app._queue_review("", capture)
    finally:
        app.title = ICON_IDLE


def transcription_worker(app):
    """Process dictations in capture order so results cannot overtake each other."""
    while True:
        item = _transcription_queue.get()
        if isinstance(item, tuple) and len(item) == 3:
            chunks, preview, capture = item
        elif isinstance(item, tuple):
            chunks, preview = item
            capture = None
        else:
            chunks, preview, capture = item, None, None
        try:
            if preview is not None:
                try:
                    preview.cancel()
                    preview.join()
                except Exception as exc:
                    _log_error("preview failed: %r" % exc)
            transcribe_and_type(app, chunks, capture)
        finally:
            if preview is not None:
                app._end_caption(preview)
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
        self._preview_worker = None
        self._preview_requested_samples = 0
        self._capture_session = None
        self._region_selection_session = None
        self._capture_hotkey_pending = False
        self._hotkey_down = False
        self._stop_on_release = False
        self._live_preview_text = ""
        self._review_pending = queue.Queue()
        self._review_waiting = None
        self._review_session = None
        self._review_panel = None
        self._review_open = False
        self._caption_lock = threading.Lock()
        self._caption_session = None
        self._caption_message = None
        self._overlay = None
        self._overlay_failed = False
        self._preview_timer = rumps.Timer(self._poll_live_preview, 0.12)
        self._preview_timer.start()

        # Modell im Hintergrund laden
        self.title = "Dic ..."
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        prune_old_captures()
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

    def _set_caption(self, session, message):
        with self._caption_lock:
            if self._caption_session is session:
                self._caption_message = message

    def _set_preview_text(self, session, message):
        with self._caption_lock:
            if self._caption_session is session:
                self._caption_message = message
                self._live_preview_text = message

    def _end_caption(self, session):
        with self._caption_lock:
            if self._caption_session is session:
                self._caption_session = None
                self._caption_message = None

    def _poll_live_preview(self, _):
        preview = None
        snapshot = None
        capture_pending = False
        with _lock:
            if recording and self._capture_hotkey_pending:
                capture_pending = True
                self._capture_hotkey_pending = False
            if recording and self._preview_worker is not None:
                samples = sum(len(chunk) for chunk in audio_chunks)
                threshold = (PREVIEW_START_SECONDS if self._preview_requested_samples == 0
                             else PREVIEW_INTERVAL_SECONDS) * SAMPLE_RATE
                if samples - self._preview_requested_samples >= threshold:
                    preview = self._preview_worker
                    snapshot = np.concatenate(audio_chunks, axis=0).flatten()
                    self._preview_requested_samples = samples
        if capture_pending:
            self._begin_capture("screen")
        if snapshot is not None:
            try:
                preview.offer(snapshot, len(snapshot) / SAMPLE_RATE)
            except RuntimeError:
                # Release can cancel a snapshot between capture and submission.
                pass

        with self._caption_lock:
            message = self._caption_message
            live_text = self._live_preview_text
        if not self._overlay_failed:
            try:
                if self._region_selection_session is not None or self._review_open:
                    if self._overlay is not None:
                        self._overlay.hide()
                elif message is not None:
                    if self._overlay is None:
                        self._overlay = LiveCaptionOverlay(self._begin_capture)
                    with _lock:
                        can_capture = recording and self._capture_session is not None and not self._capture_session.requested
                    self._overlay.show(message, can_capture=can_capture)
                elif self._overlay is not None:
                    self._overlay.hide()
            except Exception as exc:
                self._overlay_failed = True
                self.title = "Dic !"
                _log_error("live caption unavailable: %r" % exc)

        self._poll_review()
        if self._review_open and self._review_session is self._capture_session and live_text:
            session = self._review_session
            if session.last_preview_text != live_text:
                session.last_preview_text = live_text
                self._review_panel.set_live_text(live_text)

    def _begin_capture(self, mode):
        if mode not in ("screen", "region"):
            raise ValueError("Unknown screenshot mode")
        with _lock:
            if not recording or self._capture_session is None or self._capture_session.requested:
                return
            session = self._capture_session
            session.requested = True
            session.mode = mode
            session.error = None
            session.finished.clear()
        session.context = {"scope": "Ganzer Bildschirm" if mode == "screen" else "Ausschnitt"}
        if mode == "region":
            self._region_selection_session = session
        try:
            window = front_window()
            session.context.update(window.as_dict())
        except Exception:
            # App/window labels are optional context. Screen capture itself can
            # still proceed when the desktop has no ordinary foreground window.
            pass
        self._queue_review("", session, pending=True)

        display_id = None
        if mode == "screen":
            try:
                display_id = display_under_pointer()
            except Exception as exc:
                session.error = "Kein Bildschirm für die Aufnahme gefunden."
                _log_error("display selection failed: %r" % exc)
                session.finished.set()
                return

        def capture():
            cancelled_while_recording = False
            try:
                # Let AppKit remove the button before the screen is sampled.
                time.sleep(0.12)
                path = capture_screen(display_id) if mode == "screen" else capture_region()
                if path is None:
                    with _lock:
                        cancelled_while_recording = recording and self._capture_session is session
                        if cancelled_while_recording:
                            session.requested = False
                            session.mode = None
                            session.discarded = True
                    if not cancelled_while_recording:
                        session.error = "Ausschnitt abgebrochen."
                elif session.discarded:
                    discard(path)
                else:
                    session.image_path = path
            except Exception as exc:
                session.error = str(exc)
                _log_error("screenshot failed: %r" % exc)
            finally:
                session.finished.set()
                if mode == "region" and self._region_selection_session is session:
                    self._region_selection_session = None

        threading.Thread(target=capture, daemon=True).start()

    def _queue_review(self, text, session, pending=False):
        if pending:
            session.review_queued = True
            self._review_pending.put((text, session))
            return
        session.text = text
        session.text_ready.set()
        if not session.review_queued:
            session.review_queued = True
            self._review_pending.put((text, session))

    def _poll_review(self):
        pending = None
        if not self._review_open:
            pending = self._review_waiting
            if pending is None:
                try:
                    pending = self._review_pending.get_nowait()
                except queue.Empty:
                    pass
            if pending is not None:
                _, session = pending
                if not session.finished.is_set():
                    # Avoid covering the source before capture completes.
                    self._review_waiting = pending
                    pending = None
                else:
                    self._review_waiting = None
        if pending is not None:
            _, session = pending
            if session.discarded:
                return
            if self._overlay is not None:
                self._overlay.hide()
            if self._review_panel is None:
                self._review_panel = ReviewPanel(self._copy_prompt, self._discard_review)
            self._review_session = session
            self._review_open = True
            session.text_applied = session.text_ready.is_set()
            self._review_panel.show(
                session.text, session.image_path, session.context, session.error,
                pending=not session.finished.is_set(),
                text_pending=not session.text_applied,
                text_error=session.text_error,
                recording=recording and self._capture_session is session,
            )

        session = self._review_session
        if session is not None and session.finished.is_set() and not getattr(session, "shown", False):
            session.shown = True
            self._review_panel.set_image(session.image_path, session.context, session.error)
        if session is not None and session.text_ready.is_set() and not session.text_applied:
            session.text_applied = True
            self._review_panel.set_transcript(session.text, session.text_error)
        if session is not None and session.copy_when_ready and session.text_ready.is_set() and session.finished.is_set():
            session.copy_when_ready = False
            self._review_panel._copy()

    def _copy_prompt(self, text, image_path, context):
        session = self._review_session
        with _lock:
            still_recording = recording and self._capture_session is session
        if still_recording:
            session.copy_when_ready = True
            self._finish_recording()
            self._review_panel.set_recording(False)
            return False
        if session is not None and not session.text_ready.is_set():
            session.copy_when_ready = True
            self._review_panel.set_status("Diktat wird verarbeitet ...")
            return False
        prompt = format_prompt(text, image_path, context)
        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        if not pasteboard.setString_forType_(prompt, NSPasteboardTypeString):
            self._review_panel.set_status("Prompt konnte nicht kopiert werden.")
            return False
        self._review_panel.set_status("Prompt kopiert. In neuem oder bestehendem Task mit ⌘V einfügen.")
        return True

    def _discard_review(self, image_path, copied=False):
        if self._review_session is not None:
            self._review_session.discarded = True
            with _lock:
                still_recording = recording and self._capture_session is self._review_session
            if still_recording:
                self._finish_recording()
        if not copied:
            discard(image_path)
        self._review_session = None
        self._review_open = False

    def _finish_recording(self):
        global recording, audio_chunks
        if self._capture_hotkey_pending:
            self._capture_hotkey_pending = False
            self._begin_capture("screen")
        with _lock:
            if not recording:
                return
            recording = False
            chunks = audio_chunks
            audio_chunks = []
            preview = self._preview_worker
            self._preview_worker = None
            capture = self._capture_session
            self._capture_session = None
            self._capture_hotkey_pending = False
        if preview is not None:
            preview.cancel()
            self._set_caption(preview, "Prüfe Diktat ..." if capture.requested else "Setze Text ein ...")
        self.title = ICON_THINKING
        if capture is not None and capture.requested and not capture.review_queued:
            self._queue_review("", capture, pending=True)
        _transcription_queue.put((chunks, preview, capture))

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
            if key == SCREENSHOT_HOTKEY:
                with _lock:
                    if recording:
                        app._capture_hotkey_pending = True
                return
            if key == HOTKEY:
                if app._review_open:
                    return
                with _lock:
                    if app._hotkey_down:
                        return
                    app._hotkey_down = True
                    session = app._capture_session
                    app._stop_on_release = (
                        recording and not app._capture_hotkey_pending
                        and (session is None or not session.requested)
                    )
                    if not recording:
                        recording = True
                        audio_chunks = []
                        app._preview_requested_samples = 0
                        app._live_preview_text = ""
                        app._capture_session = CaptureSession()
                        worker = LatestAudioWorker(
                            transcribe_audio,
                            lambda event: app._set_preview_text(worker, event["text"]),
                            on_error=lambda exc: app._preview_failed(worker, exc),
                        )
                        app._preview_worker = worker
                        with app._caption_lock:
                            app._caption_session = worker
                            app._caption_message = "Höre zu ...  ·  Taste erneut drücken zum Beenden"
                app.title = ICON_RECORDING

        def on_release(key):
            if key == HOTKEY:
                with _lock:
                    app._hotkey_down = False
                    should_stop = app._stop_on_release
                    app._stop_on_release = False
                if should_stop:
                    app._finish_recording()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def _preview_failed(self, session, exc):
        _log_error("preview failed: %r" % exc)
        with _lock:
            screenshot_selected = self._capture_session is not None and self._capture_session.requested
        instruction = "Fertig & kopieren für den Endtext." if screenshot_selected else "Taste erneut drücken für den Endtext."
        self._set_caption(session, "Live-Vorschau nicht verfügbar. " + instruction)

    def _stop_listener(self):
        global recording, audio_chunks
        with _lock:
            was_recording = recording
        if was_recording:
            self._finish_recording()
        with _lock:
            recording = False
            audio_chunks = []
            preview = self._preview_worker
            self._preview_worker = None
            self._capture_session = None
            self._capture_hotkey_pending = False
            self._hotkey_down = False
            self._stop_on_release = False
        if preview is not None:
            preview.cancel()
            self._end_caption(preview)
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
