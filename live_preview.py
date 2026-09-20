"""One-inference worker shared by the live app and paced audio probe."""

import threading
import time

import numpy as np


class LatestAudioWorker:
    """Keep at most one waiting snapshot while one inference is running."""

    def __init__(self, decode, emit, clock=time.monotonic, on_error=None):
        self.decode = decode
        self.emit = emit
        self.clock = clock
        self.on_error = on_error
        self._condition = threading.Condition()
        self._pending = None
        self._finished = False
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.max_pending = 0
        self.error = None
        self._thread.start()

    def offer(self, audio, audio_seconds, final=False):
        with self._condition:
            if self._finished or self._cancelled:
                raise RuntimeError("worker already finished")
            if self._pending is not None and self._pending[2]:
                raise RuntimeError("cannot replace final audio")
            self._pending = (audio.copy(), audio_seconds, final)
            self.max_pending = max(self.max_pending, 1)
            if final:
                self._finished = True
            self._condition.notify()

    def cancel(self):
        """Discard queued previews; an in-flight decode finishes without publishing."""
        with self._condition:
            self._cancelled = True
            self._pending = None
            self._condition.notify()

    def join(self):
        self._thread.join()
        if self.error is not None:
            raise self.error

    def _run(self):
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._cancelled:
                        self._condition.wait()
                    if self._cancelled:
                        return
                    audio, audio_seconds, final = self._pending
                    self._pending = None
                started = self.clock()
                # Digital silence has no speech evidence. Real room noise needs
                # separate microphone validation before this could ship.
                text = self.decode(audio) if np.any(audio) else ""
                elapsed = self.clock() - started
                with self._condition:
                    stale = self._cancelled or (not final and self._pending is not None)
                if not stale and (text or final):
                    self.emit({
                        "kind": "final" if final else "provisional",
                        "audio_seconds": round(audio_seconds, 3),
                        "decode_seconds": round(elapsed, 3),
                        "text": text,
                    })
                if final:
                    return
        except Exception as exc:
            self.error = exc
            if self.on_error is not None:
                self.on_error(exc)
