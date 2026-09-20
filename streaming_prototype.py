#!/usr/bin/env python3
"""Isolated, paced MLX Whisper streaming experiment. Never types into other apps.

Input is a 16 kHz, mono, 16-bit PCM WAV file. A single worker decodes the
latest available prefix at each interval and one complete prefix at the end.
Intermediate hypotheses are replaceable; only the last event is final.
"""

import argparse
import json
import time
import wave

import numpy as np
from live_preview import LatestAudioWorker

MODEL = "mlx-community/whisper-small-mlx"
SAMPLE_RATE = 16000


def transcribe(audio):
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language=None,
        temperature=0.0,
        no_speech_threshold=0.4,
        condition_on_previous_text=False,
        fp16=True,
        verbose=None,
    )
    return result.get("text", "").strip()


def read_pcm_wav(path):
    with wave.open(path, "rb") as recording:
        if (recording.getframerate(), recording.getnchannels(), recording.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("expected 16 kHz mono 16-bit PCM WAV")
        samples = np.frombuffer(
            recording.readframes(recording.getnframes()), dtype="<i2"
        )
    return samples.astype(np.float32) / 32768.0


def run_paced(audio, interval_seconds=3.0, chunk_seconds=0.25, decode=transcribe, emit=print):
    """Replay audio at natural speed, emitting replaceable prefixes and one final result."""
    started = time.monotonic()

    def emit_event(event):
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        event["lag_seconds"] = round(event["elapsed_seconds"] - event["audio_seconds"], 3)
        emit(event)

    worker = LatestAudioWorker(decode, emit_event)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    next_preview = interval_seconds
    for end in range(chunk_samples, len(audio) + chunk_samples, chunk_samples):
        end = min(end, len(audio))
        time.sleep(max(0, started + end / SAMPLE_RATE - time.monotonic()))
        audio_seconds = end / SAMPLE_RATE
        if audio_seconds >= next_preview and end < len(audio):
            worker.offer(audio[:end], audio_seconds)
            next_preview += interval_seconds
    worker.offer(audio, len(audio) / SAMPLE_RATE, final=True)
    worker.join()
    return worker.max_pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="16 kHz mono 16-bit PCM WAV file")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    audio = read_pcm_wav(args.wav)
    # Match the running app's prewarmed model. The first invocation can fetch
    # the model, so it must not be included in the speaking-time measurement.
    transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    run_paced(audio, interval_seconds=args.interval,
              emit=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))


if __name__ == "__main__":
    main()
