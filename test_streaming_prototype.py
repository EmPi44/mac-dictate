import threading
import time
import unittest

import numpy as np

from live_preview import LatestAudioWorker


class LatestAudioWorkerTests(unittest.TestCase):
    def test_replaces_obsolete_pending_audio_and_emits_one_final(self):
        decoding = threading.Event()
        release = threading.Event()
        events = []
        calls = []

        def decode(audio):
            calls.append(len(audio))
            decoding.set()
            if len(calls) == 1:
                self.assertTrue(release.wait(2))
            return str(len(audio))

        worker = LatestAudioWorker(decode, events.append)
        worker.offer(np.ones(100, dtype=np.float32), 1)
        self.assertTrue(decoding.wait(2))
        worker.offer(np.ones(200, dtype=np.float32), 2)
        worker.offer(np.ones(300, dtype=np.float32), 3)
        worker.offer(np.ones(400, dtype=np.float32), 4, final=True)
        release.set()
        worker.join()

        self.assertEqual(calls, [100, 400])
        self.assertEqual([event["kind"] for event in events], ["final"])
        self.assertEqual(events[0]["text"], "400")
        self.assertEqual(worker.max_pending, 1)

    def test_digital_silence_produces_no_text_or_decode(self):
        events = []
        worker = LatestAudioWorker(lambda _: self.fail("silence decoded"), events.append)
        worker.offer(np.zeros(16000, dtype=np.float32), 1)
        worker.offer(np.zeros(32000, dtype=np.float32), 2, final=True)
        worker.join()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "final")
        self.assertEqual(events[0]["text"], "")

    def test_normal_preview_precedes_final(self):
        events = []
        worker = LatestAudioWorker(lambda audio: str(len(audio)), events.append)
        worker.offer(np.ones(100, dtype=np.float32), 1)
        # Wait until the preview has actually been emitted before finalizing.
        deadline = time.monotonic() + 2
        while not events and time.monotonic() < deadline:
            time.sleep(0.001)
        worker.offer(np.ones(200, dtype=np.float32), 2, final=True)
        worker.join()

        self.assertEqual([(event["kind"], event["text"]) for event in events],
                         [("provisional", "100"), ("final", "200")])

    def test_cancel_discards_pending_and_inflight_preview(self):
        decoding = threading.Event()
        release = threading.Event()
        events = []

        def decode(_):
            decoding.set()
            self.assertTrue(release.wait(2))
            return "partial"

        worker = LatestAudioWorker(decode, events.append)
        worker.offer(np.ones(100, dtype=np.float32), 1)
        self.assertTrue(decoding.wait(2))
        worker.offer(np.ones(200, dtype=np.float32), 2)
        worker.cancel()
        release.set()
        worker.join()

        self.assertEqual(events, [])

    def test_decode_failure_reaches_error_handler(self):
        errors = []

        def fail(_):
            raise RuntimeError("model failed")

        worker = LatestAudioWorker(fail, lambda _: self.fail("unexpected text"),
                                   on_error=errors.append)
        worker.offer(np.ones(100, dtype=np.float32), 1)
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            worker.join()
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
