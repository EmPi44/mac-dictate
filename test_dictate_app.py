import queue
import threading
import unittest
from unittest import mock

import numpy as np

import dictate_app


class FakeApp:
    title = None


class FakeTyper:
    def __init__(self):
        self.typed = []

    def type(self, text):
        self.typed.append(text)


class DictateAppTests(unittest.TestCase):
    def setUp(self):
        self.app = FakeApp()
        self.typer = FakeTyper()
        self.original_typer = dictate_app._typer
        self.original_queue = dictate_app._transcription_queue
        dictate_app._typer = self.typer
        dictate_app._transcription_queue = queue.Queue()

    def tearDown(self):
        dictate_app._typer = self.original_typer
        dictate_app._transcription_queue = self.original_queue

    @staticmethod
    def chunks(seconds=1.0, value=0.1):
        samples = int(dictate_app.SAMPLE_RATE * seconds)
        return [np.full((samples, 1), value, dtype=np.float32)]

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_empty_audio_is_ignored(self, transcribe):
        dictate_app.transcribe_and_type(self.app, [])

        transcribe.assert_not_called()
        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_short_audio_is_ignored(self, transcribe):
        dictate_app.transcribe_and_type(self.app, self.chunks(seconds=0.1))

        transcribe.assert_not_called()
        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_valid_audio_uses_mlx_and_types_result(self, transcribe):
        transcribe.return_value = {"text": " Hallo Welt "}

        dictate_app.transcribe_and_type(self.app, self.chunks())

        transcribe.assert_called_once()
        kwargs = transcribe.call_args.kwargs
        self.assertEqual(kwargs["path_or_hf_repo"], dictate_app.MODEL)
        self.assertIsNone(kwargs["language"])
        self.assertTrue(kwargs["fp16"])
        self.assertIsNone(kwargs["verbose"])
        self.assertEqual(self.typer.typed, ["Hallo Welt "])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    @mock.patch("dictate_app._log_error")
    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_transcription_failure_is_logged(self, transcribe, log_error):
        transcribe.side_effect = RuntimeError("test failure")

        dictate_app.transcribe_and_type(self.app, self.chunks())

        log_error.assert_called_once()
        self.assertIn("test failure", log_error.call_args.args[0])
        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_worker_preserves_two_dictations_in_capture_order(self, transcribe):
        transcribe.side_effect = [{"text": "Erste"}, {"text": "Zweite"}]
        worker = threading.Thread(
            target=dictate_app.transcription_worker,
            args=(self.app,),
            daemon=True,
        )
        worker.start()

        dictate_app._transcription_queue.put(self.chunks(value=0.1))
        dictate_app._transcription_queue.put(self.chunks(value=0.2))
        dictate_app._transcription_queue.join()

        self.assertEqual(self.typer.typed, ["Erste ", "Zweite "])
        self.assertEqual(transcribe.call_count, 2)


if __name__ == "__main__":
    unittest.main()
