import queue
import threading
import unittest
from unittest import mock

import numpy as np
from AppKit import NSApplication

import dictate_app
from review_panel import ReviewPanel


class FakeApp:
    title = None

    def __init__(self):
        self.ended = []
        self.reviews = []

    def _end_caption(self, session):
        self.ended.append(session)

    def _queue_review(self, text, capture):
        self.reviews.append((text, capture))


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

    @staticmethod
    def result(text, no_speech_prob=0.01, compression_ratio=1.0):
        return {"text": text, "segments": [{
            "text": text,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio,
        }]}

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_empty_audio_is_ignored(self, transcribe):
        dictate_app.transcribe_and_type(self.app, [])

        transcribe.assert_not_called()
        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    def test_menu_label_tracks_dictation_state(self):
        app = mock.Mock(active=True)
        item = mock.Mock()

        dictate_app.DictateApp.toggle(app, item)
        self.assertFalse(app.active)
        self.assertEqual(item.title, "Dictation: OFF")
        app._stop_listener.assert_called_once()

        dictate_app.DictateApp.toggle(app, item)
        self.assertTrue(app.active)
        self.assertEqual(item.title, "Dictation: ON")
        app._start_listener.assert_called_once()

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_short_audio_is_ignored(self, transcribe):
        dictate_app.transcribe_and_type(self.app, self.chunks(seconds=0.1))

        transcribe.assert_not_called()
        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_valid_audio_uses_mlx_and_types_result(self, transcribe):
        transcribe.return_value = self.result(" Hallo Welt ")

        dictate_app.transcribe_and_type(self.app, self.chunks())

        transcribe.assert_called_once()
        kwargs = transcribe.call_args.kwargs
        self.assertEqual(kwargs["path_or_hf_repo"], dictate_app.MODEL)
        self.assertIsNone(kwargs["language"])
        self.assertTrue(kwargs["fp16"])
        self.assertIsNone(kwargs["verbose"])
        self.assertEqual(self.typer.typed, ["Hallo Welt "])
        self.assertEqual(self.app.title, dictate_app.ICON_IDLE)

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_silent_hallucination_is_not_typed(self, transcribe):
        transcribe.return_value = self.result(" Thank you.", no_speech_prob=0.86)

        dictate_app.transcribe_and_type(self.app, self.chunks())

        self.assertEqual(self.typer.typed, [])

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_quiet_short_speech_is_not_discarded(self, transcribe):
        # The installed model gave quiet "Ja" a no-speech probability of 0.643.
        transcribe.return_value = self.result(" Ja.", no_speech_prob=0.643)

        dictate_app.transcribe_and_type(self.app, self.chunks())

        self.assertEqual(self.typer.typed, ["Ja. "])

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_repetition_loop_is_not_typed(self, transcribe):
        transcribe.return_value = self.result(" ව" * 100, compression_ratio=20.8)

        dictate_app.transcribe_and_type(self.app, self.chunks())

        self.assertEqual(self.typer.typed, [])

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_keeps_speech_when_later_segment_hallucinates(self, transcribe):
        spoken = self.result(" Hallo Welt ")["segments"][0]
        hallucinated = self.result(" Thank you.", no_speech_prob=0.86)["segments"][0]
        transcribe.return_value = {
            "text": " Hallo Welt Thank you.",
            "segments": [spoken, hallucinated],
        }

        dictate_app.transcribe_and_type(self.app, self.chunks())

        self.assertEqual(self.typer.typed, ["Hallo Welt "])

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_screenshot_dictation_opens_review_without_typing(self, transcribe):
        transcribe.return_value = self.result(" Prüfe diesen Fehler ")
        capture = dictate_app.CaptureSession(requested=True)

        dictate_app.transcribe_and_type(self.app, self.chunks(), capture)

        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.reviews, [("Prüfe diesen Fehler", capture)])

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_short_screenshot_dictation_still_opens_review(self, transcribe):
        capture = dictate_app.CaptureSession(requested=True)

        dictate_app.transcribe_and_type(self.app, self.chunks(seconds=0.1), capture)

        transcribe.assert_not_called()
        self.assertEqual(self.typer.typed, [])
        self.assertEqual(self.app.reviews, [("", capture)])

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
        transcribe.side_effect = [self.result("Erste"), self.result("Zweite")]
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

    @mock.patch("dictate_app.mlx_whisper.transcribe")
    def test_release_finishes_preview_before_inserting_one_final_text(self, transcribe):
        transcribe.return_value = self.result("Endgültiger Text")
        preview = mock.Mock()
        worker = threading.Thread(
            target=dictate_app.transcription_worker,
            args=(self.app,),
            daemon=True,
        )
        worker.start()

        dictate_app._transcription_queue.put((self.chunks(), preview))
        dictate_app._transcription_queue.join()

        preview.cancel.assert_called_once()
        preview.join.assert_called_once()
        self.assertEqual(self.typer.typed, ["Endgültiger Text "])
        self.assertEqual(self.app.ended, [preview])

    def test_live_preview_waits_for_two_seconds_then_uses_latest_audio(self):
        preview = mock.Mock()
        app = mock.Mock()
        app._preview_worker = preview
        app._preview_requested_samples = 0
        app._caption_lock = threading.Lock()
        app._caption_message = "Höre zu ..."
        app._overlay_failed = True
        original_recording = dictate_app.recording
        original_chunks = dictate_app.audio_chunks
        try:
            dictate_app.recording = True
            dictate_app.audio_chunks = self.chunks(seconds=1.5)
            dictate_app.DictateApp._poll_live_preview(app, None)
            preview.offer.assert_not_called()

            dictate_app.audio_chunks += self.chunks(seconds=0.5)
            dictate_app.DictateApp._poll_live_preview(app, None)
            self.assertEqual(preview.offer.call_count, 1)
            first_audio, first_duration = preview.offer.call_args.args
            self.assertEqual(len(first_audio), 2 * dictate_app.SAMPLE_RATE)
            self.assertEqual(first_duration, 2.0)

            dictate_app.audio_chunks += self.chunks(seconds=1.5)
            dictate_app.DictateApp._poll_live_preview(app, None)
            self.assertEqual(preview.offer.call_count, 2)
            second_audio, second_duration = preview.offer.call_args.args
            self.assertEqual(len(second_audio), int(3.5 * dictate_app.SAMPLE_RATE))
            self.assertEqual(second_duration, 3.5)
        finally:
            dictate_app.recording = original_recording
            dictate_app.audio_chunks = original_chunks

    def test_review_queue_keeps_multiple_screenshot_dictations(self):
        app = mock.Mock()
        app._review_pending = queue.Queue()
        first = dictate_app.CaptureSession(requested=True)
        second = dictate_app.CaptureSession(requested=True)

        dictate_app.DictateApp._queue_review(app, "Erste", first)
        dictate_app.DictateApp._queue_review(app, "Zweite", second)

        self.assertEqual(app._review_pending.get_nowait(), ("Erste", first))
        self.assertEqual(app._review_pending.get_nowait(), ("Zweite", second))

    @mock.patch("dictate_app.keyboard.Listener")
    @mock.patch("dictate_app.sd.InputStream")
    @mock.patch("dictate_app.LatestAudioWorker")
    def test_one_click_starts_and_next_click_stops(self, worker_class, stream_class, listener_class):
        app = mock.Mock()
        app.active = True
        app._review_open = False
        app._hotkey_down = False
        app._stop_on_release = False
        app._capture_hotkey_pending = False
        app._caption_lock = threading.Lock()
        app._finish_recording.side_effect = lambda: dictate_app.DictateApp._finish_recording(app)
        old_recording = dictate_app.recording
        old_chunks = dictate_app.audio_chunks
        try:
            dictate_app.recording = False
            dictate_app.audio_chunks = []
            dictate_app.DictateApp._start_listener(app)
            on_press = listener_class.call_args.kwargs["on_press"]
            on_release = listener_class.call_args.kwargs["on_release"]

            on_press(dictate_app.HOTKEY)
            on_press(dictate_app.HOTKEY)  # Key repeat must not stop the recording.
            self.assertTrue(dictate_app.recording)
            self.assertEqual(worker_class.call_count, 1)
            dictate_app.audio_chunks = self.chunks()
            on_release(dictate_app.HOTKEY)
            self.assertTrue(dictate_app.recording)
            on_press(dictate_app.HOTKEY)
            on_release(dictate_app.HOTKEY)
            self.assertFalse(dictate_app.recording)
            self.assertEqual(dictate_app._transcription_queue.qsize(), 1)
            chunks, preview, capture = dictate_app._transcription_queue.get_nowait()
            self.assertEqual(len(chunks), 1)
            self.assertIs(preview, worker_class.return_value)
            self.assertIsInstance(capture, dictate_app.CaptureSession)
        finally:
            dictate_app.recording = old_recording
            dictate_app.audio_chunks = old_chunks

    @mock.patch("dictate_app.keyboard.Listener")
    @mock.patch("dictate_app.sd.InputStream")
    @mock.patch("dictate_app.LatestAudioWorker")
    def test_screenshot_recording_continues_after_release_until_copy(self, worker_class, stream_class, listener_class):
        app = mock.Mock()
        app.active = True
        app._review_open = False
        app._hotkey_down = False
        app._stop_on_release = False
        app._capture_hotkey_pending = False
        app._caption_lock = threading.Lock()
        app._finish_recording.side_effect = lambda: dictate_app.DictateApp._finish_recording(app)
        old_recording = dictate_app.recording
        old_chunks = dictate_app.audio_chunks
        try:
            dictate_app.recording = False
            dictate_app.audio_chunks = []
            dictate_app.DictateApp._start_listener(app)
            on_press = listener_class.call_args.kwargs["on_press"]
            on_release = listener_class.call_args.kwargs["on_release"]

            on_press(dictate_app.HOTKEY)
            on_release(dictate_app.HOTKEY)
            session = app._capture_session
            session.requested = True
            session.mode = "screen"
            session.finished.set()
            session.review_queued = True
            app._review_session = session
            dictate_app.audio_chunks = self.chunks()
            self.assertTrue(dictate_app.recording)
            self.assertTrue(dictate_app._transcription_queue.empty())
            later_speech = self.chunks(value=0.2)[0]
            dictate_app.audio_callback(later_speech, len(later_speech), None, None)
            self.assertFalse(dictate_app.DictateApp._copy_prompt(app, "", session.image_path, session.context))
            self.assertFalse(dictate_app.recording)
            self.assertTrue(session.copy_when_ready)
            captured_chunks, _, captured_session = dictate_app._transcription_queue.get_nowait()
            self.assertIs(captured_session, session)
            self.assertEqual(len(captured_chunks), 2)
            self.assertAlmostEqual(float(captured_chunks[1].mean()), 0.2, places=5)
        finally:
            dictate_app.recording = old_recording
            dictate_app.audio_chunks = old_chunks

    @mock.patch("dictate_app.keyboard.Listener")
    @mock.patch("dictate_app.sd.InputStream")
    @mock.patch("dictate_app.LatestAudioWorker")
    def test_pending_f13_capture_keeps_recording_on_release(self, worker_class, stream_class, listener_class):
        app = mock.Mock()
        app.active = True
        app._review_open = False
        app._hotkey_down = False
        app._stop_on_release = False
        app._capture_hotkey_pending = False
        app._caption_lock = threading.Lock()
        old_recording = dictate_app.recording
        old_chunks = dictate_app.audio_chunks
        try:
            dictate_app.recording = False
            dictate_app.audio_chunks = []
            dictate_app.DictateApp._start_listener(app)
            on_press = listener_class.call_args.kwargs["on_press"]
            on_release = listener_class.call_args.kwargs["on_release"]
            on_press(dictate_app.HOTKEY)
            on_release(dictate_app.HOTKEY)
            on_press(dictate_app.SCREENSHOT_HOTKEY)
            on_press(dictate_app.HOTKEY)
            on_release(dictate_app.HOTKEY)

            self.assertTrue(dictate_app.recording)
            app._finish_recording.assert_not_called()
        finally:
            dictate_app.recording = old_recording
            dictate_app.audio_chunks = old_chunks

    @mock.patch("dictate_app.capture_region", return_value=None)
    @mock.patch("dictate_app.front_window", side_effect=RuntimeError("no window"))
    @mock.patch("dictate_app.threading.Thread")
    def test_cancelled_region_returns_to_recording(self, thread_class, front, capture):
        app = mock.Mock()
        session = dictate_app.CaptureSession()
        app._capture_session = session
        app._hotkey_down = False
        old_recording = dictate_app.recording
        try:
            dictate_app.recording = True
            thread_class.return_value.start.side_effect = lambda: thread_class.call_args.kwargs["target"]()

            dictate_app.DictateApp._begin_capture(app, "region")

            self.assertFalse(session.requested)
            self.assertIsNone(session.error)
            self.assertTrue(session.finished.is_set())
            self.assertTrue(session.discarded)
            self.assertEqual(session.mode, None)
            capture.assert_called_once()
            app._finish_recording.assert_not_called()
        finally:
            dictate_app.recording = old_recording

    def test_region_review_waits_for_selection_to_finish(self):
        app = mock.Mock()
        app._review_pending = queue.Queue()
        app._review_waiting = None
        app._review_open = False
        app._review_session = None
        app._overlay = None
        session = dictate_app.CaptureSession(requested=True, mode="region")
        session.context = {"scope": "Ausschnitt"}
        dictate_app.DictateApp._queue_review(app, "", session, pending=True)
        self.assertFalse(app._review_pending.empty())
        dictate_app.DictateApp._poll_review(app)
        app._review_panel.show.assert_not_called()

        session.finished.set()
        dictate_app.DictateApp._poll_review(app)
        app._review_panel.show.assert_called_once()
        self.assertTrue(app._review_panel.show.call_args.kwargs["text_pending"])

        dictate_app.DictateApp._queue_review(app, "Prüfe diesen Bereich", session)
        dictate_app.DictateApp._poll_review(app)
        app._review_panel.set_transcript.assert_called_once_with("Prüfe diesen Bereich", None)

    def test_review_opens_while_final_transcription_is_unfinished(self):
        app = mock.Mock()
        app._review_pending = queue.Queue()
        app._review_waiting = None
        app._review_open = False
        app._review_session = None
        app._overlay = None
        session = dictate_app.CaptureSession(requested=True, mode="screen")
        session.finished.set()
        dictate_app.DictateApp._queue_review(app, "", session, pending=True)

        dictate_app.DictateApp._poll_review(app)

        app._review_panel.show.assert_called_once()
        self.assertTrue(app._review_panel.show.call_args.kwargs["text_pending"])
        self.assertFalse(session.text_ready.is_set())

    def test_full_screen_card_waits_until_capture_finishes(self):
        app = mock.Mock()
        app._review_pending = queue.Queue()
        app._review_waiting = None
        app._review_open = False
        app._review_session = None
        app._overlay = None
        session = dictate_app.CaptureSession(requested=True, mode="screen")
        dictate_app.DictateApp._queue_review(app, "", session, pending=True)

        dictate_app.DictateApp._poll_review(app)
        app._review_panel.show.assert_not_called()
        session.finished.set()
        dictate_app.DictateApp._poll_review(app)
        app._review_panel.show.assert_called_once()

    def test_completed_transcription_copies_after_finish_was_clicked(self):
        app = mock.Mock()
        app._review_open = True
        session = dictate_app.CaptureSession(requested=True, mode="screen", copy_when_ready=True)
        session.finished.set()
        session.text = "Vollständiger Text"
        session.text_ready.set()
        app._review_session = session

        dictate_app.DictateApp._poll_review(app)

        app._review_panel.set_transcript.assert_called_once_with("Vollständiger Text", None)
        app._review_panel._copy.assert_called_once()
        self.assertFalse(session.copy_when_ready)

    def test_review_shows_live_text_until_finish(self):
        NSApplication.sharedApplication()
        panel = ReviewPanel(lambda *args: True, lambda *args: None)
        panel.show("", None, {}, text_pending=True, recording=True)
        self.assertEqual(panel.copy_button.title(), "Finish & Copy")
        self.assertTrue(panel.copy_button.isEnabled())
        self.assertFalse(panel.text_view.isEditable())

        panel.set_live_text("Vorläufiger Text")
        self.assertEqual(str(panel.text_view.string()), "Vorläufiger Text")
        panel.set_recording(False)
        self.assertFalse(panel.copy_button.isEnabled())
        panel.set_transcript("Vollständiger Text")
        self.assertEqual(str(panel.text_view.string()), "Vollständiger Text")
        self.assertTrue(panel.copy_button.isEnabled())
        panel.close()

    def test_review_disables_copy_until_text_is_ready(self):
        NSApplication.sharedApplication()
        panel = ReviewPanel(lambda *args: True, lambda *args: None)
        panel.show("", None, {}, pending=False, text_pending=True)
        self.assertFalse(panel.copy_button.isEnabled())
        self.assertFalse(panel.text_view.isEditable())
        self.assertEqual(panel.status.stringValue(), "Processing dictation ...")

        panel.set_transcript("Fertiger Text")
        self.assertTrue(panel.copy_button.isEnabled())
        self.assertTrue(panel.text_view.isEditable())
        self.assertEqual(str(panel.text_view.string()), "Fertiger Text")
        panel.close()

    def test_empty_description_is_a_placeholder_only_while_pending(self):
        NSApplication.sharedApplication()
        panel = ReviewPanel(lambda *args: True, lambda *args: None)
        panel.show("", None, {}, text_pending=True, recording=True)
        self.assertIn("Keep speaking", str(panel.text_view.string()))

        panel.set_recording(False)
        self.assertEqual(str(panel.text_view.string()), "Finalizing dictation ...")
        panel.set_transcript("")
        self.assertEqual(str(panel.text_view.string()), "")
        panel.close()

    def test_review_shows_transcription_error_if_it_finished_before_opening(self):
        NSApplication.sharedApplication()
        panel = ReviewPanel(lambda *args: True, lambda *args: None)
        panel.show("", None, {}, text_error="Transkription fehlgeschlagen.")
        self.assertEqual(panel.status.stringValue(), "Transkription fehlgeschlagen.")
        self.assertTrue(panel.text_view.isEditable())
        panel.close()

    @mock.patch("dictate_app.capture_screen", return_value="/tmp/screen-fixture.png")
    @mock.patch("dictate_app.display_under_pointer", return_value=2)
    @mock.patch("dictate_app.front_window", side_effect=RuntimeError("no window"))
    @mock.patch("dictate_app.threading.Thread")
    def test_full_screen_uses_display_under_pointer(self, thread_class, front, display, capture):
        app = mock.Mock()
        session = dictate_app.CaptureSession()
        app._capture_session = session
        old_recording = dictate_app.recording
        try:
            dictate_app.recording = True
            thread_class.return_value.start.side_effect = lambda: thread_class.call_args.kwargs["target"]()

            dictate_app.DictateApp._begin_capture(app, "screen")

            display.assert_called_once()
            capture.assert_called_once_with(2)
            self.assertEqual(session.image_path, "/tmp/screen-fixture.png")
            self.assertEqual(session.context["scope"], "Full screen")
            self.assertTrue(session.finished.is_set())
            app._queue_review.assert_called_once_with("", session, pending=True)
        finally:
            dictate_app.recording = old_recording


if __name__ == "__main__":
    unittest.main()
