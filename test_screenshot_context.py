import tempfile
import unittest
from pathlib import Path
from unittest import mock

import screenshot_context


class ScreenshotContextTests(unittest.TestCase):
    @mock.patch("screenshot_context._check_capture_access")
    @mock.patch("screenshot_context.subprocess.run")
    def test_escape_from_region_selection_returns_without_image(self, run, access):
        run.return_value.stderr = b""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(screenshot_context, "CAPTURE_DIR", Path(directory)):
                self.assertIsNone(screenshot_context.capture_region())
                self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(run.call_args.args[0][1:4], ["-i", "-s", "-x"])

    @mock.patch("screenshot_context._check_capture_access")
    @mock.patch("screenshot_context.subprocess.run")
    def test_completed_region_returns_local_png(self, run, access):
        def create_image(command, **_kwargs):
            Path(command[-1]).write_bytes(b"PNG fixture")
            return mock.Mock(stderr=b"")

        run.side_effect = create_image
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(screenshot_context, "CAPTURE_DIR", Path(directory)):
                path = screenshot_context.capture_region()
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Path(path).parent, Path(directory))


if __name__ == "__main__":
    unittest.main()
