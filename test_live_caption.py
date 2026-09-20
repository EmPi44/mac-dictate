import unittest

from AppKit import NSApplication, NSMakePoint, NSMakeRect

from live_caption import LiveCaptionOverlay, position_near_cursor


class CursorPlacementTests(unittest.TestCase):
    def test_buttons_open_next_to_cursor(self):
        result = position_near_cursor(NSMakePoint(500, 400), NSMakeRect(0, 0, 1280, 720), 222, 42)
        self.assertEqual(result, (502, 402))

    def test_buttons_flip_inside_right_and_top_edges(self):
        result = position_near_cursor(NSMakePoint(1270, 700), NSMakeRect(0, 0, 1280, 720), 222, 42)
        self.assertEqual(result, (1046, 656))

    def test_buttons_handle_screen_left_of_main_display(self):
        result = position_near_cursor(NSMakePoint(-5, 700), NSMakeRect(-1280, 0, 1280, 720), 222, 42)
        self.assertEqual(result, (-229, 656))

    def test_buttons_route_screen_and_region_separately(self):
        NSApplication.sharedApplication()
        chosen = []
        overlay = LiveCaptionOverlay(chosen.append)

        for button, expected in ((overlay.screen_button, "screen"), (overlay.region_button, "region")):
            button.performClick_(None)
            self.assertIn(expected, chosen)
            self.assertFalse(overlay.capture_queued)

        self.assertEqual(chosen, ["screen", "region"])


if __name__ == "__main__":
    unittest.main()
