import unittest

from prompt_package import format_prompt


class PromptPackageTests(unittest.TestCase):
    def test_plain_dictation_stays_plain(self):
        self.assertEqual(format_prompt(" Hallo Welt ", None, {}), "Hallo Welt")

    def test_screenshot_adds_context_for_either_task(self):
        result = format_prompt(
            "Finde den Fehler",
            "/tmp/dictation-example.png",
            {"scope": "Ausschnitt", "app": "Safari", "title": "Dashboard", "url": "https://example.com/page"},
        )
        self.assertIn("Finde den Fehler", result)
        self.assertIn("Aufnahme: Ausschnitt", result)
        self.assertIn("App: Safari", result)
        self.assertIn("Fenster: Dashboard", result)
        self.assertIn("URL: https://example.com/page", result)
        self.assertIn("Bilddatei: /tmp/dictation-example.png", result)


if __name__ == "__main__":
    unittest.main()
