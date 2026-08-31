import unittest


class GestureEngineTests(unittest.TestCase):
    def test_mode_priority_is_centralized_and_deterministic(self):
        from handtracking_engine import resolve_runtime_mode

        cases = (
            ({"spock_blocking": True, "commands_enabled": True}, "SPOCK"),
            ({"commands_enabled": False}, "LOCKED"),
            ({"paused": True}, "FIST"),
            ({"volume": True, "pointer_move": True}, "VOLUME"),
            ({"two_hand": True, "pointer_move": True}, "TWO_HAND"),
            ({"radial": True, "pointer_move": True}, "RADIAL"),
            ({"scrolling": True, "pointer_move": True}, "SCROLL"),
            ({"swipe": True, "pointer_move": True}, "SWIPE"),
            ({"pointer_move": True}, "POINTER"),
            ({"pointer_pinch": True}, "PINCH"),
            ({}, "MOUSE"),
        )
        defaults = dict(
            commands_enabled=True,
            spock_blocking=False,
            paused=False,
            volume=False,
            two_hand=False,
            radial=False,
            scrolling=False,
            swipe=False,
            pointer_move=False,
            pointer_pinch=False,
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                args = defaults | overrides
                self.assertEqual(resolve_runtime_mode(**args), expected)

    def test_engine_has_no_opencv_or_windows_dependencies(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "handtracking_engine.py")
        self.assertTrue(source.exists())
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("import cv2", text)
        self.assertNotIn("ctypes", text)
        self.assertNotIn("handtracking_windows", text)


if __name__ == "__main__":
    unittest.main()
