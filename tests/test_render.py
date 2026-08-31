import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


def point(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


class RenderTests(unittest.TestCase):
    def test_render_module_draws_hand_without_mutating_landmarks(self):
        import handtracking_render as render

        hand = [point(0.5, 0.5) for _ in range(21)]
        before = [(p.x, p.y, p.z) for p in hand]
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with (
            mock.patch.object(render.cv2, "line"),
            mock.patch.object(render.cv2, "circle"),
        ):
            render.draw_hand(frame, hand, pinch_active=False)

        self.assertEqual(before, [(p.x, p.y, p.z) for p in hand])

    def test_two_hand_overlay_labels_zoom_only(self):
        import handtracking_render as render

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with mock.patch.object(render.cv2, "putText") as put_text:
            render.draw_two_hand_transform(frame, (0.2, 0.5), (0.8, 0.5))

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("ZOOM", labels)


if __name__ == "__main__":
    unittest.main()
