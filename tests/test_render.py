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

    def test_draw_runtime_overlays_routes_control_hand_and_modes(self):
        import handtracking_render as render

        hands = [[point(0.5, 0.5) for _ in range(21)] for _ in range(2)]
        result = SimpleNamespace(hand_landmarks=hands)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        with (
            mock.patch.object(render, "draw_hand") as draw_hand,
            mock.patch.object(render, "draw_radial_menu") as draw_radial,
            mock.patch.object(render, "draw_two_hand_transform") as draw_two_hand,
        ):
            render.draw_runtime_overlays(
                frame,
                latest_result=result,
                fist_states=[False, True],
                control_index=1,
                pinch_active=True,
                scroll_active=True,
                volume_active=False,
                radial_active=True,
                radial_center=(0.5, 0.5),
                radial_selected="LEFT",
                two_hand_active=True,
                two_hand_points=((0.2, 0.5), (0.8, 0.5)),
            )

        self.assertEqual(draw_hand.call_count, 2)
        self.assertFalse(draw_hand.call_args_list[0].kwargs["pinch_active"])
        self.assertTrue(draw_hand.call_args_list[1].kwargs["pinch_active"])
        self.assertTrue(draw_hand.call_args_list[1].kwargs["scrolling"])
        draw_radial.assert_called_once_with(frame, (0.5, 0.5), "LEFT")
        draw_two_hand.assert_called_once_with(frame, (0.2, 0.5), (0.8, 0.5))


if __name__ == "__main__":
    unittest.main()
