import unittest

import numpy as np

from handtracking_state import (
    FlowState,
    PointerState,
    RadialState,
    ScrollState,
    SpockState,
    SwipeState,
    TwoHandState,
    VolumeState,
)


class FakeCursor:
    def __init__(self):
        self.active = False
        self.moves = []
        self.sync_calls = []

    def sync(self, active):
        self.active = active
        self.sync_calls.append(active)

    def add_delta(self, dx, dy, *, screen_size=None):
        self.moves.append((dx, dy, screen_size))


class FlowModuleTests(unittest.TestCase):
    def test_summarize_lk_tracks_rejects_large_forward_backward_error(self):
        from handtracking_flow import summarize_lk_tracks

        old = np.array([[[10, 10]], [[20, 20]], [[30, 30]]], dtype=np.float32)
        new = old + np.array([[[2, 0]], [[2, 0]], [[2, 0]]], dtype=np.float32)
        back = old + np.array([[[20, 0]], [[20, 0]], [[20, 0]]], dtype=np.float32)
        status = np.ones((3, 1), dtype=np.uint8)

        self.assertIsNone(
            summarize_lk_tracks(old, new, status, None, back, status, 1.0)
        )

    def test_summarize_lk_tracks_returns_normalized_median_motion(self):
        from handtracking_flow import summarize_lk_tracks

        old = np.array([[[10, 10]], [[20, 20]], [[30, 30]], [[40, 40]]], dtype=np.float32)
        new = old + np.array([[[3, 2]], [[3, 2]], [[3, 2]], [[20, 20]]], dtype=np.float32)
        back = old.copy()
        status = np.ones((4, 1), dtype=np.uint8)
        err = np.zeros((4, 1), dtype=np.float32)

        result = summarize_lk_tracks(old, new, status, err, back, status, 2.0)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.dx, 1.5, places=4)
        self.assertAlmostEqual(result.dy, 1.0, places=4)

    def test_scroll_dispatch_emits_mouse_wheel_without_runtime_global(self):
        from handtracking_flow import dispatch_flow_motion

        flow = FlowState()
        pointer = PointerState()
        volume = VolumeState()
        two_hand = TwoHandState()
        radial = RadialState()
        scroll = ScrollState(active=True)
        swipe = SwipeState()
        cursor = FakeCursor()
        wheels = []

        dispatch_flow_motion(
            motion_dx=0.0,
            motion_dy=20.0,
            motion_mag=20.0,
            now=10.0,
            mp_result_stale=False,
            paused_by_fist=False,
            commands_enabled=True,
            spock_blocking=False,
            gesture_input_block_until=0.0,
            pointer=pointer,
            volume=volume,
            two_hand=two_hand,
            radial=radial,
            scroll=scroll,
            swipe=swipe,
            flow=flow,
            cursor=cursor,
            screen_w=1920,
            screen_h=1080,
            precision_snap_active=False,
            snap_anchor=None,
            snap_started_at=None,
            execute_swipe_cb=lambda direction: direction,
            mouse_wheel_cb=wheels.append,
        )

        self.assertTrue(wheels)
        self.assertNotEqual(wheels[0], 0)


if __name__ == "__main__":
    unittest.main()
