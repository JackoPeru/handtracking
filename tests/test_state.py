import unittest

import numpy as np


class RuntimeStateTests(unittest.TestCase):
    def test_pointer_reset_restores_defaults_and_zeroes_motion(self):
        from handtracking_state import PointerState

        state = PointerState()
        state.pinch_held = True
        state.move_active = True
        state.pinch_started_at = 12.0
        state.motion_accum[:] = (4.0, -2.0)
        state.flow_travel = 9.0
        state.cursor_origin = (100, 200)
        state.reset()

        self.assertFalse(state.pinch_held)
        self.assertFalse(state.move_active)
        self.assertIsNone(state.pinch_started_at)
        self.assertTrue(np.array_equal(state.motion_accum, np.zeros(2)))
        self.assertEqual(state.flow_travel, 0.0)
        self.assertIsNone(state.cursor_origin)

    def test_scroll_reset_restores_defaults(self):
        from handtracking_state import ScrollState

        state = ScrollState(active=True, candidate_at=1.0, release_at=2.0, residual=40.0)
        state.reset()
        self.assertEqual((state.active, state.candidate_at, state.release_at, state.residual),
                         (False, None, None, 0.0))

    def test_volume_reset_clears_histories(self):
        from handtracking_state import VolumeState

        state = VolumeState()
        state.active = True
        state.delta_history.extend((1.0, 2.0))
        state.vote_history.extend((1.0, 0.0))
        state.reset()
        self.assertFalse(state.active)
        self.assertEqual(list(state.delta_history), [])
        self.assertEqual(list(state.vote_history), [])

    def test_swipe_reset_preserves_only_cooldown_when_requested(self):
        from handtracking_state import SwipeState

        state = SwipeState()
        state.tracking = True
        state.cooldown_until = 9.0
        state.pose_history.extend((0.4, 0.8))
        state.reset(preserve_cooldown=True)
        self.assertFalse(state.tracking)
        self.assertEqual(state.cooldown_until, 9.0)
        self.assertEqual(list(state.pose_history), [])

    def test_radial_two_hand_and_spock_reset(self):
        from handtracking_state import RadialState, SpockState, TwoHandState

        radial = RadialState(active=True, selected="LEFT", pinch_latched=True)
        two_hand = TwoHandState(active=True, zoom_residual=240.0, points=((0, 0), (1, 1)))
        spock = SpockState(latched=True, blocking=True, progress=1.0)
        radial.reset()
        two_hand.reset()
        spock.reset()

        self.assertFalse(radial.active)
        self.assertIsNone(radial.selected)
        self.assertFalse(radial.pinch_latched)
        self.assertFalse(two_hand.active)
        self.assertEqual(two_hand.zoom_residual, 0.0)
        self.assertIsNone(two_hand.points)
        self.assertFalse(spock.latched)
        self.assertFalse(spock.blocking)
        self.assertEqual(spock.progress, 0.0)

    def test_flow_reset_zeroes_vectors(self):
        from handtracking_state import FlowState

        state = FlowState()
        state.active = True
        state.virtual[:] = (3.0, 5.0)
        state.filtered[:] = (4.0, 6.0)
        state.reset()
        self.assertFalse(state.active)
        self.assertTrue(np.array_equal(state.virtual, np.zeros(2)))
        self.assertTrue(np.array_equal(state.filtered, np.zeros(2)))

    def test_flow_clear_motion_preserves_tracking_anchors(self):
        from handtracking_state import FlowState

        state = FlowState()
        state.prev_gray = object()
        state.points = object()
        state.active = True
        state.motion_scale = 1.2
        state.last_success = 7.0
        state.virtual[:] = (2.0, 3.0)
        state.filtered[:] = (4.0, 5.0)
        state.prev_filtered[:] = (1.0, 2.0)
        state.time = 9.0
        state.clear_motion()

        self.assertTrue(state.active)
        self.assertIsNotNone(state.prev_gray)
        self.assertIsNotNone(state.points)
        self.assertEqual(state.motion_scale, 1.2)
        self.assertEqual(state.last_success, 7.0)
        self.assertIsNone(state.time)
        self.assertTrue(np.array_equal(state.virtual, np.zeros(2)))
        self.assertTrue(np.array_equal(state.filtered, np.zeros(2)))
        self.assertTrue(np.array_equal(state.prev_filtered, np.zeros(2)))

    def test_swipe_cancel_tracking_preserves_cooldown_and_pose_history(self):
        from handtracking_state import SwipeState

        state = SwipeState(tracking=True, cooldown_until=12.0, flow_started_at=3.0,
                           flow_accum_x=5.0, flow_accum_y=2.0)
        state.pose_history.extend((0.3, 0.8))
        state.pose_last_seen = 4.0
        state.cancel_tracking()

        self.assertFalse(state.tracking)
        self.assertEqual(state.cooldown_until, 12.0)
        self.assertEqual(list(state.pose_history), [0.3, 0.8])
        self.assertIsNone(state.pose_last_seen)
        self.assertIsNone(state.flow_started_at)
        self.assertEqual((state.flow_accum_x, state.flow_accum_y), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
