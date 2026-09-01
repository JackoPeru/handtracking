import unittest
from collections import deque
from types import SimpleNamespace


class ProcessingTests(unittest.TestCase):
    def test_update_ema_metrics_initializes_and_smooths(self):
        from handtracking_processing import update_ema_metrics

        first = update_ema_metrics((0.0, 0.0, 0.0, 0.0), (20.0, 30.0, 40.0, 5.0))
        self.assertEqual(first, (20.0, 30.0, 40.0, 5.0))

        second = update_ema_metrics(first, (40.0, 50.0, 60.0, 15.0))
        self.assertAlmostEqual(second[0], 23.0)
        self.assertAlmostEqual(second[1], 33.0)
        self.assertAlmostEqual(second[2], 43.0)
        self.assertAlmostEqual(second[3], 6.5)

    def test_swipe_pose_update_tracks_recent_valid_pose_and_resets_when_blocked(self):
        from handtracking_processing import update_swipe_pose
        from handtracking_state import SwipeState

        state = SwipeState()
        update_swipe_pose(
            state,
            allowed=True,
            control_hand=object(),
            now=3.0,
            score_fn=lambda hand: (1.0, 0.2, 4),
        )
        self.assertEqual(state.pose_last_seen, 3.0)
        self.assertEqual(state.debug_extended, 4)

        update_swipe_pose(
            state,
            allowed=False,
            control_hand=object(),
            now=4.0,
            score_fn=lambda hand: (1.0, 0.2, 4),
        )
        self.assertIsNone(state.pose_last_seen)
        self.assertEqual(state.debug_score, 0.0)

    def test_precision_snap_arms_after_stable_hold_and_resets_when_disallowed(self):
        from handtracking_config import SNAP_ARM_SECONDS
        from handtracking_processing import update_precision_snap

        first = update_precision_snap(
            allowed=True,
            cursor_position=(100, 100),
            now=5.0,
            active=False,
            anchor=None,
            started_at=None,
        )
        self.assertFalse(first.active)
        second = update_precision_snap(
            allowed=True,
            cursor_position=(100, 100),
            now=5.0 + SNAP_ARM_SECONDS + 0.01,
            active=first.active,
            anchor=first.anchor,
            started_at=first.started_at,
        )
        self.assertTrue(second.active)

        reset = update_precision_snap(
            allowed=False,
            cursor_position=None,
            now=6.0,
            active=second.active,
            anchor=second.anchor,
            started_at=second.started_at,
        )
        self.assertFalse(reset.active)
        self.assertIsNone(reset.anchor)

    def test_analyze_hand_frame_centralizes_control_selection_and_mode_candidates(self):
        from handtracking_gestures import HandFeatures
        from handtracking_processing import analyze_hand_frame, update_hand_mode_metrics
        from handtracking_state import FlowState, VolumeState

        def make_hand():
            return [SimpleNamespace(x=0.1, y=0.2, z=0.0) for _ in range(21)]

        hands = [make_hand(), make_hand()]
        class_hands = [make_hand(), make_hand()]
        latest_result = SimpleNamespace(
            handedness=[
                [SimpleNamespace(category_name="Left")],
                [SimpleNamespace(category_name="Right")],
            ]
        )
        flow = FlowState(motion_scale=1.0)
        volume = VolumeState()
        votes = deque(maxlen=5)

        result = analyze_hand_frame(
            latest_result=latest_result,
            hands=hands,
            class_hands=class_hands,
            previous_point=(0.1, 0.1),
            previous_label="Left",
            paused_by_fist=False,
            fist_vote_history=votes,
            volume_active=volume.active,
            grip_fn=lambda hand: (0.2, 0.3, 1.0),
            point_fn=lambda hand: (
                (0.2, 0.3) if hand.landmarks is hands[0] else (0.7, 0.8)
            ),
            choose_fn=lambda points, labels, **kwargs: 1,
            fist_fn=lambda hand: False,
            strong_fist_fn=lambda hand: False,
        )
        mode = update_hand_mode_metrics(
            result,
            hands=hands,
            volume=volume,
            flow=flow,
            fold_fn=lambda hand: (2, 1.1),
            scroll_fn=lambda hand: False,
            pixel_distance_fn=lambda *args: 90.0,
            palm_scale_fn=lambda *args, **kwargs: 1.2,
        )

        self.assertEqual(result.control_index, 1)
        self.assertIsInstance(result.control_hand, HandFeatures)
        self.assertIs(result.control_hand.landmarks, hands[1])
        self.assertEqual(result.selected_handedness, "Right")
        self.assertFalse(result.paused_by_fist)
        self.assertFalse(result.fist_pending)
        self.assertFalse(mode.volume_candidate_now)
        self.assertAlmostEqual(flow.motion_scale, 1.036)
        self.assertEqual(mode.debug_fist_folded, 2)


if __name__ == "__main__":
    unittest.main()
