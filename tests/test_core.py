import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "handtracking_core.py"


def load_core():
    if not CORE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("handtracking_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreBehaviorTests(unittest.TestCase):
    def test_core_module_exists(self):
        self.assertTrue(CORE_PATH.exists(), "handtracking_core.py must exist")

    def test_pointer_gate_blocks_active_swipe(self):
        core = load_core()
        self.assertIsNotNone(core)
        self.assertFalse(core.pointer_mode_allowed(
            commands_enabled=True,
            spock_blocking=False,
            hand_count=1,
            paused=False,
            volume_active=False,
            two_hand_active=False,
            two_hand_candidate=False,
            radial_active=False,
            scroll_active=False,
            swipe_tracking=True,
            input_blocked=False,
        ))

    def test_spock_hold_does_not_credit_missing_time(self):
        core = load_core()
        self.assertIsNotNone(core)
        accumulated = 0.0
        accumulated = core.advance_confirmed_hold(accumulated, True, 0.05, 1.0)
        accumulated = core.advance_confirmed_hold(accumulated, False, 0.60, 1.0)
        accumulated = core.advance_confirmed_hold(accumulated, True, 0.05, 1.0)
        self.assertAlmostEqual(accumulated, 0.10, places=6)

    def test_hand_identity_prefers_same_handedness_when_crossing(self):
        core = load_core()
        self.assertIsNotNone(core)
        points = [(0.51, 0.5), (0.49, 0.5)]
        labels = ["Right", "Left"]
        index = core.choose_control_index(
            points,
            labels,
            previous_point=(0.50, 0.5),
            previous_label="Right",
        )
        self.assertEqual(index, 0)

    def test_flow_normalization_compensates_for_hand_distance(self):
        core = load_core()
        self.assertIsNotNone(core)
        near = core.normalize_flow_delta(4.0, 2.0, palm_scale=2.0)
        far = core.normalize_flow_delta(1.0, 0.5, palm_scale=0.5)
        self.assertEqual(near, far)

    def test_pixel_distance_uses_independent_frame_axes(self):
        core = load_core()
        self.assertIsNotNone(core)
        horizontal = core.normalized_points_pixel_distance(
            (0.25, 0.50), (0.25 + 90 / 640, 0.50), 640, 360,
        )
        vertical = core.normalized_points_pixel_distance(
            (0.50, 0.25), (0.50, 0.25 + 90 / 360), 640, 360,
        )
        self.assertAlmostEqual(horizontal, 90.0, places=5)
        self.assertAlmostEqual(vertical, 90.0, places=5)

    def test_stale_tracking_result_triggers_fail_safe(self):
        core = load_core()
        self.assertIsNotNone(core)
        self.assertFalse(core.tracking_result_is_stale(10.0, 10.09, 0.12))
        self.assertTrue(core.tracking_result_is_stale(10.0, 10.13, 0.12))
        self.assertTrue(core.tracking_result_is_stale(None, 10.13, 0.12))

    def test_camera_target_does_not_force_30_on_unknown_report(self):
        core = load_core()
        self.assertIsNotNone(core)
        self.assertEqual(core.choose_camera_target_fps(0.0, 60, 30), 60)
        self.assertEqual(core.choose_camera_target_fps(60.0, 60, 30), 60)
        self.assertEqual(core.choose_camera_target_fps(30.0, 60, 30), 30)

    def test_any_hand_fist_can_pause(self):
        core = load_core()
        self.assertIsNotNone(core)
        evidence = core.fist_evidence_from_hands(
            raw_fists=[False, True],
            strong_fists=[False, True],
            volume_scores=[0.1, 0.1],
            gap_scores=[0.2, 0.2],
            volume_active=False,
            volume_score_on=0.52,
            suppress_gap=0.54,
        )
        self.assertTrue(evidence)


if __name__ == "__main__":
    unittest.main()
