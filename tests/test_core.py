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


if __name__ == "__main__":
    unittest.main()

