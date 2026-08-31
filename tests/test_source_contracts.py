import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "handtracking_runtime.py").read_text(encoding="utf-8")


class SourceContractTests(unittest.TestCase):
    def test_media_pipe_queue_does_not_copy_fresh_frames(self):
        self.assertNotIn(
            "mp_pending = (detect_frame.copy(), gray.copy()",
            SOURCE,
        )

    def test_camera_requests_mjpeg(self):
        self.assertIn("CAP_PROP_FOURCC", SOURCE)
        self.assertIn("MJPG", SOURCE)

    def test_camera_failure_is_explicit(self):
        self.assertIn('raise RuntimeError("Impossibile aprire la webcam")', SOURCE)

    def test_camera_and_mediapipe_diagnostics_are_visible(self):
        self.assertIn("camera_codec", SOURCE)
        self.assertIn("mp_error_count", SOURCE)
        self.assertIn("mp_last_error", SOURCE)
        self.assertIn("MP ERR", SOURCE)

    def test_mediapipe_errors_are_observable(self):
        self.assertRegex(SOURCE, r"except Exception as \w+")
        self.assertIn("mp_error_count", SOURCE)
        self.assertIn("mp_last_error", SOURCE)

    def test_runtime_snapshots_are_not_pytest_candidates_in_root(self):
        offenders = [p.name for p in ROOT.glob("test_pre_*.py")]
        self.assertEqual(offenders, [])

    def test_removed_legacy_pointer_symbols_do_not_return(self):
        legacy_symbols = (
            "CLICK_PINCH_ON", "CLICK_PINCH_OFF", "CLICK_CONFIRM_SECONDS",
            "CLICK_RELEASE_GRACE", "DRAG_PINCH_ON", "DRAG_PINCH_OFF",
            "DRAG_CONFIRM_SECONDS", "DRAG_DOMINANCE_MARGIN", "DRAG_RELEASE_GRACE",
            "def flow_points_from_pinch", "def is_volume_gesture",
            "def is_flat_swipe_pose", "def is_spock_pose",
        )
        for symbol in legacy_symbols:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, SOURCE)

    def test_dead_runtime_state_is_removed(self):
        dead_state = (
            "drag_candidate_at", "drag_release_at", "pinch_history",
            "pinch_candidate_at", "pinch_release_at", "pinch_started_at",
            "mouse_control_ref", "pointer_contact_ref", "swipe_cursor_origin",
            "swipe_start_at", "spock_detected", "was_paused", "volume_states",
            "dwell_ready", "dwell_progress",
        )
        for symbol in dead_state:
            with self.subTest(symbol=symbol):
                self.assertNotRegex(SOURCE, rf"\b{re.escape(symbol)}\b")

    def test_dead_gesture_statuses_are_removed(self):
        self.assertNotIn('gesture_mode == "DRAG"', SOURCE)
        self.assertNotIn('gesture_mode == "CLICK"', SOURCE)
        self.assertNotIn("DRAG LEGACY", SOURCE)

    def test_worker_packet_has_no_unused_input_sequence(self):
        self.assertNotIn("enqueued_at, input_seq = packet", SOURCE)
        self.assertNotIn("gray, ts, now, mp_input_seq)", SOURCE)

    def test_precision_snap_does_not_use_obsolete_dwell_names(self):
        self.assertNotIn("DWELL_RADIUS_PX", SOURCE)
        self.assertNotIn("dwell assistito", SOURCE)


if __name__ == "__main__":
    unittest.main()
