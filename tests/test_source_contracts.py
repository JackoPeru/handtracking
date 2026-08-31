import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "handtracking_runtime.py").read_text(encoding="utf-8")
WORKER_SOURCE = (ROOT / "handtracking_mediapipe.py").read_text(encoding="utf-8")
HUD_SOURCE = (
    (ROOT / "handtracking_hud.py").read_text(encoding="utf-8")
    if (ROOT / "handtracking_hud.py").exists() else ""
)


class SourceContractTests(unittest.TestCase):
    def test_runtime_imports_extracted_config_and_gesture_classifiers(self):
        self.assertIn("from handtracking_config import", SOURCE)
        self.assertIn("from handtracking_gestures import", SOURCE)
        self.assertNotIn("CAMERA_W, CAMERA_H =", SOURCE)
        for function_name in (
            "dist", "dist3", "joint_angle3", "control_point",
            "normalized_pinch_ratio", "is_fist", "is_strong_fist",
            "is_scroll_gesture", "swipe_pose_metrics", "spock_pose_score",
            "two_hand_geometry", "radial_direction",
        ):
            with self.subTest(function_name=function_name):
                self.assertNotIn(f"def {function_name}(", SOURCE)

    def test_runtime_uses_central_engine_for_mode_priority(self):
        self.assertIn("from handtracking_engine import resolve_runtime_mode", SOURCE)
        gestures_source = (ROOT / "handtracking_gestures.py").read_text(encoding="utf-8")
        self.assertNotIn("def resolve_gesture_mode(", gestures_source)

    def test_runtime_uses_extracted_windows_and_render_modules(self):
        self.assertIn("from handtracking_windows import", SOURCE)
        self.assertIn("from handtracking_render import", SOURCE)
        self.assertNotIn("ctypes.windll.user32", SOURCE)
        self.assertNotIn("AudioUtilities.GetSpeakers", SOURCE)
        for function_name in (
            "mouse_down", "mouse_up", "mouse_wheel", "key_down", "key_up",
            "tap_combo", "ctrl_wheel", "foreground_window_title",
            "execute_swipe", "execute_radial_action", "get_system_volume",
            "set_system_volume", "left_click", "cursor_worker",
            "draw_hand", "draw_radial_menu", "draw_two_hand_transform",
        ):
            with self.subTest(function_name=function_name):
                self.assertNotIn(f"def {function_name}(", SOURCE)

    def test_runtime_uses_phase_two_processing_modules(self):
        for import_text in (
            "from handtracking_flow import",
            "from handtracking_handlers import",
            "from handtracking_hud import",
            "from handtracking_processing import",
        ):
            with self.subTest(import_text=import_text):
                self.assertIn(import_text, SOURCE)
        self.assertNotIn("calcOpticalFlowPyrLK", SOURCE)
        self.assertNotIn("cv2.putText", SOURCE)

    def test_runtime_orchestrator_stays_below_phase_two_size_budget(self):
        tree = ast.parse(SOURCE)
        run_impl = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_impl"
        )
        self.assertLess(run_impl.end_lineno - run_impl.lineno + 1, 800)

    def test_media_pipe_queue_does_not_copy_fresh_frames(self):
        self.assertNotIn(
            "mp_pending = (detect_frame.copy(), gray.copy()",
            SOURCE,
        )

    def test_camera_requests_mjpeg(self):
        self.assertIn("CAP_PROP_FOURCC", SOURCE)
        self.assertIn("MJPG", SOURCE)
        self.assertNotIn("cap.set(cv2.CAP_PROP_FPS, FALLBACK_FPS)", SOURCE)

    def test_camera_failure_is_explicit(self):
        self.assertIn('raise RuntimeError("Impossibile aprire la webcam")', SOURCE)

    def test_camera_and_mediapipe_diagnostics_are_visible(self):
        self.assertIn("camera_codec", SOURCE)
        self.assertIn("mp_error_count", SOURCE)
        self.assertIn("mp_last_error", SOURCE)
        self.assertIn("MP ERR", HUD_SOURCE)

    def test_mediapipe_errors_are_observable(self):
        self.assertRegex(WORKER_SOURCE, r"except Exception as \w+")
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
                self.assertNotRegex(SOURCE, rf"(?<!\.)\b{re.escape(symbol)}\b")

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

    def test_model_path_is_resolved_from_runtime_file(self):
        self.assertIn("Path(__file__)", SOURCE)
        self.assertNotIn('model_asset_path="hand_landmarker.task"', SOURCE)

    def test_runtime_has_stale_mediapipe_fail_safe(self):
        self.assertIn("MP_RESULT_STALE_SECONDS", SOURCE)
        self.assertIn("tracking_result_is_stale", SOURCE)

    def test_landmarker_is_owned_by_worker_module(self):
        self.assertTrue((ROOT / "handtracking_mediapipe.py").exists())
        self.assertNotIn("def mp_worker", SOURCE)
        self.assertNotIn("with HandLandmarker.create_from_options", SOURCE)

    def test_two_hand_mode_does_not_advertise_unimplemented_rotation(self):
        self.assertNotIn("TWO_HAND_ROTATE_LEFT_VK", SOURCE)
        self.assertNotIn("TWO_HAND_ROTATE_RIGHT_VK", SOURCE)
        self.assertNotIn("ROT {", SOURCE)


if __name__ == "__main__":
    unittest.main()
