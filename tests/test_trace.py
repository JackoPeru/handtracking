import json
import copy
import contextlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _settings():
    return SimpleNamespace(
        camera_index=0,
        profile="standard",
        sensitivity=1.0,
        pinch_on=0.46,
        pinch_off=0.62,
    )


class _Cursor:
    def __init__(self):
        self._position = (100.0, 200.0)
        self._active = False
        self.running = True
        self.last_error = ""
        self.closed = False

    @property
    def active(self):
        return self._active

    def position(self):
        return self._position

    def screen_size(self):
        return 800, 600

    def sync(self, active):
        self._active = bool(active)

    def add_delta(self, dx, dy, *, screen_size=None):
        self._position = (self._position[0] + dx, self._position[1] + dy)

    def set_position(self, x, y):
        self._position = (x, y)

    def start(self):
        return None

    def close(self):
        self.closed = True
        self.running = False

    def set_input_time(self, value):
        pass

    def renew_freshness(self, value):
        pass


def record_synthetic_pinch(path, *, max_frames=1800, moving=False, lose_lease=False,
                           reject_lease=False):
    """Synthetic inputs through the actual runtime; no webcam/CV/native IO."""
    from handtracking_runtime import _run_impl
    from handtracking_session import RuntimeSession
    from handtracking_settings import RuntimeSettings
    from handtracking_trace import TraceRecorder
    from handtracking_flow import LKMotion
    from tests import test_runtime_smoke as helpers

    times = [10.02, 10.10, 10.16]
    gray = np.zeros((8, 8), dtype=np.uint8)
    packets = []
    for index in range(3):
        packet = list(helpers.RuntimeSmokeTests._pinch_packet(index + 1))
        packet[2] = gray
        if index and not moving:
            packet[1].hand_landmarks[0][4].x += .30
        packets.append(tuple(packet))

    class Camera:
        index = 0
        def read_frame(self):
            if self.index == len(times):
                return None
            self.index += 1
            return np.zeros((8, 8, 3), dtype=np.uint8)
        def prepare_detection(self, frame):
            return frame, gray
        def close(self):
            pass
        def show(self, frame):
            raise AssertionError("headless recording must not display")

    camera = Camera()
    class Worker:
        def snapshot_state(self):
            index = camera.index - 1
            return dict(latest=packets[index], seq=index + 1, input_seq=index + 1,
                        overwrites=0, error_count=0, alive=True, last_error="",
                        last_success_at=times[index] - .001,
                        last_result_input_at=times[index] - .005)
        def submit(self, *args):
            pass
        def stop(self):
            pass
        def join(self, timeout=None):
            pass

    class Cursor(_Cursor):
        def renew_freshness(self, value):
            return not reject_lease
        @property
        def active(self):
            return False if lose_lease and camera.index == 3 else self._active
        def sync(self, active):
            self._active = bool(active) and not (lose_lease and camera.index == 3)

    settings = RuntimeSettings()
    cursor = Cursor()
    session = RuntimeSession(
        camera=camera, worker=Worker(), cursor=cursor,
        screen_w=800, screen_h=600, start_time=10.0, last_hand_seen=10.0,
        fps_window_start=10.0, mp_fps_window_start=10.0,
        camera_target_fps=60, commands_enabled=True, render_enabled=False,
        settings=settings,
    )
    recorder = TraceRecorder(
        path=path, settings=settings, screen_size=(800, 600), started_at=10.0,
        initial_cursor=cursor.position(), initial_commands_enabled=True,
        source="synthetic", max_frames=max_frames,
    )
    session.trace = recorder
    session.cursor = recorder.wrap_cursor(cursor)
    points = np.zeros((5, 1, 2), dtype=np.float32)
    def measure(*args):
        return LKMotion(points, 5.0, 0.0, 5.0) if moving else None
    try:
        with mock.patch("handtracking_windows.left_click") as click, \
                mock.patch("handtracking_windows.get_user32", side_effect=AssertionError("native IO")), \
                mock.patch("handtracking_windows.AudioUtilities.GetSpeakers", side_effect=AssertionError("audio IO")):
            _run_impl(session, now_fn=lambda: times[camera.index - 1], measure_flow_fn=measure)
            return click.call_count, session.gesture_mode
    finally:
        session.close()


def replay_without_native_io(path):
    from handtracking_trace import replay_trace
    with contextlib.ExitStack() as stack:
        for name in ("get_user32", "left_click", "mouse_wheel", "ctrl_wheel",
                     "execute_swipe", "execute_radial_action", "get_system_volume",
                     "set_system_volume", "AudioUtilities.GetSpeakers"):
            stack.enter_context(mock.patch("handtracking_windows." + name,
                                           side_effect=AssertionError("native IO forbidden")))
        stack.enter_context(mock.patch("handtracking_session.RuntimeSession.create",
                                       side_effect=AssertionError("live session forbidden")))
        stack.enter_context(mock.patch("handtracking_camera.CameraRuntime.open",
                                       side_effect=AssertionError("webcam forbidden")))
        stack.enter_context(mock.patch("handtracking_flow.cv2.calcOpticalFlowPyrLK",
                                       side_effect=AssertionError("CV forbidden")))
        stack.enter_context(mock.patch("handtracking_runtime.cv2.imshow",
                                       side_effect=AssertionError("display forbidden")))
        return replay_trace(path)


class _Session:
    def __init__(self, cursor):
        self.cursor = cursor
        self.gesture_mode = "MOUSE"
        self.flow = SimpleNamespace(
            points=None,
            active=False,
            prev_gray=None,
        )


class TraceRecorderTests(unittest.TestCase):
    def test_record_roundtrip_runs_real_pinch_release_and_one_click(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.assertEqual(record_synthetic_pinch(path), (1, "MOUSE"))
            result = replay_without_native_io(path)
        self.assertTrue(result["verified"], result["mismatches"])
        self.assertEqual(result["source"], "synthetic")
        self.assertEqual(result["modes"], ["PINCH", "PINCH", "MOUSE"])
        self.assertEqual(sum(c["kind"] == "click" for c in result["commands"]), 1)

    def test_synthetic_golden_fixture_has_exactly_one_click(self):
        path = Path(__file__).with_name("fixtures") / "pinch_release.jsonl"
        self.assertTrue(path.is_file(), "synthetic golden fixture is required")
        result = replay_without_native_io(path)
        self.assertTrue(result["verified"], result["mismatches"])
        self.assertEqual(result["source"], "synthetic")
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["modes"], ["PINCH", "PINCH", "MOUSE"])
        self.assertEqual(sum(c["kind"] == "click" for c in result["commands"]), 1)

    def test_replay_rejects_recorded_motion_the_runtime_did_not_admit(self):
        path = Path(__file__).with_name("fixtures") / "pinch_release.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[1]["motion"] = dict(dx=5.0, dy=0.0, magnitude=5.0,
                                     next_points=np.zeros((5, 1, 2)).tolist())
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "trace.jsonl"
            invalid.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            result = replay_without_native_io(invalid)
        self.assertFalse(result["verified"])
        self.assertTrue(any("motion" in item for item in result["mismatches"]))

    def test_strict_loader_rejects_missing_nonfinite_private_and_wrong_types(self):
        from handtracking_trace import TraceFormatError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            record_synthetic_pinch(path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            def mutate(target, key, value):
                data = copy.deepcopy(records)
                node = data
                for part in target:
                    node = node[part]
                if value is ...:
                    del node[key]
                else:
                    node[key] = value
                return data
            cases = [
                mutate([0], "source", []),
                mutate([0], "schemaVersion", True),
                mutate([0, "settings"], "profile", "unknown"),
                mutate([0, "settings"], "sensitivity", ...),
                mutate([1], "index", False),
                mutate([1], "mode", []),
                mutate([1], "video", "private payload"),
                mutate([1, "worker"], "seq", ...),
                mutate([1, "worker"], "last_result_input_at", float("nan")),
                mutate([1, "worker"], "last_result_input_at", 999),
                mutate([1, "worker", "latest", "result"], "handedness", [["secret title"]]),
                mutate([3], "commands", [{"kind": []}]),
                mutate([3], "commands", [{"kind": "move", "dy": 1}]),
                mutate([3], "commands", [{"kind": "radial", "action": []}]),
                mutate([1, "cursor"], "active_reads", [1]),
                mutate([1, "cursor"], "reads", [[10**400, 0]]),
            ]
            for points in (5, "1", [True], np.zeros((5, 2)).tolist(),
                           [[[True, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]):
                cases.append(mutate([1], "motion", dict(dx=5, dy=0, magnitude=5,
                                                        next_points=points)))
            for index, data in enumerate(cases):
                path.write_text("\n".join(json.dumps(r) for r in data) + "\n", encoding="utf-8")
                with self.subTest(case=index), self.assertRaises(TraceFormatError):
                    replay_without_native_io(path)
            for text in ('{"type":"header","type":"frame"}', "[" * 30000 + "]" * 30000):
                path.write_text(text + "\n", encoding="utf-8")
                with self.assertRaises(TraceFormatError):
                    replay_without_native_io(path)

    def test_live_points_reject_scalar_string_bool_and_wrong_shape(self):
        from handtracking_trace import _points_payload, TraceValidationError
        for points in (5, "1", [True], np.zeros((5, 2)),
                       np.ones((5, 1, 2), dtype=bool),
                       [[[True, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]):
            with self.subTest(points=repr(points)), self.assertRaises(TraceValidationError):
                _points_payload(points)

    def test_loader_checks_byte_line_and_frame_limits_before_replay(self):
        from handtracking_trace import TraceFormatError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            record_synthetic_pinch(path)
            for name, bound in (("MAX_TRACE_BYTES", 10), ("MAX_TRACE_LINE_BYTES", 10),
                                ("MAX_TRACE_FRAMES", 1)):
                with self.subTest(bound=name), mock.patch("handtracking_trace." + name, bound), \
                        self.assertRaises(TraceFormatError):
                    replay_without_native_io(path)

    def test_record_frame_limit_does_not_disable_real_gesture_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.assertEqual(record_synthetic_pinch(path, max_frames=1), (1, "MOUSE"))
            self.assertEqual(len(path.read_text().splitlines()), 2)
            result = replay_without_native_io(path)
        self.assertTrue(result["verified"], result["mismatches"])
        self.assertEqual(result["frames"], 1)

    def test_record_creates_explicit_parent_and_byte_limit_only_stops_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces" / "trace.jsonl"
            with mock.patch("handtracking_trace.MAX_TRACE_BYTES", 500):
                self.assertEqual(record_synthetic_pinch(path), (1, "MOUSE"))
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_replay_captures_external_lease_loss_during_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            record_synthetic_pinch(path, moving=True, lose_lease=True)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertIn(False, records[-1]["cursor"]["active_reads"])
            result = replay_without_native_io(path)
            self.assertTrue(result["verified"], result["mismatches"])
            records[-1]["cursor"]["active_reads"] = []
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            result = replay_without_native_io(path)
        self.assertFalse(result["verified"])
        self.assertTrue(any("active" in item for item in result["mismatches"]))

    def test_replay_captures_rejected_lease_after_frame_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.assertEqual(record_synthetic_pinch(path, reject_lease=True)[0], 0)
            result = replay_without_native_io(path)
        self.assertTrue(result["verified"], result["mismatches"])
        self.assertEqual(result["command_count"], 0)

    def test_recorder_reuses_strict_runtime_settings(self):
        from handtracking_trace import TraceRecorder, TraceValidationError
        with tempfile.TemporaryDirectory() as directory:
            invalid = _settings()
            invalid.profile = "secret-title"
            with self.assertRaises(TraceValidationError):
                TraceRecorder(path=Path(directory) / "trace.jsonl", settings=invalid,
                              screen_size=(800, 600), started_at=10,
                              initial_cursor=(0, 0))

    def test_completion_observation_may_be_newer_than_camera_frame(self):
        from handtracking_trace import TraceRecorder
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            recorder = TraceRecorder(path=path, settings=_settings(), screen_size=(800, 600),
                                     started_at=10.0, initial_cursor=(0, 0),
                                     initial_commands_enabled=True)
            try:
                recorder.start_frame(10.1, dict(latest=None, last_success_at=10.1001,
                                               last_result_input_at=10.09), (0, 0))
                recorder.wrap_cursor(_Cursor()).renew_freshness(10.09)
                recorder.finish_frame(_Session(_Cursor()))
            finally:
                recorder.close()
            result = replay_without_native_io(path)
        self.assertTrue(result["verified"], result["mismatches"])

    def test_header_and_completed_frame_are_written_with_cursor_commands(self):
        from handtracking_trace import TraceRecorder

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            recorder = TraceRecorder(
                path=path,
                settings=_settings(),
                screen_size=(800, 600),
                started_at=10.0,
                initial_cursor=(100, 200),
            )
            cursor = _Cursor()
            session = _Session(cursor)
            session.cursor = recorder.wrap_cursor(cursor)
            recorder.start_frame(10.1, {"latest": None}, session.cursor.position())
            session.cursor.add_delta(2.5, -1.0, screen_size=(800, 600))
            session.cursor.set_position(11, 12)
            recorder.finish_frame(session)
            recorder.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(records[0]["schemaVersion"], 1)
        self.assertEqual(records[0]["source"], "recorded")
        self.assertEqual(records[1]["type"], "frame")
        self.assertEqual(
            [command["kind"] for command in records[1]["commands"]],
            ["move", "set_position"],
        )
        self.assertEqual(records[1]["cursor"]["reads"], [])

    def test_external_commands_and_volume_observations_are_recorded(self):
        from handtracking_trace import TraceRecorder

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            recorder = TraceRecorder(
                path=path,
                settings=_settings(),
                screen_size=(800, 600),
                started_at=10.0,
                initial_cursor=(100, 200),
            )
            session = _Session(_Cursor())
            recorder.start_frame(10.1, {"latest": None}, (100, 200))
            with mock.patch("handtracking_windows.execute_swipe", return_value="ignored"), \
                 mock.patch("handtracking_windows.mouse_wheel"), \
                 mock.patch("handtracking_windows.get_system_volume", return_value=0.4), \
                 mock.patch("handtracking_windows.set_system_volume", return_value=True):
                self.assertEqual(recorder.execute_swipe("LEFT"), "ignored")
                recorder.mouse_wheel(120)
                self.assertEqual(recorder.get_system_volume(), 0.4)
                self.assertTrue(recorder.set_system_volume(0.8))
            session.gesture_mode = "SCROLL"
            recorder.finish_frame(session)
            recorder.close()
            frame = json.loads(path.read_text().splitlines()[1])

        self.assertEqual(
            [command["kind"] for command in frame["commands"]],
            ["swipe", "wheel", "volume"],
        )
        self.assertEqual(frame["volume_reads"], [0.4])
        self.assertEqual(frame["volume_sets"], [{"value": 0.8, "success": True}])

    def test_rejects_non_monotonic_and_nonfinite_frames(self):
        from handtracking_trace import TraceRecorder, TraceValidationError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            recorder = TraceRecorder(
                path=path,
                settings=_settings(),
                screen_size=(800, 600),
                started_at=10.0,
                initial_cursor=(100, 200),
            )
            with self.assertRaises(TraceValidationError):
                recorder.start_frame(9.0, {"latest": None}, (100, 200))
            recorder.start_frame(10.1, {"latest": None}, (100, 200))
            with self.assertRaises(TraceValidationError):
                recorder.set_motion(
                    SimpleNamespace(dx=math.inf, dy=0.0, magnitude=0.0,
                                    next_points=np.zeros((5, 1, 2)))
                )
            recorder.close()

    def test_open_x_prevents_overwrite_and_close_is_idempotent(self):
        from handtracking_trace import TraceRecorder

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                TraceRecorder(
                    path=path,
                    settings=_settings(),
                    screen_size=(800, 600),
                    started_at=10.0,
                    initial_cursor=(100, 200),
                )

            recorder = TraceRecorder(
                path=Path(directory) / "other.jsonl",
                settings=_settings(),
                screen_size=(800, 600),
                started_at=10.0,
                initial_cursor=(100, 200),
            )
            recorder.close()
            recorder.close()

    def test_trace_loader_rejects_unknown_keys_and_private_payloads(self):
        from handtracking_trace import TraceFormatError, replay_trace

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps({
                    "schemaVersion": 1,
                    "source": "synthetic",
                    "settings": {},
                    "screen": {"width": 800, "height": 600},
                    "started_at": 1.0,
                    "initial_cursor": [0, 0],
                    "initial_volume": 0.5,
                    "initial_commands_enabled": False,
                    "camera_target_fps": 60,
                    "unknown": True,
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TraceFormatError):
                replay_trace(path)


if __name__ == "__main__":
    unittest.main()
