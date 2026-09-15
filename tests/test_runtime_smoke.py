import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


class FakeCapture:
    def __init__(self, frames=4):
        self.frames = frames
        self.released = False
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 60.0)

    def read(self):
        if self.frames <= 0:
            return False, None
        self.frames -= 1
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeWorker:
    def __init__(self):
        self.seq = 0
        self.latest = None
        self.stopped = False
        self.joined = False
        self.input_seq = 0
        self.overwrites = 0
        self.error_count = 0
        self.last_error = ""
        self.last_success_at = time.perf_counter()
        self.last_result_input_at = time.perf_counter()
        self.running = True

    def start(self):
        self.running = True
        return None

    def is_alive(self):
        return True

    def submit(self, frame, gray, timestamp_ms, enqueued_at):
        self.input_seq += 1
        self.seq += 1
        result = SimpleNamespace(
            hand_landmarks=[],
            hand_world_landmarks=[],
            handedness=[],
        )
        self.latest = (self.seq, result, gray, 1.0, 1.0, 16.0, 0.0)

    def snapshot_state(self):
        return {
            "latest": self.latest,
            "seq": self.seq,
            "input_seq": self.input_seq,
            "overwrites": self.overwrites,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "last_result_input_at": self.last_result_input_at,
            "alive": self.is_alive(),
        }

    def stop(self):
        self.stopped = True
        self.running = False

    def join(self, timeout=None):
        self.joined = True


class FakeCursor:
    def __init__(self):
        self.closed = False
        self.started = False
        self._active = False
        self._position = (100, 100)
        self.running = True
        self.last_error = ""

    @property
    def active(self):
        return self._active

    def screen_size(self):
        return 1920, 1080

    def position(self):
        return self._position

    def set_position(self, x, y):
        self._position = (int(round(x)), int(round(y)))

    def sync(self, active):
        self._active = bool(active)

    def add_delta(self, dx, dy, *, screen_size=None):
        self._position = (self._position[0] + dx, self._position[1] + dy)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True
        self._active = False
        self.running = False


class FakeRuntimeCamera:
    def __init__(self, frames=1):
        self.frames = frames
        self.gray = np.zeros((360, 640), dtype=np.uint8)
        self.closed = False
        self.prepare_calls = 0
        self.reported_fps = 60.0
        self.reported_w = 1280
        self.reported_h = 720
        self.codec = "MJPG"
        self.target_fps = 60

    def read_frame(self):
        if self.frames <= 0:
            return None
        self.frames -= 1
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def prepare_detection(self, frame):
        self.prepare_calls += 1
        return frame, self.gray

    def show(self, frame):
        return self.frames > 0

    def close(self):
        self.closed = True


class PacketWorker:
    def __init__(self, packets, *, stale=False):
        self.packets = list(packets)
        self.stale = stale
        self.index = 0
        self.stopped = False
        self.joined = False
        self.input_seq = 0
        self.overwrites = 0
        self.error_count = 0
        self.last_error = ""

    def start(self):
        return None

    def is_alive(self):
        return not self.stopped

    def submit(self, frame, gray, timestamp_ms, enqueued_at):
        self.input_seq += 1

    def snapshot_state(self):
        packet = self.packets[min(self.index, len(self.packets) - 1)]
        self.index += 1
        result_input_at = time.perf_counter() - (1.0 if self.stale else 0.0)
        return {
            "latest": packet,
            "seq": packet[0] if packet is not None else 0,
            "input_seq": self.input_seq,
            "overwrites": self.overwrites,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_success_at": time.perf_counter(),
            "last_result_input_at": result_input_at,
            "alive": self.is_alive(),
        }

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True


class NullHud:
    def should_refresh(self, frame, now):
        return False

    def apply(self, frame):
        return None


class RuntimeSmokeTests(unittest.TestCase):
    @staticmethod
    def _session_create_side_effect(runtime, worker, cursor):
        real_create = runtime.RuntimeSession.create

        def create(*, camera):
            return real_create(
                camera=camera,
                worker_cls=lambda **kwargs: worker,
                cursor_cls=lambda: cursor,
                landmarker_factory=object(),
                options=object(),
                image_builder=lambda frame: frame,
                get_volume=lambda: 0.5,
            )

        return create

    @staticmethod
    def _make_runtime_session(camera, worker, cursor):
        from handtracking_session import RuntimeSession

        now = time.perf_counter()
        session = RuntimeSession(
            camera=camera,
            worker=worker,
            cursor=cursor,
            screen_w=1920,
            screen_h=1080,
            start_time=now - 1.0,
            last_hand_seen=now,
            fps_window_start=now,
            mp_fps_window_start=now,
            camera_target_fps=60,
        )
        session.hud_layer = NullHud()
        session.commands_enabled = True
        return session

    @staticmethod
    def _run_runtime_session(session, *, motion=None, execute_swipe=None,
                             mouse_wheel=None):
        import handtracking_runtime as runtime

        patches = [
            mock.patch.object(runtime, "draw_runtime_overlays"),
            mock.patch.object(runtime, "draw_runtime_hud"),
        ]
        if motion is not None:
            patches.append(
                mock.patch.object(runtime, "measure_optical_flow",
                                  return_value=motion)
            )
        if execute_swipe is not None:
            patches.append(mock.patch.object(runtime, "execute_swipe",
                                              execute_swipe))
        if mouse_wheel is not None:
            patches.append(mock.patch.object(runtime, "mouse_wheel", mouse_wheel))
        try:
            for patcher in patches:
                patcher.start()
            try:
                return runtime._run_impl(session)
            finally:
                for patcher in reversed(patches):
                    patcher.stop()
        finally:
            session.close()

    @staticmethod
    def _packet(seq):
        result = SimpleNamespace(
            hand_landmarks=[],
            hand_world_landmarks=[],
            handedness=[],
        )
        return (seq, result, object(), 1.0, 1.0, 16.0, 0.0)

    def test_runtime_does_not_replay_stale_packet_after_fail_safe(self):
        camera = FakeRuntimeCamera()
        worker = PacketWorker([self._packet(8)], stale=True)
        cursor = FakeCursor()
        session = self._make_runtime_session(camera, worker, cursor)
        session.latest_result_seq = 7
        session.pointer.pinch_held = True

        self._run_runtime_session(session)

        self.assertEqual(session.latest_result_seq, 7)
        self.assertIsNone(session.latest_result)
        self.assertFalse(session.pointer.pinch_held)

    def test_packet_only_preprocessing_is_skipped_when_stale_or_locked(self):
        for stale, commands_enabled in ((True, True), (False, False)):
            with self.subTest(stale=stale, commands_enabled=commands_enabled):
                camera = FakeRuntimeCamera()
                worker = PacketWorker([self._packet(1)], stale=stale)
                session = self._make_runtime_session(camera, worker, FakeCursor())
                session.commands_enabled = commands_enabled
                session.mp_scheduler = SimpleNamespace(should_submit=lambda *a, **k: False)
                self._run_runtime_session(session)
                self.assertEqual(camera.prepare_calls, 0)

    def test_fresh_packet_resets_lost_motion_before_flow_dispatch(self):
        from handtracking_flow import LKMotion

        cases = ("scroll", "swipe", "pointer")
        for mode in cases:
            with self.subTest(mode=mode):
                camera = FakeRuntimeCamera()
                worker = PacketWorker([self._packet(1)])
                cursor = FakeCursor()
                session = self._make_runtime_session(camera, worker, cursor)
                session.last_hand_seen = time.perf_counter() - 1.0
                session.flow.prev_gray = np.zeros((360, 640), dtype=np.uint8)
                session.flow.points = np.zeros((5, 1, 2), dtype=np.float32)
                session.flow.active = True
                session.flow.time = time.perf_counter() - 0.1

                wheel = mock.Mock()
                swipe = mock.Mock(return_value="sent")
                if mode == "scroll":
                    session.scroll.active = True
                    motion = LKMotion(session.flow.points, 0.0, 2.0, 2.0)
                elif mode == "swipe":
                    session.swipe.tracking = True
                    motion = LKMotion(session.flow.points, 6.0, 0.0, 6.0)
                else:
                    session.pointer.pinch_held = True
                    session.pointer.move_active = True
                    motion = LKMotion(session.flow.points, 0.0, 2.0, 2.0)

                self._run_runtime_session(
                    session,
                    motion=motion,
                    execute_swipe=swipe,
                    mouse_wheel=wheel,
                )

                wheel.assert_not_called()
                swipe.assert_not_called()
                self.assertEqual(cursor.position(), (100, 100))

    def test_duplicate_fresh_packet_still_dispatches_camera_rate_flow(self):
        from handtracking_flow import LKMotion

        camera = FakeRuntimeCamera()
        worker = PacketWorker([self._packet(1)])
        cursor = FakeCursor()
        session = self._make_runtime_session(camera, worker, cursor)
        session.latest_result_seq = 1
        session.scroll.active = True
        session.flow.prev_gray = np.zeros((360, 640), dtype=np.uint8)
        session.flow.points = np.zeros((5, 1, 2), dtype=np.float32)
        session.flow.active = True
        wheel = mock.Mock()

        self._run_runtime_session(
            session,
            motion=LKMotion(session.flow.points, 0.0, 2.0, 2.0),
            mouse_wheel=wheel,
        )

        wheel.assert_called_once()

    def test_consecutive_fresh_pointer_packets_do_not_starve_lk_motion(self):
        from handtracking_flow import LKMotion

        camera = FakeRuntimeCamera(frames=3)
        worker = PacketWorker([self._packet(1), self._packet(2), self._packet(3)])
        cursor = FakeCursor()
        session = self._make_runtime_session(camera, worker, cursor)
        session.pointer.pinch_held = True
        session.flow.prev_gray = np.zeros((360, 640), dtype=np.uint8)
        session.flow.points = np.zeros((5, 1, 2), dtype=np.float32)
        session.flow.active = True

        self._run_runtime_session(
            session,
            motion=LKMotion(session.flow.points, 5.0, 0.0, 5.0),
        )

        self.assertNotEqual(cursor.position(), (100, 100))

    def test_runtime_cleans_up_when_cursor_worker_dies(self):
        import handtracking_runtime as runtime

        class DeadCursor(FakeCursor):
            def __init__(self):
                super().__init__()
                self.running = False
                self.last_error = "RuntimeError: cursor failed"

        camera = FakeRuntimeCamera()
        worker = PacketWorker([self._packet(1)])
        cursor = DeadCursor()
        session = self._make_runtime_session(camera, worker, cursor)

        with self.assertRaisesRegex(RuntimeError, "Cursor worker stopped.*cursor failed"):
            self._run_runtime_session(session)

        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertTrue(camera.closed)
        self.assertTrue(cursor.closed)

    def test_no_hand_runtime_starts_and_cleans_up_without_real_devices(self):
        import handtracking_runtime as runtime

        capture = FakeCapture()
        worker = FakeWorker()
        cursor = FakeCursor()

        with (
            mock.patch.object(runtime.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(runtime.cv2, "namedWindow"),
            mock.patch.object(runtime.cv2, "setWindowProperty"),
            mock.patch.object(runtime.cv2, "imshow"),
            mock.patch.object(runtime.cv2, "waitKey", return_value=-1),
            mock.patch.object(runtime.cv2, "destroyAllWindows"),
            mock.patch.object(
                runtime.RuntimeSession,
                "create",
                side_effect=self._session_create_side_effect(runtime, worker, cursor),
            ),
        ):
            runtime.run()

        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertTrue(capture.released)
        self.assertTrue(cursor.started)
        self.assertTrue(cursor.closed)

    def test_runtime_cleans_up_all_resources_when_loop_raises(self):
        import handtracking_runtime as runtime

        capture = FakeCapture(frames=1)
        worker = FakeWorker()
        cursor = FakeCursor()
        destroyed = mock.Mock()

        with (
            mock.patch.object(runtime.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(runtime.cv2, "namedWindow"),
            mock.patch.object(runtime.cv2, "setWindowProperty"),
            mock.patch.object(runtime.cv2, "resize", side_effect=RuntimeError("synthetic crash")),
            mock.patch.object(runtime.cv2, "destroyAllWindows", destroyed),
            mock.patch.object(
                runtime.RuntimeSession,
                "create",
                side_effect=self._session_create_side_effect(runtime, worker, cursor),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
                runtime.run()

        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertTrue(capture.released)
        self.assertTrue(cursor.closed)
        destroyed.assert_called_once()

    def test_runtime_exits_if_mediapipe_worker_dies(self):
        import handtracking_runtime as runtime

        class DeadWorker(FakeWorker):
            def is_alive(self):
                return False

            def snapshot_state(self):
                state = super().snapshot_state()
                state["error_count"] = 1
                state["last_error"] = "RuntimeError: init failed"
                state["last_success_at"] = None
                state["last_result_input_at"] = None
                return state

        capture = FakeCapture(frames=2)
        worker = DeadWorker()
        cursor = FakeCursor()

        with (
            mock.patch.object(runtime.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(runtime.cv2, "namedWindow"),
            mock.patch.object(runtime.cv2, "setWindowProperty"),
            mock.patch.object(runtime.cv2, "imshow"),
            mock.patch.object(runtime.cv2, "waitKey", return_value=-1),
            mock.patch.object(runtime.cv2, "destroyAllWindows"),
            mock.patch.object(
                runtime.RuntimeSession,
                "create",
                side_effect=self._session_create_side_effect(runtime, worker, cursor),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "MediaPipe worker stopped.*init failed"):
                runtime.run()

        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertTrue(capture.released)
        self.assertTrue(cursor.closed)


if __name__ == "__main__":
    unittest.main()
