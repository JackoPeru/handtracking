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

    def start(self):
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

    def snapshot(self):
        return self.latest

    def stats(self):
        return {
            "seq": self.seq,
            "input_seq": self.input_seq,
            "overwrites": 0,
            "error_count": 0,
            "last_error": "",
            "last_success_at": time.perf_counter(),
            "last_result_input_at": time.perf_counter(),
        }

    def snapshot_state(self):
        state = self.stats()
        state["latest"] = self.latest
        state["alive"] = self.is_alive()
        return state

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True


class FakeCursor:
    def __init__(self):
        self.closed = False
        self.started = False
        self._active = False
        self._position = (100, 100)

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

            def stats(self):
                state = super().stats()
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
