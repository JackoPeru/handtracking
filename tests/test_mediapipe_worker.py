import threading
import time
import unittest


class FakeLandmarker:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.closed = False
        self.closed_while_detecting = False
        self.detecting = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.detecting:
            self.closed_while_detecting = True
        self.closed = True

    def detect_for_video(self, image, ts):
        self.detecting = True
        try:
            time.sleep(self.delay)
            return {"ts": ts}
        finally:
            self.detecting = False


class FakeFactory:
    def __init__(self, landmarker):
        self.landmarker = landmarker

    def create_from_options(self, options):
        return self.landmarker


class MediaPipeWorkerTests(unittest.TestCase):
    def test_worker_owns_landmarker_until_inference_finishes(self):
        from handtracking_mediapipe import MediaPipeWorker

        landmarker = FakeLandmarker(delay=0.15)
        worker = MediaPipeWorker(
            factory=FakeFactory(landmarker),
            options=object(),
            image_builder=lambda frame: frame,
        )
        worker.start()
        worker.submit("frame", "gray", 1, time.perf_counter())
        time.sleep(0.02)
        worker.stop()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertTrue(landmarker.closed)
        self.assertFalse(landmarker.closed_while_detecting)

    def test_worker_tracks_age_of_frame_that_produced_latest_result(self):
        from handtracking_mediapipe import MediaPipeWorker

        worker = MediaPipeWorker(
            factory=FakeFactory(FakeLandmarker()),
            options=object(),
            image_builder=lambda frame: frame,
        )
        enqueued_at = time.perf_counter() - 0.25
        worker.start()
        worker.submit("frame", "gray", 1, enqueued_at)
        deadline = time.time() + 1.0
        while worker.stats()["seq"] == 0 and time.time() < deadline:
            time.sleep(0.01)
        stats = worker.stats()
        worker.stop()
        worker.join(timeout=1.0)

        self.assertAlmostEqual(stats["last_result_input_at"], enqueued_at, places=5)

    def test_worker_reports_errors_without_dying(self):
        from handtracking_mediapipe import MediaPipeWorker

        class RaisingLandmarker(FakeLandmarker):
            def detect_for_video(self, image, ts):
                raise RuntimeError("boom")

        worker = MediaPipeWorker(
            factory=FakeFactory(RaisingLandmarker()),
            options=object(),
            image_builder=lambda frame: frame,
        )
        worker.start()
        worker.submit("frame", "gray", 1, time.perf_counter())
        deadline = time.time() + 1.0
        while worker.error_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        worker.stop()
        worker.join(timeout=1.0)

        self.assertGreaterEqual(worker.error_count, 1)
        self.assertIn("RuntimeError", worker.last_error)


if __name__ == "__main__":
    unittest.main()
