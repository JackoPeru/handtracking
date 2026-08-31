import unittest


class FakeCamera:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeWorker:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.joined = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True


class FakeCursor:
    instances = []

    def __init__(self):
        self.started = False
        self.closed = False
        self.synced = []
        self.__class__.instances.append(self)

    def screen_size(self):
        return 1920, 1080

    def sync(self, active):
        self.synced.append(bool(active))

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class RuntimeSessionTests(unittest.TestCase):
    def setUp(self):
        FakeWorker.instances.clear()
        FakeCursor.instances.clear()

    def test_create_starts_runtime_resources_and_initializes_state(self):
        from handtracking_session import RuntimeSession

        camera = FakeCamera()
        session = RuntimeSession.create(
            camera=camera,
            worker_cls=FakeWorker,
            cursor_cls=FakeCursor,
            landmarker_factory=object(),
            options=object(),
            image_builder=lambda frame: frame,
            get_volume=lambda: 0.37,
            time_fn=lambda: 10.0,
        )

        self.assertTrue(session.worker.started)
        self.assertTrue(session.cursor.started)
        self.assertEqual(session.cursor.synced, [False])
        self.assertEqual((session.screen_w, session.screen_h), (1920, 1080))
        self.assertAlmostEqual(session.volume.level, 0.37)
        self.assertEqual(session.start_time, 10.0)
        self.assertEqual(session.last_hand_seen, 10.0)
        self.assertEqual(session.fps_window_start, 10.0)
        self.assertEqual(session.mp_fps_window_start, 10.0)
        self.assertFalse(session.commands_enabled)
        self.assertEqual(session.gesture_mode, "MOUSE")
        self.assertEqual(session.latest_result_seq, -1)

        session.close()

    def test_close_is_idempotent_and_closes_worker_cursor_and_camera_once(self):
        from handtracking_session import RuntimeSession

        camera = FakeCamera()
        session = RuntimeSession.create(
            camera=camera,
            worker_cls=FakeWorker,
            cursor_cls=FakeCursor,
            landmarker_factory=object(),
            options=object(),
            image_builder=lambda frame: frame,
            get_volume=lambda: 0.5,
            time_fn=lambda: 5.0,
        )
        worker = session.worker
        cursor = session.cursor

        session.close()
        session.close()

        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertTrue(cursor.closed)
        self.assertTrue(camera.closed)

    def test_create_cleans_started_worker_and_camera_if_cursor_initialization_fails(self):
        from handtracking_session import RuntimeSession

        class BrokenCursor:
            def __init__(self):
                raise RuntimeError("cursor init failed")

        camera = FakeCamera()
        with self.assertRaisesRegex(RuntimeError, "cursor init failed"):
            RuntimeSession.create(
                camera=camera,
                worker_cls=FakeWorker,
                cursor_cls=BrokenCursor,
                landmarker_factory=object(),
                options=object(),
                image_builder=lambda frame: frame,
                get_volume=lambda: 0.5,
                time_fn=lambda: 5.0,
            )

        worker = FakeWorker.instances[-1]
        self.assertTrue(worker.started)
        self.assertTrue(worker.stopped)
        self.assertTrue(worker.joined)
        self.assertTrue(camera.closed)


if __name__ == "__main__":
    unittest.main()
