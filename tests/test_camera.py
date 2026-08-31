import unittest
from unittest import mock

import numpy as np


class FakeCapture:
    def __init__(self, *, opened=True, frame=None):
        self.opened = opened
        self.frame = frame
        self.released = False
        self.set_calls = []
        self.props = {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        if self.frame is None:
            return False, None
        frame, self.frame = self.frame, None
        return True, frame

    def release(self):
        self.released = True


class CameraRuntimeTests(unittest.TestCase):
    def test_open_falls_back_from_msmf_and_applies_current_capture_contract(self):
        import handtracking_camera as camera

        first = FakeCapture(opened=False)
        second = FakeCapture(opened=True)
        second.props = {
            camera.cv2.CAP_PROP_FPS: 60.0,
            camera.cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            camera.cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            camera.cv2.CAP_PROP_FOURCC: camera.cv2.VideoWriter_fourcc(*"MJPG"),
        }

        fake_cv2 = mock.Mock(wraps=camera.cv2)
        fake_cv2.VideoCapture.side_effect = [first, second]
        fake_cv2.namedWindow = mock.Mock()
        fake_cv2.setWindowProperty = mock.Mock()

        runtime = camera.CameraRuntime.open(cv2_module=fake_cv2)

        self.assertTrue(first.released)
        self.assertIs(runtime.capture, second)
        fake_cv2.VideoCapture.assert_has_calls([
            mock.call(0, fake_cv2.CAP_MSMF),
            mock.call(0),
        ])
        self.assertIn(
            (fake_cv2.CAP_PROP_FOURCC, fake_cv2.VideoWriter_fourcc(*"MJPG")),
            second.set_calls,
        )
        self.assertIn((fake_cv2.CAP_PROP_FRAME_WIDTH, camera.CAMERA_W), second.set_calls)
        self.assertIn((fake_cv2.CAP_PROP_FRAME_HEIGHT, camera.CAMERA_H), second.set_calls)
        self.assertIn((fake_cv2.CAP_PROP_FPS, camera.TARGET_FPS), second.set_calls)
        self.assertIn((fake_cv2.CAP_PROP_BUFFERSIZE, 1), second.set_calls)
        self.assertEqual(runtime.reported_fps, 60.0)
        self.assertEqual(runtime.reported_w, 1280)
        self.assertEqual(runtime.reported_h, 720)

    def test_read_prepared_flips_resizes_and_builds_grayscale_detection_frame(self):
        import handtracking_camera as camera

        source = np.zeros((720, 1280, 3), dtype=np.uint8)
        source[:, :10] = 255
        capture = FakeCapture(opened=True, frame=source)
        runtime = camera.CameraRuntime(
            capture=capture,
            reported_fps=60.0,
            reported_w=1280,
            reported_h=720,
            codec="MJPG",
            target_fps=60,
            cv2_module=camera.cv2,
        )

        prepared = runtime.read_prepared()

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.frame.shape, (720, 1280, 3))
        self.assertEqual(prepared.detect_frame.shape[:2], (camera.DETECTION_H, camera.DETECTION_W))
        self.assertEqual(prepared.gray.shape, (camera.DETECTION_H, camera.DETECTION_W))
        self.assertGreater(prepared.frame[:, -10:].mean(), 200.0)

    def test_show_returns_false_on_escape(self):
        import handtracking_camera as camera

        fake_cv2 = mock.Mock(wraps=camera.cv2)
        fake_cv2.imshow = mock.Mock()
        fake_cv2.waitKey.return_value = 27
        runtime = camera.CameraRuntime(
            capture=FakeCapture(),
            reported_fps=60.0,
            reported_w=1280,
            reported_h=720,
            codec="MJPG",
            target_fps=60,
            cv2_module=fake_cv2,
        )

        self.assertFalse(runtime.show(np.zeros((10, 10, 3), dtype=np.uint8)))
        fake_cv2.imshow.assert_called_once()


if __name__ == "__main__":
    unittest.main()
