import math
import unittest
from types import SimpleNamespace


def point(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def blank_hand():
    return [point() for _ in range(21)]


class GestureModuleTests(unittest.TestCase):
    def test_config_exports_current_camera_and_pointer_thresholds(self):
        import handtracking_config as config

        self.assertEqual((config.CAMERA_W, config.CAMERA_H), (1280, 720))
        self.assertEqual((config.DETECTION_W, config.DETECTION_H), (640, 360))
        self.assertAlmostEqual(config.POINTER_PINCH_ON, 0.46)
        self.assertAlmostEqual(config.POINTER_PINCH_OFF, 0.62)

    def test_normalized_pinch_ratio_uses_palm_height_as_scale(self):
        from handtracking_gestures import normalized_pinch_ratio

        hand = blank_hand()
        hand[0] = point(0.0, 0.0)
        hand[9] = point(0.0, 0.2)
        hand[4] = point(0.1, 0.0)
        hand[8] = point(0.1, 0.1)

        self.assertAlmostEqual(normalized_pinch_ratio(hand, 8), 0.5)

    def test_two_hand_geometry_returns_distance_and_control_points(self):
        from handtracking_gestures import two_hand_geometry

        a = blank_hand()
        b = blank_hand()
        for index in (0, 9, 13):
            a[index] = point(0.2, 0.4)
            b[index] = point(0.6, 0.4)

        distance, point_a, point_b = two_hand_geometry(a, b)

        self.assertAlmostEqual(distance, 0.4)
        self.assertEqual(point_a, (0.2, 0.4))
        self.assertEqual(point_b, (0.6, 0.4))

    def test_wrapped_angle_delta_crosses_pi_without_large_jump(self):
        from handtracking_gestures import wrapped_angle_delta

        delta = wrapped_angle_delta(math.radians(-179), math.radians(179))
        self.assertAlmostEqual(math.degrees(delta), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
