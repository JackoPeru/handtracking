import unittest

import numpy as np


class OverlayLayerTests(unittest.TestCase):
    def test_layer_refresh_cadence_and_cached_apply(self):
        from handtracking_display import OverlayLayer

        layer = OverlayLayer(refresh_hz=20.0)
        frame = np.zeros((40, 60, 3), dtype=np.uint8)

        self.assertTrue(layer.should_refresh(frame, 1.0))
        canvas = layer.begin(frame)
        canvas[10:20, 10:20] = (10, 20, 30)
        layer.finish(1.0)

        self.assertFalse(layer.should_refresh(frame, 1.02))
        target = np.zeros_like(frame)
        layer.apply(target)
        self.assertTrue(np.array_equal(target[12, 12], np.array([10, 20, 30])))
        self.assertTrue(layer.should_refresh(frame, 1.051))

    def test_shape_change_forces_refresh(self):
        from handtracking_display import OverlayLayer

        layer = OverlayLayer(refresh_hz=30.0)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        layer.begin(frame)
        layer.finish(0.0)
        resized = np.zeros((21, 20, 3), dtype=np.uint8)
        self.assertTrue(layer.should_refresh(resized, 0.001))

    def test_height_limited_layer_only_copies_top_region(self):
        from handtracking_display import OverlayLayer

        layer = OverlayLayer(refresh_hz=10.0, height=10)
        frame = np.zeros((30, 20, 3), dtype=np.uint8)
        canvas = layer.begin(frame)
        self.assertEqual(canvas.shape, (10, 20, 3))
        canvas[:] = (1, 2, 3)
        layer.finish(0.0)

        target = np.zeros_like(frame)
        layer.apply(target)
        self.assertTrue((target[:10] != 0).any())
        self.assertFalse((target[10:] != 0).any())


if __name__ == "__main__":
    unittest.main()
