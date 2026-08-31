import unittest

import numpy as np


class HudTests(unittest.TestCase):
    def test_status_text_covers_primary_runtime_modes(self):
        from handtracking_hud import build_status_text

        self.assertIn("SPOCK: 50%", build_status_text(
            "SPOCK", spock_progress=0.5, volume_level=0.0, radial_selected=None
        ))
        self.assertIn("VOLUME: 73%", build_status_text(
            "VOLUME", spock_progress=0.0, volume_level=0.73, radial_selected=None
        ))
        self.assertIn("LEFT", build_status_text(
            "RADIAL", spock_progress=0.0, volume_level=0.0, radial_selected="LEFT"
        ))
        self.assertIn("CURSORE FERMO", build_status_text(
            "MOUSE", spock_progress=0.0, volume_level=0.0, radial_selected=None
        ))

    def test_draw_runtime_hud_writes_pixels_without_mutating_metrics(self):
        from handtracking_hud import draw_runtime_hud

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        metrics = dict(
            actual_fps=60.0,
            actual_mp_fps=20.0,
            mp_infer_ms=30.0,
            mp_worker_ms=35.0,
            mp_cycle_ms=50.0,
            mp_queue_ms=2.0,
            mp_overwrites=1,
            mp_input_seq=10,
            camera_codec="MJPG",
            reported_w=640,
            reported_h=360,
            reported_fps=60.0,
            camera_target_fps=60,
            debug_fist_score=0.1,
            debug_volume_score=0.2,
            debug_grip_gap=0.3,
            debug_fist_folded=2,
            debug_fist_tightness=1.1,
            debug_strong_fist=False,
            spock_debug_score=0.4,
            spock_debug_stable=0.5,
            swipe_debug_score=0.6,
            swipe_debug_stable=0.7,
            swipe_debug_gap=0.8,
            swipe_debug_extended=4,
            mp_error_count=0,
            mp_last_error="",
        )
        before = metrics.copy()

        draw_runtime_hud(
            frame,
            gesture_mode="MOUSE",
            gesture_event="",
            gesture_event_until=0.0,
            now=1.0,
            flow_active=True,
            commands_enabled=True,
            spock_blocking=False,
            spock_latched=False,
            spock_progress=0.0,
            volume_level=0.5,
            radial_selected=None,
            **metrics,
        )

        self.assertGreater(int(frame.sum()), 0)
        self.assertEqual(metrics, before)


if __name__ == "__main__":
    unittest.main()
