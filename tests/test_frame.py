import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from handtracking_session import RuntimeSession


class FakeCursor:
    def __init__(self):
        self._active = False
        self.synced = []
        self._position = (100, 100)

    @property
    def active(self):
        return self._active

    def sync(self, active):
        self._active = bool(active)
        self.synced.append(bool(active))

    def position(self):
        return self._position

    def set_position(self, x, y):
        self._position = (x, y)

    def add_delta(self, dx, dy, *, screen_size=None):
        self._position = (self._position[0] + dx, self._position[1] + dy)


def make_hand():
    return [
        SimpleNamespace(x=0.10 + (index % 5) * 0.01,
                        y=0.20 + (index // 5) * 0.01,
                        z=0.0)
        for index in range(21)
    ]


def make_session():
    return RuntimeSession(
        camera=object(),
        worker=object(),
        cursor=FakeCursor(),
        screen_w=1920,
        screen_h=1080,
        start_time=0.0,
        last_hand_seen=0.0,
        fps_window_start=0.0,
        mp_fps_window_start=0.0,
    )


class FrameProcessorTests(unittest.TestCase):
    def test_packet_accepts_recorded_reanchor_without_running_opencv(self):
        from handtracking_frame import process_mediapipe_packet
        from tests import test_runtime_smoke as helpers
        session = make_session()
        session.commands_enabled = True
        packet = helpers.RuntimeSmokeTests._pinch_packet(1)
        reanchor = mock.Mock(return_value=None)
        with mock.patch("handtracking_flow.cv2.calcOpticalFlowPyrLK",
                        side_effect=AssertionError("replay must not use LK")):
            try:
                result = process_mediapipe_packet(
                    session, packet, gray=object(), now=10.0,
                    camera_target_fps=60, reanchor_cb=reanchor,
                )
            except TypeError as exc:
                self.fail("reanchor replay boundary unavailable: " + str(exc))
        self.assertTrue(result.processed)
        self.assertTrue(session.pointer.pinch_held)
        reanchor.assert_called_once()

    def test_stale_packet_is_ignored_without_rearming_gesture_state(self):
        from handtracking_frame import process_mediapipe_packet

        session = make_session()
        session.latest_result_seq = 7
        previous = object()
        session.latest_result = previous
        session.pointer.pinch_held = False
        stale_result = SimpleNamespace(
            hand_landmarks=[make_hand()], hand_world_landmarks=[], handedness=[],
        )
        packet = (8, stale_result, object(), 1.0, 1.0, 1.0, 1.0)

        result = process_mediapipe_packet(
            session,
            packet,
            gray=object(),
            now=5.0,
            camera_target_fps=60,
            mp_result_stale=True,
        )

        self.assertFalse(result.processed)
        self.assertFalse(result.skip_frame)
        self.assertEqual(session.latest_result_seq, 7)
        self.assertIs(session.latest_result, previous)
        self.assertFalse(session.pointer.pinch_held)

    def test_reanchor_skips_lk_when_no_motion_consumer_is_active(self):
        from handtracking_frame import reanchor_flow

        session = make_session()
        session.commands_enabled = True
        session.flow.prev_gray = object()
        previous_points = object()
        session.flow.points = previous_points
        session.flow.active = True

        with mock.patch(
            "handtracking_flow.cv2.calcOpticalFlowPyrLK",
            side_effect=AssertionError("idle reanchor must not call LK"),
        ):
            corrected = reanchor_flow(
                session,
                make_hand(),
                object(),
                object(),
            )

        self.assertIsNone(corrected)
        self.assertIs(session.flow.points, previous_points)

    def test_reanchor_seeds_identical_frame_without_running_lk(self):
        from handtracking_frame import reanchor_flow

        session = make_session()
        session.commands_enabled = True
        session.pointer.pinch_held = True
        current_gray = object()
        result_gray = current_gray

        with mock.patch(
            "handtracking_flow.cv2.calcOpticalFlowPyrLK",
            side_effect=AssertionError("engagement seed must not call LK"),
        ):
            corrected = reanchor_flow(
                session,
                make_hand(),
                result_gray,
                current_gray,
            )

        self.assertIsNotNone(corrected)
        self.assertIsNotNone(session.flow.points)
        self.assertIs(session.flow.prev_gray, current_gray)

    def test_reanchor_corrects_delayed_points_on_each_motion_engagement(self):
        from handtracking_frame import reanchor_flow

        expected = np.array([[[72, 75]], [[72, 78.6]], [[97.6, 78.6]],
                             [[91.2, 82.2]], [[84.8, 85.8]]], dtype=np.float32)
        for mode in ("pointer", "scroll", "swipe"):
            with self.subTest(mode=mode):
                session = make_session()
                session.commands_enabled = True
                session.last_hand_seen = 10.0
                if mode == "pointer":
                    session.pointer.pinch_held = True
                elif mode == "scroll":
                    session.scroll.active = True
                else:
                    session.swipe.pose_last_seen = 9.95
                old_gray = np.zeros((360, 640), dtype=np.uint8)
                current_gray = np.ones((360, 640), dtype=np.uint8)

                def lk(old, new, points, unused, **kwargs):
                    self.assertIs(old, old_gray)
                    self.assertIs(new, current_gray)
                    return (points + np.array([[[8, 3]]], dtype=np.float32),
                            np.ones((5, 1), dtype=np.uint8), np.zeros((5, 1)))

                with mock.patch("handtracking_flow.cv2.calcOpticalFlowPyrLK", lk):
                    corrected = reanchor_flow(session, make_hand(), old_gray, current_gray)

                self.assertIsNotNone(corrected)
                np.testing.assert_allclose(corrected, expected, atol=0.0001)
                np.testing.assert_allclose(session.flow.points, expected, atol=0.0001)
                self.assertIs(session.flow.prev_gray, current_gray)
                self.assertTrue(session.flow.active)

    def test_failed_reanchor_invalidates_points_before_next_flow(self):
        from handtracking_frame import reanchor_flow

        session = make_session()
        session.commands_enabled = True
        session.pointer.pinch_held = True
        session.flow.points = object()
        session.flow.prev_gray = object()
        session.flow.active = True
        with mock.patch("handtracking_flow.cv2.calcOpticalFlowPyrLK",
                        return_value=(None, None, None)):
            corrected = reanchor_flow(session, make_hand(), object(), object())
        self.assertIsNone(corrected)
        self.assertIsNone(session.flow.points)
        self.assertFalse(session.flow.active)

    def test_duplicate_packet_is_ignored_without_mutating_session(self):
        from handtracking_frame import process_mediapipe_packet

        session = make_session()
        session.latest_result_seq = 7
        previous = object()
        session.latest_result = previous
        packet = (7, object(), object(), 10.0, 11.0, 12.0, 1.0)

        result = process_mediapipe_packet(
            session,
            packet,
            gray=object(),
            now=5.0,
            camera_target_fps=60,
        )

        self.assertFalse(result.processed)
        self.assertFalse(result.skip_frame)
        self.assertIs(session.latest_result, previous)
        self.assertEqual(session.latest_result_seq, 7)

    def test_new_packet_without_hands_updates_result_metrics_and_missing_hand_state(self):
        from handtracking_frame import process_mediapipe_packet

        session = make_session()
        session.last_hand_seen = 0.0
        session.paused_by_fist = True
        session.pointer.pinch_held = True
        result_obj = SimpleNamespace(
            hand_landmarks=[],
            hand_world_landmarks=[],
            handedness=[],
        )
        packet = (1, result_obj, object(), 20.0, 30.0, 40.0, 5.0)

        result = process_mediapipe_packet(
            session,
            packet,
            gray=object(),
            now=1.0,
            camera_target_fps=60,
        )

        self.assertTrue(result.processed)
        self.assertFalse(result.skip_frame)
        self.assertEqual(session.latest_result_seq, 1)
        self.assertIs(session.latest_result, result_obj)
        self.assertEqual(session.mp_infer_ms_ema, 20.0)
        self.assertEqual(session.mp_worker_ms_ema, 30.0)
        self.assertEqual(session.mp_cycle_ms_ema, 40.0)
        self.assertEqual(session.mp_queue_ms_ema, 5.0)
        self.assertFalse(session.paused_by_fist)
        self.assertFalse(session.pointer.pinch_held)
        self.assertEqual(session.fist_states, [])
        self.assertEqual(session.debug_fist_score, 0.0)
        self.assertEqual(session.swipe.debug_score, 0.0)

    def test_volume_hand_switch_returns_skip_frame_before_mode_processing(self):
        import handtracking_frame as frame_processor

        session = make_session()
        session.volume.active = True
        session.mp_control_ref = (0.1, 0.1)
        session.control_handedness = "Left"
        session.last_hand_seen = 0.0
        hand = [SimpleNamespace(x=0.1, y=0.2, z=0.0) for _ in range(21)]
        result_obj = SimpleNamespace(
            hand_landmarks=[hand],
            hand_world_landmarks=[hand],
            handedness=[[SimpleNamespace(category_name="Right")]],
        )
        packet = (1, result_obj, object(), 1.0, 1.0, 16.0, 0.0)
        analysis = SimpleNamespace(
            paused_by_fist=False,
            old_pause=False,
            fist_pending=False,
            control_index=0,
            control_distance=99.0,
            selected_handedness="Right",
            control_hand=hand,
            control_class_hand=hand,
            points=[(0.9, 0.9)],
        )

        with (
            mock.patch.object(frame_processor, "spock_pose_score", return_value=0.0),
            mock.patch.object(frame_processor, "spock_all_fingers_up", return_value=False),
            mock.patch.object(frame_processor, "analyze_hand_frame", return_value=analysis),
            mock.patch.object(frame_processor, "update_hand_mode_metrics") as mode_metrics,
        ):
            result = frame_processor.process_mediapipe_packet(
                session,
                packet,
                gray=object(),
                now=1.0,
                camera_target_fps=60,
            )

        self.assertTrue(result.processed)
        self.assertTrue(result.skip_frame)
        self.assertFalse(session.volume.active)
        self.assertIsNone(session.mp_control_ref)
        self.assertIsNone(session.control_handedness)
        mode_metrics.assert_not_called()


if __name__ == "__main__":
    unittest.main()
