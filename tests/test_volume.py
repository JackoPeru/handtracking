import unittest
from unittest import mock

from handtracking_config import (
    VOLUME_CONFIRM_SECONDS,
    VOLUME_POSE_LOSS_GRACE,
)
from handtracking_session import RuntimeSession
from handtracking_state import FlowState, ScrollState, VolumeState


class FakeCursor:
    def __init__(self):
        self.synced = []

    def sync(self, active):
        self.synced.append(bool(active))


class VolumeHandlerTests(unittest.TestCase):
    def test_failed_volume_write_preserves_level_and_cancels_motion(self):
        from handtracking_volume import update_volume_state

        volume = VolumeState(active=True, last_angle=0.0, level=0.40)
        flow = FlowState()
        flow.time = 10.0
        cursor = FakeCursor()

        result = update_volume_state(
            volume,
            now=10.1,
            dedicated_mode_block=False,
            volume_gesture_now=False,
            volume_candidate_now=False,
            control_hand=object(),
            control_class_hand=object(),
            fist_pending=False,
            debug_volume_score=1.0,
            scroll=ScrollState(),
            cursor=cursor,
            flow=flow,
            get_volume_cb=lambda: 0.40,
            set_volume_cb=lambda level: False,
            angle_fn=lambda hand: 0.1,
            release_pose_fn=lambda hand: False,
            open_hand_fn=lambda hand: False,
            angle_delta_fn=lambda current, previous: current - previous,
        )

        self.assertFalse(result)
        self.assertAlmostEqual(volume.level, 0.40)
        self.assertFalse(volume.active)
        self.assertIsNone(volume.last_angle)
        self.assertIsNone(flow.time)
        self.assertEqual(cursor.synced[-1], True)

    def test_none_volume_write_result_remains_success(self):
        from handtracking_volume import update_volume_state

        volume = VolumeState(active=True, last_angle=0.0, level=0.40)
        flow = FlowState()

        result = update_volume_state(
            volume,
            now=10.1,
            dedicated_mode_block=False,
            volume_gesture_now=False,
            volume_candidate_now=False,
            control_hand=object(),
            control_class_hand=object(),
            fist_pending=False,
            debug_volume_score=1.0,
            scroll=ScrollState(),
            cursor=FakeCursor(),
            flow=flow,
            get_volume_cb=lambda: 0.40,
            set_volume_cb=lambda level: None,
            angle_fn=lambda hand: 0.1,
            release_pose_fn=lambda hand: False,
            open_hand_fn=lambda hand: False,
            angle_delta_fn=lambda current, previous: current - previous,
        )

        self.assertNotEqual(result, False)
        self.assertGreater(volume.level, 0.40)

    def test_modes_report_unavailable_volume_after_failed_write(self):
        from handtracking_modes import process_volume_scroll

        session = RuntimeSession(
            camera=object(), worker=object(), cursor=FakeCursor(),
            screen_w=1920, screen_h=1080, start_time=0.0,
        )
        session.commands_enabled = True
        session.volume.active = True

        with mock.patch("handtracking_modes.update_volume_state", return_value=False):
            process_volume_scroll(
                session,
                control_hand=object(),
                control_class_hand=object(),
                fist_pending=False,
                volume_gesture_now=False,
                volume_candidate_now=False,
                scroll_gesture_now=False,
                now=10.0,
                get_volume_cb=lambda: 0.5,
                set_volume_cb=lambda level: False,
            )

        self.assertEqual(session.gesture_event, "VOLUME NON DISPONIBILE")
    def test_candidate_confirms_volume_lock_and_samples_current_level(self):
        from handtracking_volume import update_volume_state

        volume = VolumeState(candidate_at=10.0 - VOLUME_CONFIRM_SECONDS - 0.01)
        scroll = ScrollState(active=True, residual=12.0)
        flow = FlowState()
        flow.virtual[:] = (4.0, 5.0)
        cursor = FakeCursor()

        update_volume_state(
            volume,
            now=10.0,
            dedicated_mode_block=False,
            volume_gesture_now=True,
            volume_candidate_now=True,
            control_hand=object(),
            control_class_hand=object(),
            fist_pending=False,
            debug_volume_score=1.0,
            scroll=scroll,
            cursor=cursor,
            flow=flow,
            get_volume_cb=lambda: 0.73,
            set_volume_cb=lambda level: None,
            angle_fn=lambda hand: 1.25,
            release_pose_fn=lambda hand: False,
            open_hand_fn=lambda hand: False,
        )

        self.assertTrue(volume.active)
        self.assertAlmostEqual(volume.level, 0.73)
        self.assertEqual(volume.last_angle, 1.25)
        self.assertFalse(scroll.active)
        self.assertEqual(cursor.synced[-1], False)
        self.assertIsNone(flow.time)

    def test_release_pose_freezes_rotation_immediately_before_release_grace(self):
        from handtracking_volume import update_volume_state

        volume = VolumeState(active=True, last_angle=1.0, level=0.4)
        volume.delta_history.extend([0.1, 0.2])
        flow = FlowState()
        cursor = FakeCursor()
        writes = []

        update_volume_state(
            volume,
            now=20.0,
            dedicated_mode_block=False,
            volume_gesture_now=False,
            volume_candidate_now=False,
            control_hand=object(),
            control_class_hand=object(),
            fist_pending=False,
            debug_volume_score=1.0,
            scroll=ScrollState(),
            cursor=cursor,
            flow=flow,
            get_volume_cb=lambda: 0.4,
            set_volume_cb=writes.append,
            angle_fn=lambda hand: 2.0,
            release_pose_fn=lambda hand: True,
            open_hand_fn=lambda hand: False,
        )

        self.assertTrue(volume.active)
        self.assertEqual(volume.release_at, 20.0)
        self.assertIsNone(volume.last_angle)
        self.assertEqual(list(volume.delta_history), [])
        self.assertEqual(writes, [])

    def test_pose_loss_releases_volume_after_grace_and_rearms_cursor(self):
        from handtracking_volume import update_volume_state

        volume = VolumeState(
            active=True,
            pose_lost_at=30.0 - VOLUME_POSE_LOSS_GRACE - 0.01,
            last_angle=1.0,
        )
        flow = FlowState()
        flow.virtual[:] = (5.0, 5.0)
        cursor = FakeCursor()

        update_volume_state(
            volume,
            now=30.0,
            dedicated_mode_block=False,
            volume_gesture_now=False,
            volume_candidate_now=False,
            control_hand=object(),
            control_class_hand=object(),
            fist_pending=False,
            debug_volume_score=0.0,
            scroll=ScrollState(),
            cursor=cursor,
            flow=flow,
            get_volume_cb=lambda: 0.5,
            set_volume_cb=lambda level: None,
            angle_fn=lambda hand: 1.0,
            release_pose_fn=lambda hand: False,
            open_hand_fn=lambda hand: False,
        )

        self.assertFalse(volume.active)
        self.assertEqual(cursor.synced[-1], True)
        self.assertIsNone(flow.time)


if __name__ == "__main__":
    unittest.main()
