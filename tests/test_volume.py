import unittest

from handtracking_config import (
    VOLUME_CONFIRM_SECONDS,
    VOLUME_POSE_LOSS_GRACE,
)
from handtracking_state import FlowState, ScrollState, VolumeState


class FakeCursor:
    def __init__(self):
        self.synced = []

    def sync(self, active):
        self.synced.append(bool(active))


class VolumeHandlerTests(unittest.TestCase):
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
