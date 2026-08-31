import unittest
from collections import deque

from handtracking_state import (
    FlowState,
    PointerState,
    RadialState,
    ScrollState,
    SpockState,
    SwipeState,
    TwoHandState,
    VolumeState,
)


class FakeCursor:
    def __init__(self):
        self.synced = []

    def sync(self, active):
        self.synced.append(bool(active))


class TrackingLifecycleTests(unittest.TestCase):
    def test_stale_fail_safe_resets_gesture_output_and_requires_spock_release(self):
        from handtracking_tracking import apply_stale_fail_safe

        spock = SpockState(latched=True, blocking=True, debug_score=0.9)
        pointer = PointerState(pinch_held=True, move_active=True, last_click_at=3.0)
        volume = VolumeState(active=True, level=0.4)
        scroll = ScrollState(active=True, residual=50.0)
        two_hand = TwoHandState(active=True)
        radial = RadialState(active=True)
        swipe = SwipeState(tracking=True, debug_score=0.8)
        flow = FlowState()
        flow.points = object()
        flow.active = True
        flow.virtual[:] = (3.0, 4.0)
        fist_history = deque([1.0, 1.0], maxlen=5)
        cursor = FakeCursor()

        result = apply_stale_fail_safe(
            spock=spock,
            fist_vote_history=fist_history,
            volume=volume,
            scroll=scroll,
            two_hand=two_hand,
            radial=radial,
            swipe=swipe,
            pointer=pointer,
            flow=flow,
            cursor=cursor,
        )

        self.assertTrue(spock.release_required)
        self.assertTrue(spock.blocking)
        self.assertFalse(pointer.pinch_held)
        self.assertEqual(pointer.last_click_at, 3.0)
        self.assertFalse(volume.active)
        self.assertFalse(scroll.active)
        self.assertFalse(two_hand.active)
        self.assertFalse(radial.active)
        self.assertFalse(swipe.tracking)
        self.assertIsNone(flow.points)
        self.assertFalse(flow.active)
        self.assertEqual(fist_history, deque([], maxlen=5))
        self.assertEqual(cursor.synced[-1], False)
        self.assertFalse(result.paused_by_fist)
        self.assertIsNone(result.latest_result)
        self.assertEqual(result.fist_states, [])
        self.assertIsNone(result.mp_control_ref)
        self.assertIsNone(result.control_handedness)
        self.assertFalse(result.precision_snap_active)

    def test_missing_hands_freezes_pointer_before_full_tracking_reset(self):
        from handtracking_tracking import handle_missing_hands

        pointer = PointerState(pinch_held=True, move_active=True, last_click_at=2.0)
        volume = VolumeState(level=0.5)
        scroll = ScrollState()
        two_hand = TwoHandState()
        radial = RadialState()
        swipe = SwipeState()
        flow = FlowState()
        flow.points = object()
        flow.active = True
        cursor = FakeCursor()
        history = deque([1.0], maxlen=5)

        result = handle_missing_hands(
            now=10.10,
            last_hand_seen=10.0,
            pointer=pointer,
            volume=volume,
            scroll=scroll,
            two_hand=two_hand,
            radial=radial,
            swipe=swipe,
            flow=flow,
            cursor=cursor,
            fist_vote_history=history,
            paused_by_fist=True,
            gesture_input_block_until=12.0,
            mp_control_ref=(0.2, 0.3),
            control_handedness="Left",
        )

        self.assertTrue(result.paused_by_fist)
        self.assertFalse(result.full_reset)
        self.assertTrue(pointer.pinch_held)

        result = handle_missing_hands(
            now=11.0,
            last_hand_seen=10.0,
            pointer=pointer,
            volume=volume,
            scroll=scroll,
            two_hand=two_hand,
            radial=radial,
            swipe=swipe,
            flow=flow,
            cursor=cursor,
            fist_vote_history=history,
            paused_by_fist=True,
            gesture_input_block_until=12.0,
            mp_control_ref=(0.2, 0.3),
            control_handedness="Left",
        )

        self.assertTrue(result.full_reset)
        self.assertFalse(result.paused_by_fist)
        self.assertEqual(result.gesture_input_block_until, 0.0)
        self.assertIsNone(result.mp_control_ref)
        self.assertIsNone(result.control_handedness)
        self.assertFalse(pointer.pinch_held)
        self.assertEqual(pointer.last_click_at, 2.0)
        self.assertEqual(cursor.synced[-1], False)

    def test_expire_lost_flow_drops_points_after_tracking_grace(self):
        from handtracking_tracking import expire_lost_flow

        flow = FlowState(active=False, last_success=1.0)
        flow.points = object()

        expire_lost_flow(flow, now=2.0)

        self.assertIsNone(flow.points)


if __name__ == "__main__":
    unittest.main()
