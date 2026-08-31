import unittest

from handtracking_state import RadialState, ScrollState, SwipeState, TwoHandState, VolumeState


class FakeCursor:
    def __init__(self):
        self.sync_calls = []
        self._position = (320, 180)
        self.set_calls = []

    def sync(self, active):
        self.sync_calls.append(active)

    def position(self):
        return self._position

    def set_position(self, x, y):
        self._position = (x, y)
        self.set_calls.append((x, y))


class FakeFlow:
    def __init__(self):
        self.clear_count = 0

    def clear_motion(self):
        self.clear_count += 1


class HandlerTests(unittest.TestCase):
    def test_two_hand_release_resets_state_and_rearms_cursor(self):
        from handtracking_config import TWO_HAND_RELEASE_GRACE
        from handtracking_handlers import update_two_hand_state

        state = TwoHandState(active=True, release_at=None)
        cursor = FakeCursor()
        flow = FakeFlow()
        radial = RadialState()
        swipe = SwipeState()
        scroll = ScrollState()
        volume = VolumeState()

        first = update_two_hand_state(
            state,
            held=False,
            pair_geometry=None,
            now=1.0,
            input_block_until=0.0,
            radial=radial,
            swipe=swipe,
            scroll=scroll,
            volume=volume,
            cursor=cursor,
            flow=flow,
            ctrl_wheel_cb=lambda delta: None,
        )
        self.assertTrue(state.active)
        self.assertEqual(state.release_at, 1.0)

        second = update_two_hand_state(
            state,
            held=False,
            pair_geometry=None,
            now=1.0 + TWO_HAND_RELEASE_GRACE + 0.01,
            input_block_until=first,
            radial=radial,
            swipe=swipe,
            scroll=scroll,
            volume=volume,
            cursor=cursor,
            flow=flow,
            ctrl_wheel_cb=lambda delta: None,
        )
        self.assertFalse(state.active)
        self.assertGreater(second, first)
        self.assertEqual(cursor.sync_calls[-1], True)
        self.assertGreater(flow.clear_count, 0)

    def test_radial_confirmed_pinch_executes_selected_action(self):
        from handtracking_config import RADIAL_PINCH_CONFIRM
        from handtracking_handlers import update_radial_state

        radial = RadialState(
            active=True,
            center=(0.5, 0.5),
            selected="LEFT",
            selection_candidate="LEFT",
            selection_since=0.0,
        )
        cursor = FakeCursor()
        flow = FakeFlow()
        swipe = SwipeState()
        scroll = ScrollState()
        events = []

        first = update_radial_state(
            radial,
            now=2.0,
            priority_block=False,
            control_hand=object(),
            control_class_hand=object(),
            current_anchor=(0.5, 0.5),
            volume_candidate_now=False,
            volume_candidate_at=None,
            input_block_until=0.0,
            scroll=scroll,
            swipe=swipe,
            cursor=cursor,
            flow=flow,
            direction_fn=lambda hand, center: "LEFT",
            pinch_ratio_fn=lambda hand, finger: 0.0,
            open_pose_fn=lambda hand: True,
            execute_action_cb=lambda selected: events.append(selected) or "BACK",
        )
        self.assertIsNone(first.event)

        second = update_radial_state(
            radial,
            now=2.0 + RADIAL_PINCH_CONFIRM + 0.01,
            priority_block=False,
            control_hand=object(),
            control_class_hand=object(),
            current_anchor=(0.5, 0.5),
            volume_candidate_now=False,
            volume_candidate_at=None,
            input_block_until=first.input_block_until,
            scroll=scroll,
            swipe=swipe,
            cursor=cursor,
            flow=flow,
            direction_fn=lambda hand, center: "LEFT",
            pinch_ratio_fn=lambda hand, finger: 0.0,
            open_pose_fn=lambda hand: True,
            execute_action_cb=lambda selected: events.append(selected) or "BACK",
        )
        self.assertEqual(events, ["LEFT"])
        self.assertEqual(second.event, "RADIAL: BACK")
        self.assertFalse(radial.active)
        self.assertEqual(cursor.sync_calls[-1], True)

    def test_pointer_quick_release_emits_click_and_preserves_double_click_clock(self):
        from handtracking_config import (
            POINTER_CLICK_MAX_SECONDS,
            POINTER_PINCH_OFF,
            POINTER_RELEASE_GRACE,
        )
        from handtracking_handlers import update_pointer_state
        from handtracking_state import FlowState, PointerState, SwipeState

        now = 2.0 + POINTER_RELEASE_GRACE + 0.01
        pointer = PointerState(
            pinch_held=True,
            pinch_started_at=now - POINTER_CLICK_MAX_SECONDS * 0.5,
            release_at=2.0,
            cursor_origin=(100, 80),
            flow_travel=0.0,
        )
        cursor = FakeCursor()
        flow = FlowState()
        swipe = SwipeState()
        clicks = []

        result = update_pointer_state(
            pointer,
            control_hand=object(),
            now=now,
            commands_enabled=True,
            spock_blocking=False,
            hand_count=1,
            paused=False,
            volume_active=False,
            two_hand_active=False,
            two_hand_candidate=False,
            radial_active=False,
            scroll_active=False,
            swipe_tracking=False,
            input_blocked=False,
            volume_candidate=False,
            cursor=cursor,
            flow=flow,
            swipe=swipe,
            precision_snap_active=True,
            snap_anchor=(1.0, 1.0),
            snap_started_at=1.0,
            left_click_cb=lambda: clicks.append(True),
            ratio_fn=lambda hand, finger: POINTER_PINCH_OFF + 1.0,
            fingers_valid_fn=lambda hand: True,
            pose_fn=lambda hand, limit: False,
        )

        self.assertEqual(clicks, [True])
        self.assertEqual(result.event, "PINCH RAPIDO: CLICK")
        self.assertFalse(pointer.pinch_held)
        self.assertIsNotNone(pointer.last_click_at)
        self.assertFalse(result.precision_snap_active)
        self.assertEqual(cursor.set_calls[-1], (100, 80))

    def test_pointer_pose_arms_pinch_and_records_cursor_origin(self):
        from handtracking_handlers import update_pointer_state
        from handtracking_state import FlowState, PointerState, SwipeState

        pointer = PointerState()
        cursor = FakeCursor()
        flow = FlowState()
        swipe = SwipeState()

        update_pointer_state(
            pointer,
            control_hand=object(),
            now=5.0,
            commands_enabled=True,
            spock_blocking=False,
            hand_count=1,
            paused=False,
            volume_active=False,
            two_hand_active=False,
            two_hand_candidate=False,
            radial_active=False,
            scroll_active=False,
            swipe_tracking=False,
            input_blocked=False,
            volume_candidate=False,
            cursor=cursor,
            flow=flow,
            swipe=swipe,
            precision_snap_active=False,
            snap_anchor=None,
            snap_started_at=None,
            left_click_cb=lambda: None,
            ratio_fn=lambda hand, finger: 0.0,
            fingers_valid_fn=lambda hand: True,
            pose_fn=lambda hand, limit: True,
        )

        self.assertTrue(pointer.pinch_held)
        self.assertEqual(pointer.cursor_origin, cursor.position())


if __name__ == "__main__":
    unittest.main()
