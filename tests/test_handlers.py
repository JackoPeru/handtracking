import unittest

from handtracking_state import RadialState, ScrollState, SwipeState, TwoHandState, VolumeState


class FakeCursor:
    def __init__(self):
        self.sync_calls = []

    def sync(self, active):
        self.sync_calls.append(active)


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


if __name__ == "__main__":
    unittest.main()
