import unittest

from handtracking_config import SCROLL_CONFIRM_SECONDS, SCROLL_RELEASE_GRACE
from handtracking_state import FlowState, ScrollState


class FakeCursor:
    def __init__(self):
        self.synced = []

    def sync(self, active):
        self.synced.append(bool(active))


class ScrollHandlerTests(unittest.TestCase):
    def test_scroll_arms_after_confirmed_pose(self):
        from handtracking_scroll import update_scroll_state

        scroll = ScrollState(candidate_at=10.0 - SCROLL_CONFIRM_SECONDS - 0.01)
        flow = FlowState()
        cursor = FakeCursor()

        update_scroll_state(
            scroll,
            now=10.0,
            gesture_now=True,
            blocked=False,
            cursor=cursor,
            flow=flow,
        )

        self.assertTrue(scroll.active)
        self.assertEqual(scroll.residual, 0.0)
        self.assertEqual(cursor.synced[-1], False)

    def test_scroll_releases_after_pose_loss_grace(self):
        from handtracking_scroll import update_scroll_state

        scroll = ScrollState(
            active=True,
            release_at=20.0 - SCROLL_RELEASE_GRACE - 0.01,
        )
        flow = FlowState()
        cursor = FakeCursor()

        update_scroll_state(
            scroll,
            now=20.0,
            gesture_now=False,
            blocked=False,
            cursor=cursor,
            flow=flow,
        )

        self.assertFalse(scroll.active)
        self.assertEqual(cursor.synced[-1], True)

    def test_dedicated_mode_block_resets_scroll_immediately(self):
        from handtracking_scroll import update_scroll_state

        scroll = ScrollState(active=True, residual=22.0)

        update_scroll_state(
            scroll,
            now=5.0,
            gesture_now=True,
            blocked=True,
            cursor=FakeCursor(),
            flow=FlowState(),
        )

        self.assertFalse(scroll.active)
        self.assertEqual(scroll.residual, 0.0)


if __name__ == "__main__":
    unittest.main()
