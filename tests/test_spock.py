import unittest

from handtracking_state import SpockState


class SpockProcessingTests(unittest.TestCase):
    def test_spock_update_toggles_after_confirmed_hold(self):
        from handtracking_spock import update_spock_state

        state = SpockState()
        first = update_spock_state(
            state,
            raw_score=1.0,
            upright_now=True,
            now=1.0,
            sample_seconds=0.5,
            commands_enabled=False,
            input_block_until=0.0,
        )
        self.assertFalse(first.toggled)
        self.assertTrue(state.blocking)

        second = update_spock_state(
            state,
            raw_score=1.0,
            upright_now=True,
            now=1.5,
            sample_seconds=0.5,
            commands_enabled=first.commands_enabled,
            input_block_until=first.input_block_until,
        )
        self.assertTrue(second.toggled)
        self.assertTrue(second.commands_enabled)
        self.assertEqual(second.event, "CONTROLLI ATTIVI")

    def test_spock_update_reports_release_after_latched_pose_is_opened(self):
        from handtracking_config import SPOCK_RELEASE_SECONDS
        from handtracking_spock import update_spock_state

        state = SpockState(latched=True, blocking=True, progress=1.0)
        first = update_spock_state(
            state,
            raw_score=0.0,
            upright_now=False,
            now=2.0,
            sample_seconds=0.05,
            commands_enabled=True,
            input_block_until=0.0,
        )
        self.assertFalse(first.released)

        second = update_spock_state(
            state,
            raw_score=0.0,
            upright_now=False,
            now=2.0 + SPOCK_RELEASE_SECONDS + 0.01,
            sample_seconds=0.05,
            commands_enabled=True,
            input_block_until=first.input_block_until,
        )
        self.assertTrue(second.released)
        self.assertFalse(state.latched)
        self.assertFalse(state.blocking)

    def test_no_hand_update_releases_latched_spock_after_grace(self):
        from handtracking_config import SPOCK_RELEASE_SECONDS
        from handtracking_spock import update_spock_without_hands

        state = SpockState(latched=True, blocking=True, progress=1.0)
        first = update_spock_without_hands(
            state,
            now=5.0,
            input_block_until=0.0,
        )
        self.assertTrue(state.latched)
        self.assertEqual(state.release_at, 5.0)

        second = update_spock_without_hands(
            state,
            now=5.0 + SPOCK_RELEASE_SECONDS + 0.01,
            input_block_until=first,
        )
        self.assertFalse(state.latched)
        self.assertFalse(state.blocking)
        self.assertGreater(second, 5.0)

    def test_no_hand_update_clears_expired_candidate(self):
        from handtracking_config import SPOCK_MISS_GRACE
        from handtracking_spock import update_spock_without_hands

        state = SpockState(
            candidate_at=2.0,
            last_seen=2.0,
            blocking=True,
            progress=0.5,
            confirmed_seconds=0.5,
        )

        update_spock_without_hands(
            state,
            now=2.0 + SPOCK_MISS_GRACE + 0.01,
            input_block_until=0.0,
        )

        self.assertIsNone(state.candidate_at)
        self.assertIsNone(state.last_seen)
        self.assertFalse(state.blocking)
        self.assertEqual(state.progress, 0.0)


if __name__ == "__main__":
    unittest.main()
