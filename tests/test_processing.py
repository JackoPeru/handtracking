import unittest

from handtracking_state import SpockState


class ProcessingTests(unittest.TestCase):
    def test_update_ema_metrics_initializes_and_smooths(self):
        from handtracking_processing import update_ema_metrics

        first = update_ema_metrics((0.0, 0.0, 0.0, 0.0), (20.0, 30.0, 40.0, 5.0))
        self.assertEqual(first, (20.0, 30.0, 40.0, 5.0))

        second = update_ema_metrics(first, (40.0, 50.0, 60.0, 15.0))
        self.assertAlmostEqual(second[0], 23.0)
        self.assertAlmostEqual(second[1], 33.0)
        self.assertAlmostEqual(second[2], 43.0)
        self.assertAlmostEqual(second[3], 6.5)

    def test_spock_update_toggles_after_confirmed_hold(self):
        from handtracking_processing import update_spock_state

        state = SpockState()
        commands = False
        block_until = 0.0

        first = update_spock_state(
            state,
            raw_score=1.0,
            upright_now=True,
            now=1.0,
            sample_seconds=0.5,
            commands_enabled=commands,
            input_block_until=block_until,
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
        from handtracking_processing import update_spock_state

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


if __name__ == "__main__":
    unittest.main()
