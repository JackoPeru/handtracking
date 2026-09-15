import unittest
from unittest import mock

from handtracking_gestures import normalized_pinch_ratio
from handtracking_settings import RuntimeSettings
from tests import test_runtime_smoke as helpers


class SettingsIntegrationTests(unittest.TestCase):
    def make_session(self, settings):
        from handtracking_session import RuntimeSession
        return RuntimeSession(
            camera=helpers.FakeRuntimeCamera(), worker=helpers.PacketWorker([]),
            cursor=helpers.FakeCursor(), screen_w=1920, screen_h=1080,
            start_time=1.0, commands_enabled=True, settings=settings,
        )

    def test_pointer_coordinator_uses_calibrated_threshold(self):
        from handtracking_modes import process_pointer
        hand = helpers.RuntimeSmokeTests._pinch_packet(1)[1].hand_landmarks[0]
        hand[4].x += .10
        ratio = normalized_pinch_ratio(hand)
        self.assertGreater(ratio, .05)
        self.assertLess(ratio, .46)
        outcomes = []
        for threshold in (.05, .46):
            session = self.make_session(RuntimeSettings(pinch_on=threshold))
            process_pointer(session, hands=[hand], control_hand=hand,
                            volume_candidate_now=False, now=10.0,
                            left_click_cb=lambda: self.fail("not a release"))
            outcomes.append(session.pointer.pinch_held)
        self.assertEqual(outcomes, [False, True])

    def test_mediapipe_fallback_uses_the_same_calibrated_gain(self):
        from handtracking_modes import sync_cursor_and_fallback
        displacements = []
        for sensitivity in (1.0, 2.0):
            session = self.make_session(RuntimeSettings(sensitivity=sensitivity))
            session.pointer.pinch_held = session.pointer.move_active = True
            session.cursor.sync(True)
            session.mp_control_ref = (.502, .5)
            sync_cursor_and_fallback(session, corrected=None,
                                     old_mp_ref=(.5, .5), old_pause=False, now=10.0)
            displacements.append(session.cursor.position()[0] - 100)
        self.assertGreater(displacements[0], 0)
        self.assertAlmostEqual(displacements[1], 2 * displacements[0])

    def test_runtime_opens_configured_camera_and_keeps_settings(self):
        import handtracking_runtime
        settings = RuntimeSettings(camera_index=3, profile="precisione")
        camera = helpers.FakeRuntimeCamera()
        session = self.make_session(settings)
        with mock.patch.object(handtracking_runtime.CameraRuntime, "open",
                               return_value=camera) as opened, \
                mock.patch.object(handtracking_runtime.RuntimeSession, "create",
                                  return_value=session) as created, \
                mock.patch.object(handtracking_runtime, "_run_impl"):
            try:
                handtracking_runtime.run(settings=settings)
            except TypeError as exc:
                self.fail("runtime settings not wired: " + str(exc))
        self.assertEqual(opened.call_args.kwargs["camera_index"], 3)
        self.assertIs(created.call_args.kwargs["settings"], settings)

    def test_camera_rate_lk_uses_calibrated_gain(self):
        import numpy as np
        from handtracking_flow import LKMotion
        deltas = []
        for sensitivity in (1.0, 2.0):
            session = self.make_session(RuntimeSettings(sensitivity=sensitivity))
            session.worker = helpers.PacketWorker([helpers.RuntimeSmokeTests._pinch_packet(1)])
            session.pointer.pinch_held = session.pointer.move_active = True
            session.mp_control_ref = (.541, .604)
            session.flow.prev_gray = session.camera.gray
            session.flow.points = np.zeros((5, 1, 2), dtype=np.float32)
            session.flow.time = session.start_time
            session.cursor.sync(True)
            before = session.cursor.position()[0]
            motion = LKMotion(session.flow.points, 1.28, 0.0, 1.28)
            with mock.patch("handtracking_flow.cv2.calcOpticalFlowPyrLK",
                            return_value=(None, None, None)):
                helpers.RuntimeSmokeTests()._run_runtime_session(session, motion=motion)
            deltas.append(session.cursor.position()[0] - before)
        self.assertGreater(deltas[0], 0)
        self.assertAlmostEqual(deltas[1], 2 * deltas[0], places=4)


if __name__ == "__main__":
    unittest.main()
