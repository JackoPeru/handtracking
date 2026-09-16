import time
import unittest
from unittest import mock


class FakeUser32:
    def __init__(self):
        self.x = 100
        self.y = 200
        self.moves = []
        self.mouse_events = []
        self.key_events = []
        self.cursor_reads = 0

    def GetCursorPos(self, point_ptr):
        self.cursor_reads += 1
        point = point_ptr._obj
        point.x = self.x
        point.y = self.y
        return 1

    def SetCursorPos(self, x, y):
        self.x = x
        self.y = y
        self.moves.append((x, y))
        return 1

    def GetSystemMetrics(self, index):
        return 1920 if index == 0 else 1080

    def mouse_event(self, *args):
        self.mouse_events.append(args)

    def keybd_event(self, *args):
        self.key_events.append(args)

    def GetForegroundWindow(self):
        return 0


class FailingInputUser32(FakeUser32):
    def __init__(self, *, fail_mouse_at=None, fail_key_at=None,
                 fail_cursor_read=False, fail_cursor_write=False):
        super().__init__()
        self.fail_mouse_at = fail_mouse_at
        self.fail_key_at = fail_key_at
        self.fail_cursor_read = fail_cursor_read
        self.fail_cursor_write = fail_cursor_write
        self.mouse_calls = 0
        self.key_calls = 0

    def GetCursorPos(self, point_ptr):
        if self.fail_cursor_read:
            return 0
        return super().GetCursorPos(point_ptr)

    def SetCursorPos(self, x, y):
        if self.fail_cursor_write:
            return 0
        return super().SetCursorPos(x, y)

    def mouse_event(self, *args):
        self.mouse_calls += 1
        if self.mouse_calls == self.fail_mouse_at:
            raise RuntimeError(f"mouse failure {self.mouse_calls}")
        return super().mouse_event(*args)

    def keybd_event(self, *args):
        self.key_calls += 1
        if self.key_calls == self.fail_key_at:
            raise RuntimeError(f"key failure {self.key_calls}")
        return super().keybd_event(*args)


class ControlledClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class WindowsAdapterTests(unittest.TestCase):
    def test_foreground_title_uses_bounded_buffer(self):
        from handtracking_windows import foreground_window_title
        class HugeTitleUser32(FakeUser32):
            def GetForegroundWindow(self):
                return 1
            def GetWindowTextLengthW(self, hwnd):
                return 10_000_000
            def GetWindowTextW(self, hwnd, buffer, size):
                self.requested_title_size = size
                buffer.value = "X" * min(size - 1, 16)
                return len(buffer.value)
        api = HugeTitleUser32()
        self.assertEqual(foreground_window_title(api), "X" * 16)
        self.assertLessEqual(api.requested_title_size, 4097)

    def test_cursor_controller_is_inactive_until_explicitly_started(self):
        from handtracking_windows import CursorController

        fake = FakeUser32()
        cursor = CursorController(user32=fake, output_hz=240.0, interp_tau=0.001)

        self.assertFalse(cursor.active)
        self.assertFalse(cursor.running)
        self.assertEqual(cursor.position(), (100, 200))

    def test_cursor_controller_clamps_target_to_screen(self):
        from handtracking_windows import CursorController

        fake = FakeUser32()
        cursor = CursorController(user32=fake)
        cursor.sync(True)
        cursor.add_delta(5000, -5000, screen_size=(1920, 1080))

        self.assertEqual(cursor.target, (1919.0, 0.0))

    def test_cursor_renewal_does_not_activate_and_stale_renewal_cannot_extend(self):
        from handtracking_windows import CursorController

        clock = ControlledClock(10.0)
        cursor = CursorController(
            user32=FakeUser32(), clock=clock, freshness_timeout=0.22
        )

        self.assertTrue(cursor.renew_freshness(10.0))
        self.assertFalse(cursor.active)
        self.assertFalse(cursor.renew_freshness(9.0))
        self.assertFalse(cursor.active)
        self.assertTrue(cursor.renew_freshness(10.0))
        cursor.sync(True)
        self.assertTrue(cursor.active)

        clock.value = 10.21
        cursor.sync(False)
        cursor.sync(True)
        self.assertTrue(cursor.active)

        clock.value = 10.23
        cursor.sync(False)
        cursor.sync(True)
        self.assertFalse(cursor.active)

    def test_cursor_invalid_freshness_inputs_fail_closed(self):
        from handtracking_windows import CursorController

        cursor = CursorController(user32=FakeUser32())
        cursor.sync(True)
        self.assertFalse(cursor.renew_freshness(float("nan")))
        self.assertFalse(cursor.active)
        self.assertFalse(cursor.renew_freshness(float("inf")))
        self.assertFalse(cursor.active)

    def test_cursor_rejects_future_or_stale_renewal_and_fails_closed(self):
        from handtracking_windows import CursorController

        clock = ControlledClock(10.0)
        cursor = CursorController(user32=FakeUser32(), clock=clock)
        cursor.renew_freshness(10.0)
        cursor.sync(True)

        self.assertFalse(cursor.renew_freshness(10.1))
        self.assertFalse(cursor.active)

        clock.value = 10.0
        cursor = CursorController(user32=FakeUser32(), clock=clock)
        cursor.renew_freshness(10.0)
        cursor.sync(True)
        self.assertFalse(cursor.renew_freshness(9.0))
        self.assertFalse(cursor.active)

        broken_clock = CursorController(
            user32=FakeUser32(), clock=lambda: float("nan")
        )
        self.assertFalse(broken_clock.renew_freshness(10.0))
        self.assertFalse(broken_clock.active)

    def test_cursor_worker_skips_write_if_read_crosses_lease_deadline(self):
        from handtracking_windows import CursorController

        clock = ControlledClock(10.0)
        fake = FakeUser32()
        cursor = CursorController(
            user32=fake, clock=clock, output_hz=500.0, interp_tau=0.001
        )
        cursor.renew_freshness(10.0)
        cursor.sync(True)
        cursor.add_delta(10, 0, screen_size=(1920, 1080))

        original_position = cursor.position

        def blocked_position():
            result = original_position()
            clock.value = 10.23
            return result

        cursor.position = blocked_position
        cursor.start()
        self.assertTrue(wait_until(lambda: not cursor.active))
        self.assertFalse(cursor.active)
        self.assertEqual(fake.moves, [])
        cursor.close()

    def test_cursor_output_latency_records_once_per_target_generation(self):
        from handtracking_windows import CursorController

        clock = ControlledClock(10.0)
        fake = FakeUser32()
        cursor = CursorController(
            user32=fake, clock=clock, output_hz=500.0, interp_tau=0.001
        )
        cursor.renew_freshness(10.0)
        cursor.sync(True)
        cursor.set_input_time(9.9)
        cursor.add_delta(10, 0, screen_size=(1920, 1080))
        cursor.start()
        self.assertTrue(wait_until(lambda: bool(fake.moves)))
        time.sleep(0.01)
        self.assertEqual(cursor.output_latency().samples, 1)

        cursor.set_input_time(9.95)
        cursor.add_delta(10, 0, screen_size=(1920, 1080))
        self.assertTrue(wait_until(lambda: cursor.output_latency().samples == 2))
        cursor.close()
        metric = cursor.output_latency()
        self.assertAlmostEqual(metric.p50_ms, 50.0)
        self.assertAlmostEqual(metric.p95_ms, 100.0)
        self.assertAlmostEqual(metric.p99_ms, 100.0)

    def test_worker_expires_without_main_loop_reading_active_or_renewing(self):
        from handtracking_windows import CursorController

        clock = ControlledClock(10.0)
        fake = FakeUser32()
        cursor = CursorController(user32=fake, clock=clock, output_hz=500,
                                  interp_tau=0.001)
        original_send = fake.SetCursorPos
        def expire_after_send(x, y):
            result = original_send(x, y)
            clock.value = 10.3
            return result
        fake.SetCursorPos = expire_after_send
        cursor.renew_freshness(10.0)
        cursor.sync(True)
        cursor.add_delta(100, 0)
        cursor.start()
        try:
            self.assertTrue(wait_until(lambda: len(fake.moves) == 1))
            # No cursor.active/property calls: expiry must happen in the worker.
            time.sleep(.03)
            self.assertEqual(len(fake.moves), 1)
            self.assertTrue(cursor.running)
        finally:
            cursor.close()

    def test_cursor_controller_stops_its_worker(self):
        from handtracking_windows import CursorController

        fake = FakeUser32()
        cursor = CursorController(user32=fake, output_hz=500.0, interp_tau=0.001)
        cursor.start()
        cursor.sync(True)
        cursor.add_delta(10, 0, screen_size=(1920, 1080))
        time.sleep(0.02)
        cursor.close()

        self.assertFalse(cursor.running)

    def test_cursor_worker_records_get_cursor_pos_failure_and_deactivates(self):
        from handtracking_windows import CursorController

        fake = FailingInputUser32()
        cursor = CursorController(user32=fake, output_hz=500.0)
        cursor.sync(True)
        fake.fail_cursor_read = True
        cursor.start()

        self.assertTrue(wait_until(lambda: not cursor.running))
        self.assertFalse(cursor.active)
        self.assertIn("GetCursorPos failed", cursor.last_error)
        cursor.close()

    def test_cursor_worker_records_set_cursor_pos_failure_and_deactivates(self):
        from handtracking_windows import CursorController

        fake = FailingInputUser32()
        cursor = CursorController(user32=fake, output_hz=500.0, interp_tau=0.001)
        cursor.sync(True)
        cursor.add_delta(10, 0, screen_size=(1920, 1080))
        fake.fail_cursor_write = True
        cursor.start()

        self.assertTrue(wait_until(lambda: not cursor.running))
        self.assertFalse(cursor.active)
        self.assertIn("SetCursorPos failed", cursor.last_error)
        cursor.close()

    def test_cursor_start_clears_previous_worker_error(self):
        from handtracking_windows import CursorController

        fake = FailingInputUser32()
        cursor = CursorController(user32=fake, output_hz=500.0)
        cursor.sync(True)
        fake.fail_cursor_read = True
        cursor.start()
        self.assertTrue(wait_until(lambda: not cursor.running))
        self.assertNotEqual(cursor.last_error, "")

        fake.fail_cursor_read = False
        cursor.start()

        self.assertEqual(cursor.last_error, "")
        cursor.close()

    def test_cursor_controller_can_set_absolute_position(self):
        from handtracking_windows import CursorController

        user32 = FakeUser32()
        cursor = CursorController(user32=user32)
        cursor.set_position(321, 123)

        self.assertEqual((user32.x, user32.y), (321, 123))
        self.assertEqual(user32.moves[-1], (321, 123))

    def test_cursor_controller_rejects_failed_win32_position_calls(self):
        from handtracking_windows import CursorController

        fake = FailingInputUser32(fail_cursor_read=True)
        cursor = CursorController(user32=fake)
        with self.assertRaisesRegex(OSError, "GetCursorPos failed"):
            cursor.position()

        fake.fail_cursor_read = False
        fake.fail_cursor_write = True
        with self.assertRaisesRegex(OSError, "SetCursorPos failed"):
            cursor.set_position(1, 2)

    def test_cursor_sync_only_reads_os_position_when_activating(self):
        from handtracking_windows import CursorController

        user32 = FakeUser32()
        cursor = CursorController(user32=user32)

        cursor.sync(False)
        cursor.sync(False)
        self.assertEqual(user32.cursor_reads, 0)

        cursor.sync(True)
        self.assertEqual(user32.cursor_reads, 1)
        cursor.sync(True)
        self.assertEqual(user32.cursor_reads, 1)

        cursor.sync(False)
        self.assertEqual(user32.cursor_reads, 1)

    def test_left_click_releases_after_down_failure_and_preserves_original_error(self):
        from handtracking_windows import left_click

        fake = FailingInputUser32(fail_mouse_at=1)
        with self.assertRaisesRegex(RuntimeError, "mouse failure 1"):
            left_click(fake)

        self.assertEqual(
            [event[0] for event in fake.mouse_events],
            [0x0004],
        )
        self.assertEqual(fake.mouse_calls, 2)

    def test_left_click_keeps_normal_down_up_order(self):
        from handtracking_windows import MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, left_click

        fake = FakeUser32()
        left_click(fake)

        self.assertEqual(
            [event[0] for event in fake.mouse_events],
            [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP],
        )

    def test_left_click_retries_release_after_up_failure(self):
        from handtracking_windows import MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, left_click

        fake = FailingInputUser32(fail_mouse_at=2)
        with self.assertRaisesRegex(RuntimeError, "mouse failure 2"):
            left_click(fake)

        self.assertEqual(fake.mouse_calls, 3)
        self.assertEqual(
            [event[0] for event in fake.mouse_events],
            [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP],
        )

    def test_left_click_preserves_down_error_when_release_also_fails(self):
        from handtracking_windows import left_click

        fake = FailingInputUser32(fail_mouse_at=1)
        def fail_both(*args):
            fake.mouse_calls += 1
            raise RuntimeError(
                "down failure" if fake.mouse_calls == 1 else "up failure"
            )

        fake.mouse_event = fail_both
        with self.assertRaisesRegex(RuntimeError, "down failure"):
            left_click(fake)
        self.assertEqual(fake.mouse_calls, 2)

    def test_tap_combo_releases_all_pressed_keys_after_mid_sequence_failure(self):
        from handtracking_windows import VK_CONTROL, VK_TAB, tap_combo

        fake = FailingInputUser32(fail_key_at=2)
        with self.assertRaisesRegex(RuntimeError, "key failure 2"):
            tap_combo(VK_TAB, (VK_CONTROL,), fake)

        self.assertEqual(
            [(event[0], event[2]) for event in fake.key_events],
            [(VK_CONTROL, 0), (VK_TAB, 2), (VK_CONTROL, 2)],
        )
        self.assertEqual(fake.key_calls, 4)

    def test_tap_combo_attempts_every_release_and_preserves_operation_error(self):
        from handtracking_windows import VK_CONTROL, VK_TAB, tap_combo

        fake = FailingInputUser32()
        events = []

        def key_event(*args):
            events.append(args)
            if len(events) == 3:
                raise RuntimeError("main key up failed")
            if args[2] and len(events) == 4:
                raise RuntimeError("modifier up failed")

        fake.keybd_event = key_event
        with self.assertRaisesRegex(RuntimeError, "main key up failed"):
            tap_combo(VK_TAB, (VK_CONTROL,), fake)

        self.assertEqual(len(events), 6)
        self.assertEqual(events[-1][0:3], (VK_CONTROL, 0, 2))

    def test_tap_combo_keeps_normal_modifier_order(self):
        from handtracking_windows import KEYEVENTF_KEYUP, VK_CONTROL, VK_TAB, tap_combo

        fake = FakeUser32()
        tap_combo(VK_TAB, (VK_CONTROL,), fake)

        self.assertEqual(
            [(event[0], event[2] == KEYEVENTF_KEYUP) for event in fake.key_events],
            [(VK_CONTROL, False), (VK_TAB, False), (VK_TAB, True), (VK_CONTROL, True)],
        )

    def test_ctrl_wheel_releases_control_after_wheel_failure(self):
        from handtracking_windows import ctrl_wheel

        fake = FailingInputUser32(fail_mouse_at=1)
        with self.assertRaisesRegex(RuntimeError, "mouse failure 1"):
            ctrl_wheel(120, fake)

        self.assertEqual(fake.key_calls, 2)

    def test_tap_combo_retries_transient_modifier_release_failure(self):
        from handtracking_windows import VK_CONTROL, VK_TAB, tap_combo

        fake = FailingInputUser32(fail_key_at=4)
        with self.assertRaisesRegex(RuntimeError, "key failure 4"):
            tap_combo(VK_TAB, (VK_CONTROL,), fake)
        self.assertEqual(fake.key_calls, 5)
        self.assertEqual(fake.key_events[-1][0:3], (VK_CONTROL, 0, 2))

    def test_ctrl_wheel_retries_transient_control_release_failure(self):
        from handtracking_windows import VK_CONTROL, ctrl_wheel

        fake = FailingInputUser32(fail_key_at=2)
        with self.assertRaisesRegex(RuntimeError, "key failure 2"):
            ctrl_wheel(120, fake)
        self.assertEqual(fake.key_calls, 3)
        self.assertEqual(fake.key_events[-1][0:3], (VK_CONTROL, 0, 2))


class WindowsAudioTests(unittest.TestCase):
    def setUp(self):
        import handtracking_windows as windows

        self.windows = windows
        windows._volume_endpoint = None

    def tearDown(self):
        self.windows._volume_endpoint = None

    @staticmethod
    def _speaker(endpoint):
        return mock.Mock(EndpointVolume=endpoint)

    def test_get_system_volume_retries_after_initial_endpoint_failure(self):
        endpoint = mock.Mock()
        endpoint.GetMasterVolumeLevelScalar.return_value = 0.31
        with mock.patch.object(
            self.windows.AudioUtilities,
            "GetSpeakers",
            side_effect=[RuntimeError("endpoint unavailable"), self._speaker(endpoint)],
        ) as get_speakers:
            self.assertEqual(self.windows.get_system_volume(), 0.5)
            self.assertAlmostEqual(self.windows.get_system_volume(), 0.31)

        self.assertEqual(get_speakers.call_count, 2)

    def test_get_system_volume_refreshes_default_endpoint_each_call(self):
        first = mock.Mock()
        first.GetMasterVolumeLevelScalar.return_value = 0.2
        second = mock.Mock()
        second.GetMasterVolumeLevelScalar.return_value = 0.8
        with mock.patch.object(
            self.windows.AudioUtilities,
            "GetSpeakers",
            side_effect=[self._speaker(first), self._speaker(second)],
        ) as get_speakers:
            self.assertAlmostEqual(self.windows.get_system_volume(), 0.2)
            self.assertAlmostEqual(self.windows.get_system_volume(), 0.8)

        self.assertEqual(get_speakers.call_count, 2)

    def test_set_system_volume_invalidates_stale_endpoint_and_retries(self):
        stale = mock.Mock()
        stale.SetMasterVolumeLevelScalar.side_effect = RuntimeError("stale endpoint")
        fresh = mock.Mock()

        with mock.patch.object(
            self.windows.AudioUtilities,
            "GetSpeakers",
            side_effect=[self._speaker(stale), self._speaker(fresh)],
        ) as get_speakers:
            self.assertTrue(self.windows.set_system_volume(0.7))

        stale.SetMasterVolumeLevelScalar.assert_called_once()
        fresh.SetMasterVolumeLevelScalar.assert_called_once_with(0.7, None)
        self.assertEqual(get_speakers.call_count, 2)

    def test_set_system_volume_refreshes_default_endpoint_before_write(self):
        stale = mock.Mock()
        fresh = mock.Mock()
        self.windows._volume_endpoint = stale

        with mock.patch.object(
            self.windows.AudioUtilities,
            "GetSpeakers",
            return_value=self._speaker(fresh),
        ) as get_speakers:
            self.assertTrue(self.windows.set_system_volume(0.7))

        stale.SetMasterVolumeLevelScalar.assert_not_called()
        fresh.SetMasterVolumeLevelScalar.assert_called_once_with(0.7, None)
        get_speakers.assert_called_once_with()

    def test_set_system_volume_returns_false_after_retry_failure(self):
        broken = mock.Mock()
        broken.SetMasterVolumeLevelScalar.side_effect = RuntimeError("set failed")
        replacement = mock.Mock()
        replacement.SetMasterVolumeLevelScalar.side_effect = RuntimeError("set failed again")

        with mock.patch.object(
            self.windows.AudioUtilities,
            "GetSpeakers",
            side_effect=[self._speaker(broken), self._speaker(replacement)],
        ) as get_speakers:
            self.assertFalse(self.windows.set_system_volume(0.4))

        self.assertEqual(get_speakers.call_count, 2)
        self.assertIsNone(self.windows._volume_endpoint)


if __name__ == "__main__":
    unittest.main()
