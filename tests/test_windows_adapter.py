import time
import unittest


class FakeUser32:
    def __init__(self):
        self.x = 100
        self.y = 200
        self.moves = []
        self.mouse_events = []
        self.key_events = []

    def GetCursorPos(self, point_ptr):
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


class WindowsAdapterTests(unittest.TestCase):
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

    def test_cursor_controller_can_set_absolute_position(self):
        from handtracking_windows import CursorController

        user32 = FakeUser32()
        cursor = CursorController(user32=user32)
        cursor.set_position(321, 123)

        self.assertEqual((user32.x, user32.y), (321, 123))
        self.assertEqual(user32.moves[-1], (321, 123))


if __name__ == "__main__":
    unittest.main()
