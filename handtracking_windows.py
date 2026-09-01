"""Lazy Windows input/audio adapters and cursor output controller."""

import ctypes
import math
import threading
import time

from pycaw.pycaw import AudioUtilities

from handtracking_config import CURSOR_INTERP_TAU, CURSOR_OUTPUT_HZ
from handtracking_gestures import clamp


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_TAB = 0x09
VK_LEFT = 0x25
VK_LWIN = 0x5B
VK_D = 0x44
VK_BROWSER_BACK = 0xA6
VK_BROWSER_FORWARD = 0xA7


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


_user32 = None
_volume_endpoint = None
_volume_initialized = False


def get_user32():
    global _user32
    if _user32 is None:
        _user32 = ctypes.windll.user32
        try:
            _user32.SetProcessDPIAware()
        except Exception:
            pass
    return _user32


def screen_size(user32=None):
    api = user32 or get_user32()
    return int(api.GetSystemMetrics(0)), int(api.GetSystemMetrics(1))


def cursor_position(user32=None):
    api = user32 or get_user32()
    point = POINT()
    api.GetCursorPos(ctypes.byref(point))
    return int(point.x), int(point.y)


def mouse_down(user32=None):
    (user32 or get_user32()).mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def mouse_up(user32=None):
    (user32 or get_user32()).mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def left_click(user32=None):
    mouse_down(user32)
    mouse_up(user32)


def mouse_wheel(delta, user32=None):
    wheel_data = ctypes.c_uint32(int(delta) & 0xFFFFFFFF).value
    (user32 or get_user32()).mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_data, 0)


def key_down(vk, user32=None):
    (user32 or get_user32()).keybd_event(vk, 0, 0, 0)


def key_up(vk, user32=None):
    (user32 or get_user32()).keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def tap_combo(vk, modifiers=(), user32=None):
    api = user32 or get_user32()
    for mod in modifiers:
        key_down(mod, api)
    key_down(vk, api)
    key_up(vk, api)
    for mod in reversed(modifiers):
        key_up(mod, api)


def ctrl_wheel(delta, user32=None):
    api = user32 or get_user32()
    key_down(VK_CONTROL, api)
    try:
        mouse_wheel(delta, api)
    finally:
        key_up(VK_CONTROL, api)


def foreground_window_title(user32=None):
    api = user32 or get_user32()
    hwnd = api.GetForegroundWindow()
    if not hwnd:
        return "?"
    length = api.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(max(length + 1, 2))
    api.GetWindowTextW(hwnd, buf, len(buf))
    return buf.value or "?"


def execute_swipe(direction, user32=None):
    api = user32 or get_user32()
    target = foreground_window_title(api)
    if direction == "LEFT":
        tap_combo(VK_BROWSER_BACK, user32=api)
        return f"BACK SENT -> {target[:28]}"
    if direction == "RIGHT":
        tap_combo(VK_BROWSER_FORWARD, user32=api)
        return f"FORWARD SENT -> {target[:28]}"
    return ""


def execute_radial_action(action, user32=None):
    api = user32 or get_user32()
    if action == "LEFT":
        tap_combo(VK_LEFT, (VK_MENU,), api)
        return "BACK"
    if action == "RIGHT":
        tap_combo(VK_TAB, (VK_MENU,), api)
        return "NEXT APP"
    if action == "UP":
        tap_combo(VK_TAB, (VK_LWIN,), api)
        return "TASK VIEW"
    if action == "DOWN":
        tap_combo(VK_D, (VK_LWIN,), api)
        return "DESKTOP"
    return ""


def _get_volume_endpoint():
    global _volume_endpoint, _volume_initialized
    if not _volume_initialized:
        _volume_initialized = True
        try:
            _volume_endpoint = AudioUtilities.GetSpeakers().EndpointVolume
        except Exception:
            _volume_endpoint = None
    return _volume_endpoint


def get_system_volume():
    endpoint = _get_volume_endpoint()
    if endpoint is None:
        return 0.5
    return float(endpoint.GetMasterVolumeLevelScalar())


def set_system_volume(value):
    endpoint = _get_volume_endpoint()
    if endpoint is not None:
        endpoint.SetMasterVolumeLevelScalar(clamp(value, 0.0, 1.0), None)


class CursorController:
    def __init__(self, *, user32=None, output_hz=CURSOR_OUTPUT_HZ,
                 interp_tau=CURSOR_INTERP_TAU):
        self._user32 = user32
        self._output_hz = float(output_hz)
        self._interp_tau = float(interp_tau)
        self._lock = threading.Lock()
        self._target = [0.0, 0.0]
        self._active = False
        self._stop_event = threading.Event()
        self._thread = None

    @property
    def user32(self):
        if self._user32 is None:
            self._user32 = get_user32()
        return self._user32

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def target(self):
        with self._lock:
            return float(self._target[0]), float(self._target[1])

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def screen_size(self):
        return screen_size(self.user32)

    def position(self):
        return cursor_position(self.user32)

    def set_position(self, x, y):
        self.user32.SetCursorPos(int(round(x)), int(round(y)))

    def sync(self, active):
        active = bool(active)
        with self._lock:
            if self._active == active:
                return
            if not active:
                self._active = False
                return

        x, y = self.position()
        with self._lock:
            self._target[0] = float(x)
            self._target[1] = float(y)
            self._active = True

    def add_delta(self, dx, dy, *, screen_size=None):
        width, height = screen_size or self.screen_size()
        with self._lock:
            self._target[0] = clamp(self._target[0] + dx, 0, width - 1)
            self._target[1] = clamp(self._target[1] + dy, 0, height - 1)

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="cursor-output",
        )
        self._thread.start()

    def close(self):
        try:
            self.sync(False)
        except Exception:
            with self._lock:
                self._active = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)

    def _worker(self):
        last = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            dt = max(now - last, 1.0 / 500.0)
            last = now
            with self._lock:
                active = self._active
                tx, ty = self._target
            if active:
                x, y = self.position()
                alpha = 1.0 - math.exp(-dt / self._interp_tau)
                nx = x + (tx - x) * alpha
                ny = y + (ty - y) * alpha
                if abs(tx - x) > 0.5 or abs(ty - y) > 0.5:
                    self.user32.SetCursorPos(int(round(nx)), int(round(ny)))
            self._stop_event.wait(1.0 / self._output_hz)
