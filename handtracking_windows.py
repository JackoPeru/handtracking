"""Lazy Windows input/audio adapters and cursor output controller."""

import ctypes
from collections import deque
import math
import threading
import time

from pycaw.pycaw import AudioUtilities

from handtracking_config import CURSOR_INTERP_TAU, CURSOR_OUTPUT_HZ, MP_RESULT_STALE_SECONDS
from handtracking_gestures import clamp
from handtracking_perf import PERF_HISTORY_SIZE, PerfMetric, percentile_metric


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
    if not api.GetCursorPos(ctypes.byref(point)):
        raise OSError("GetCursorPos failed")
    return int(point.x), int(point.y)


def mouse_down(user32=None):
    (user32 or get_user32()).mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def mouse_up(user32=None):
    (user32 or get_user32()).mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def left_click(user32=None):
    error = None
    try:
        mouse_down(user32)
        mouse_up(user32)
    except BaseException as exc:
        error = exc
        try:
            mouse_up(user32)
        except BaseException:
            pass
    if error is not None:
        raise error


def mouse_wheel(delta, user32=None):
    wheel_data = ctypes.c_uint32(int(delta) & 0xFFFFFFFF).value
    (user32 or get_user32()).mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_data, 0)


def key_down(vk, user32=None):
    (user32 or get_user32()).keybd_event(vk, 0, 0, 0)


def key_up(vk, user32=None):
    (user32 or get_user32()).keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def tap_combo(vk, modifiers=(), user32=None):
    api = user32 or get_user32()
    pressed = []
    error = None
    try:
        for mod in modifiers:
            pressed.append(mod)
            key_down(mod, api)
        pressed.append(vk)
        key_down(vk, api)
        key_up(vk, api)
        pressed.pop()
    except BaseException as exc:
        error = exc

    release_error = None
    while pressed:
        key = pressed.pop()
        try:
            key_up(key, api)
        except BaseException as exc:
            if release_error is None:
                release_error = exc
            try:
                key_up(key, api)
            except BaseException:
                pass
    if error is not None:
        raise error
    if release_error is not None:
        raise release_error


def ctrl_wheel(delta, user32=None):
    api = user32 or get_user32()
    error = None
    try:
        key_down(VK_CONTROL, api)
        mouse_wheel(delta, api)
    except BaseException as exc:
        error = exc
    try:
        key_up(VK_CONTROL, api)
    except BaseException as exc:
        if error is None:
            error = exc
        try:
            key_up(VK_CONTROL, api)
        except BaseException:
            pass
    if error is not None:
        raise error


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


def _invalidate_volume_endpoint():
    global _volume_endpoint
    _volume_endpoint = None


def _get_volume_endpoint(*, refresh=False):
    global _volume_endpoint
    if refresh or _volume_endpoint is None:
        try:
            endpoint = AudioUtilities.GetSpeakers().EndpointVolume
        except Exception:
            _invalidate_volume_endpoint()
            return None
        if endpoint is None:
            _invalidate_volume_endpoint()
            return None
        _volume_endpoint = endpoint
    return _volume_endpoint


def get_system_volume():
    endpoint = _get_volume_endpoint(refresh=True)
    if endpoint is None:
        return 0.5
    try:
        return float(endpoint.GetMasterVolumeLevelScalar())
    except Exception:
        _invalidate_volume_endpoint()
        endpoint = _get_volume_endpoint(refresh=True)
        if endpoint is None:
            return 0.5
        try:
            return float(endpoint.GetMasterVolumeLevelScalar())
        except Exception:
            _invalidate_volume_endpoint()
            return 0.5


def set_system_volume(value):
    endpoint = _get_volume_endpoint(refresh=True)
    if endpoint is None:
        return False
    level = clamp(value, 0.0, 1.0)
    try:
        endpoint.SetMasterVolumeLevelScalar(level, None)
        return True
    except Exception:
        _invalidate_volume_endpoint()
        endpoint = _get_volume_endpoint()
        if endpoint is None:
            return False
        try:
            endpoint.SetMasterVolumeLevelScalar(level, None)
            return True
        except Exception:
            _invalidate_volume_endpoint()
            return False


class CursorController:
    def __init__(self, *, user32=None, output_hz=CURSOR_OUTPUT_HZ,
                 interp_tau=CURSOR_INTERP_TAU, clock=time.perf_counter,
                 freshness_timeout=MP_RESULT_STALE_SECONDS):
        self._user32 = user32
        self._output_hz = float(output_hz)
        self._interp_tau = float(interp_tau)
        self._clock = time.perf_counter if clock is None else clock
        self._freshness_timeout = float(freshness_timeout)
        if not math.isfinite(self._freshness_timeout) or self._freshness_timeout <= 0:
            raise ValueError("freshness_timeout must be finite and positive")
        self._lock = threading.Lock()
        self._target = [0.0, 0.0]
        self._active = False
        self._freshness_deadline = None
        self._freshness_input_at = None
        self._lease_established = False
        self._input_time = None
        self._target_generation = 0
        self._target_source_time = None
        self._latency_recorded_generation = 0
        self._latency_samples = deque(maxlen=PERF_HISTORY_SIZE)
        self._latency_total_samples = 0
        self._latency_last_ms = 0.0
        self._latency_ema_ms = 0.0
        self._last_error = ""
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
            if self._active and self._lease_expired_locked(self._clock_value()):
                self._active = False
            return self._active

    @property
    def target(self):
        with self._lock:
            return float(self._target[0]), float(self._target[1])

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    def screen_size(self):
        return screen_size(self.user32)

    def position(self):
        return cursor_position(self.user32)

    def set_position(self, x, y):
        if not self.user32.SetCursorPos(int(round(x)), int(round(y))):
            raise OSError("SetCursorPos failed")

    def _clock_value(self):
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    def _lease_expired_locked(self, now):
        return now is None or (
            self._freshness_deadline is not None and
            now >= self._freshness_deadline
        )

    def _fail_closed(self, *, clear_input_time=False):
        with self._lock:
            self._active = False
            self._freshness_deadline = 0.0
            self._freshness_input_at = None
            if clear_input_time:
                self._input_time = None

    def renew_freshness(self, input_at):
        try:
            input_at = float(input_at)
        except (TypeError, ValueError, OverflowError):
            self._fail_closed()
            return False
        if not math.isfinite(input_at):
            self._fail_closed()
            return False
        deadline = input_at + self._freshness_timeout
        if not math.isfinite(deadline):
            self._fail_closed()
            return False
        with self._lock:
            now = self._clock_value()
            if (
                now is None or
                input_at > now or
                self._freshness_input_at is not None and
                input_at < self._freshness_input_at
            ) or deadline <= now:
                self._active = False
                self._freshness_deadline = 0.0
                self._freshness_input_at = None
                return False
            self._freshness_input_at = input_at
            self._freshness_deadline = deadline
            self._lease_established = True
            return True

    def set_input_time(self, frame_ready_at):
        try:
            frame_ready_at = float(frame_ready_at)
        except (TypeError, ValueError, OverflowError):
            self._fail_closed(clear_input_time=True)
            return False
        if not math.isfinite(frame_ready_at):
            self._fail_closed(clear_input_time=True)
            return False
        with self._lock:
            if (
                self._input_time is not None and
                frame_ready_at < self._input_time
            ):
                return False
            self._input_time = frame_ready_at
            return True

    def sync(self, active):
        active = bool(active)
        with self._lock:
            if not active:
                self._active = False
                return
            now = self._clock_value()
            if self._active:
                if self._lease_expired_locked(now):
                    self._active = False
                else:
                    return
            if self._freshness_deadline is None:
                if self._lease_established:
                    return
                if now is None:
                    return
                self._freshness_deadline = now + self._freshness_timeout
                self._lease_established = True
            elif self._lease_expired_locked(now):
                return

        x, y = self.position()
        with self._lock:
            if self._lease_expired_locked(self._clock_value()):
                self._active = False
                return
            self._target[0] = float(x)
            self._target[1] = float(y)
            self._active = True

    def add_delta(self, dx, dy, *, screen_size=None, source_at=None):
        try:
            dx = float(dx)
            dy = float(dy)
        except (TypeError, ValueError, OverflowError):
            self._fail_closed(clear_input_time=True)
            return False
        if not math.isfinite(dx) or not math.isfinite(dy):
            self._fail_closed(clear_input_time=True)
            return False
        if source_at is not None:
            try:
                source_at = float(source_at)
            except (TypeError, ValueError, OverflowError):
                self._fail_closed(clear_input_time=True)
                return False
            if not math.isfinite(source_at):
                self._fail_closed(clear_input_time=True)
                return False
        try:
            width, height = screen_size or self.screen_size()
            width, height = int(width), int(height)
        except (TypeError, ValueError, OverflowError):
            self._fail_closed(clear_input_time=True)
            return False
        if width <= 0 or height <= 0:
            self._fail_closed(clear_input_time=True)
            return False
        with self._lock:
            if source_at is None:
                source_at = self._input_time
                if source_at is None:
                    source_at = self._clock_value()
            if source_at is None or not math.isfinite(source_at):
                self._active = False
                self._freshness_deadline = 0.0
                self._freshness_input_at = None
                self._input_time = None
                return False
            self._target[0] = clamp(self._target[0] + dx, 0, width - 1)
            self._target[1] = clamp(self._target[1] + dy, 0, height - 1)
            self._target_generation += 1
            self._target_source_time = source_at
            return True

    def start(self):
        if self.running:
            return
        with self._lock:
            self._last_error = ""
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
        try:
            last = self._clock_value()
            if last is None:
                with self._lock:
                    self._active = False
                return
            while not self._stop_event.is_set():
                now = self._clock_value()
                if now is None:
                    with self._lock:
                        self._active = False
                    return
                dt = max(now - last, 1.0 / 500.0)
                last = now
                with self._lock:
                    active = self._active
                    tx, ty = self._target
                    generation = self._target_generation
                    source_at = self._target_source_time
                    if active and self._lease_expired_locked(now):
                        self._active = False
                        active = False
                if active:
                    x, y = self.position()
                    after_read = self._clock_value()
                    with self._lock:
                        if (
                            not self._active or
                            self._lease_expired_locked(after_read)
                        ):
                            self._active = False
                            active = False
                    if not active:
                        self._stop_event.wait(1.0 / self._output_hz)
                        continue
                    alpha = 1.0 - math.exp(-dt / self._interp_tau)
                    nx = x + (tx - x) * alpha
                    ny = y + (ty - y) * alpha
                    if abs(tx - x) > 0.5 or abs(ty - y) > 0.5:
                        self.set_position(nx, ny)
                        sent_at = self._clock_value()
                        with self._lock:
                            if (
                                sent_at is not None and
                                source_at is not None and
                                generation > self._latency_recorded_generation
                            ):
                                latency_ms = max((sent_at - source_at) * 1000.0, 0.0)
                                if math.isfinite(latency_ms):
                                    self._latency_samples.append(latency_ms)
                                    self._latency_total_samples += 1
                                    self._latency_last_ms = latency_ms
                                    if self._latency_total_samples == 1:
                                        self._latency_ema_ms = latency_ms
                                    else:
                                        self._latency_ema_ms = (
                                            self._latency_ema_ms * 0.88 +
                                            latency_ms * 0.12
                                        )
                                    self._latency_recorded_generation = generation
                self._stop_event.wait(1.0 / self._output_hz)
        except Exception as exc:
            with self._lock:
                self._active = False
                self._last_error = f"{type(exc).__name__}: {exc}"[:140]

    def output_latency(self):
        with self._lock:
            samples = self._latency_total_samples
            last_ms = self._latency_last_ms
            ema_ms = self._latency_ema_ms
            history = tuple(self._latency_samples)
        p50_ms, p95_ms, p99_ms = percentile_metric(history)
        return PerfMetric(
            samples, last_ms, ema_ms, p50_ms, p95_ms, p99_ms,
        )
