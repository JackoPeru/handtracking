"""Private, deterministic JSONL traces for the hand-tracking runtime.

Traces contain sensor/classifier inputs and external command observations.  They
never contain images, window titles, exception text, or credentials.  Replay
uses the real runtime loop and replaces only camera, MediaPipe, cursor, CV, and
native side-effect boundaries.
"""

from __future__ import annotations

import importlib
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from handtracking_settings import RuntimeSettings
from handtracking_config import MP_RESULT_STALE_SECONDS


SCHEMA_VERSION = 2
MAX_TRACE_FRAMES = 18000
DEFAULT_TRACE_FRAMES = 1800
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_TRACE_LINE_BYTES = 4 * 1024 * 1024
MAX_TRACE_COMMANDS = 256
MAX_TRACE_READS = 256
MAX_TRACE_CV_EVENTS = 32
MAX_TRACE_POINTS = 512
_VALID_SOURCES = {"recorded", "synthetic"}
_VALID_MODES = {
    "LOCKED", "MOUSE", "PINCH", "POINTER", "SCROLL", "VOLUME", "RADIAL",
    "TWO_HAND", "SWIPE", "FIST", "SPOCK",
}
_VALID_RADIAL_ACTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}

_SETTINGS_KEYS = {"camera_index", "profile", "sensitivity", "pinch_on", "pinch_off"}
_HEADER_KEYS = {
    "type", "schemaVersion", "source", "settings", "screen_size",
    "started_at", "initial_cursor", "initial_volume",
    "initial_commands_enabled", "camera_target_fps",
}
_FRAME_KEYS = {
    "type", "index", "now", "worker", "cursor", "motion", "cv",
    "commands", "volume_reads", "volume_sets", "mode",
}
_WORKER_KEYS = {
    "seq", "input_seq", "overwrites", "error_count", "alive",
    "last_success_at", "last_result_input_at", "latest",
}
_PACKET_KEYS = {
    "seq", "result", "result_gray_present", "infer_ms", "worker_ms",
    "cycle_ms", "queue_ms",
}
_RESULT_KEYS = {"hand_landmarks", "hand_world_landmarks", "handedness"}
_MOTION_KEYS = {"dx", "dy", "magnitude", "next_points"}
_CURSOR_KEYS = {
    "start", "reads", "active_reads", "lease_accepted", "safety_checks",
}
_CV_KEYS = {"reanchors"}
_CV_EVENT_KEYS = {"corrected", "points", "active", "prev_gray"}


class TraceError(ValueError):
    """Base error for invalid or unusable traces."""


class TraceValidationError(TraceError):
    """Live recorder input violates the trace contract."""


class TraceFormatError(TraceError):
    """Stored JSONL does not satisfy the trace contract."""


def _error(message, *, format_error=False):
    raise (TraceFormatError if format_error else TraceValidationError)(message)


def _finite(value, name, *, format_error=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _error(f"{name} must be finite", format_error=format_error)
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError):
        _error(f"{name} must be finite", format_error=format_error)
    if not math.isfinite(value):
        _error(f"{name} must be finite", format_error=format_error)
    return value


def _integer(value, name, *, minimum=None, maximum=None, format_error=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _error(f"{name} must be an integer", format_error=format_error)
    value = int(value)
    if minimum is not None and value < minimum:
        _error(f"{name} below bound", format_error=format_error)
    if maximum is not None and value > maximum:
        _error(f"{name} above bound", format_error=format_error)
    return value


def _pair(value, name, *, format_error=False):
    if isinstance(value, (str, bytes)):
        _error(f"{name} must contain two numbers", format_error=format_error)
    try:
        values = list(value)
    except (TypeError, ValueError):
        _error(f"{name} must contain two numbers", format_error=format_error)
    if len(values) != 2:
        _error(f"{name} must contain two numbers", format_error=format_error)
    return [
        _finite(values[0], f"{name}[0]", format_error=format_error),
        _finite(values[1], f"{name}[1]", format_error=format_error),
    ]


def _strict_keys(value, expected, name, *, format_error=False):
    if not isinstance(value, Mapping):
        _error(f"{name} must be an object", format_error=format_error)
    unknown = set(value) - expected
    if unknown:
        joined = ", ".join(sorted(map(str, unknown)))
        _error(f"{name} contains unknown key(s): {joined}", format_error=format_error)


def _exact_keys(value, expected, name, *, format_error=False):
    _strict_keys(value, expected, name, format_error=format_error)
    missing = set(expected) - set(value)
    if missing:
        joined = ", ".join(sorted(map(str, missing)))
        _error(f"{name} missing key(s): {joined}", format_error=format_error)


def _validate_json_values(value, name="trace", *, format_error=False, depth=0):
    if depth > 16:
        _error(f"{name} is too deeply nested", format_error=format_error)
    if isinstance(value, float) and not math.isfinite(value):
        _error(f"{name} must contain finite values", format_error=format_error)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _error(f"{name} contains non-string key", format_error=format_error)
            _validate_json_values(
                item, f"{name}.{key}", format_error=format_error, depth=depth + 1
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_values(
                item, f"{name}[{index}]", format_error=format_error, depth=depth + 1
            )


def _as_mapping(value):
    if isinstance(value, Mapping):
        return value
    return {key: getattr(value, key) for key in _SETTINGS_KEYS if hasattr(value, key)}


def _settings_payload(settings):
    try:
        return asdict(RuntimeSettings(**_as_mapping(settings)))
    except (TypeError, ValueError) as exc:
        raise TraceValidationError("Invalid runtime settings") from exc


def _landmark_payload(hand, name):
    try:
        points = list(hand)
    except (TypeError, ValueError):
        _error(f"{name} must contain landmarks")
    if len(points) != 21:
        _error(f"{name} must contain exactly 21 landmarks")
    result = []
    for index, point in enumerate(points):
        try:
            values = [point.x, point.y, point.z]
        except AttributeError:
            try:
                values = list(point)
            except (TypeError, ValueError):
                _error(f"{name}[{index}] must contain x/y/z")
        if len(values) != 3:
            _error(f"{name}[{index}] must contain x/y/z")
        result.append([
            _finite(values[0], f"{name}[{index}][0]"),
            _finite(values[1], f"{name}[{index}][1]"),
            _finite(values[2], f"{name}[{index}][2]"),
        ])
    return result


def _hands_payload(hands, name):
    if hands is None:
        return []
    try:
        hands = list(hands)
    except (TypeError, ValueError):
        _error(f"{name} must be a list")
    if len(hands) > 2:
        _error(f"{name} supports at most two hands")
    return [_landmark_payload(hand, f"{name}[{index}]") for index, hand in enumerate(hands)]


def _handedness_payload(value):
    if value is None:
        return []
    try:
        value = list(value)
    except (TypeError, ValueError):
        _error("handedness must be a list")
    if len(value) > 2:
        _error("handedness supports at most two hands")
    result = []
    for hand_index, entries in enumerate(value):
        try:
            entries = list(entries)
        except (TypeError, ValueError):
            _error(f"handedness[{hand_index}] must be a list")
        if len(entries) > 1:
            _error(f"handedness[{hand_index}] supports one classification")
        encoded = []
        for entry in entries:
            if isinstance(entry, Mapping):
                category = entry.get("category_name", "")
            else:
                category = getattr(entry, "category_name", "")
            if not isinstance(category, str) or category not in ("Left", "Right", ""):
                _error(f"handedness[{hand_index}] category invalid")
            encoded.append(category)
        result.append(encoded)
    return result


def _result_payload(result):
    if result is None:
        return {"hand_landmarks": [], "hand_world_landmarks": [], "handedness": []}
    get = result.get if isinstance(result, Mapping) else lambda key, default=None: getattr(result, key, default)
    return {
        "hand_landmarks": _hands_payload(get("hand_landmarks", []), "hand_landmarks"),
        "hand_world_landmarks": _hands_payload(
            get("hand_world_landmarks", []), "hand_world_landmarks"
        ),
        "handedness": _handedness_payload(get("handedness", [])),
    }


def _point_array(points, name, *, format_error=False):
    try:
        array = np.asarray(points)
    except (OverflowError, TypeError, ValueError):
        _error(f"{name} must contain numeric points", format_error=format_error)
    if (array.dtype.kind not in "iuf" or array.ndim != 3 or
            array.shape[1:] != (1, 2) or array.shape[0] < 3 or
            array.size > MAX_TRACE_POINTS):
        _error(f"{name} must be numeric Nx1x2 tracks, N >= 3", format_error=format_error)
    if not isinstance(points, np.ndarray):
        # Check before coercion: mixed JSON bool/float arrays become float dtype.
        for point in points:
            for coordinate in point[0]:
                _finite(coordinate, name, format_error=format_error)
    if not np.isfinite(array).all():
        _error(f"{name} must contain finite values", format_error=format_error)
    return array


def _points_payload(points, name="points"):
    return None if points is None else _point_array(points, name).tolist()


def _packet_payload(packet):
    if packet is None:
        return None
    try:
        values = tuple(packet)
    except (TypeError, ValueError):
        _error("worker.latest must be a packet")
    if len(values) != 7:
        _error("worker.latest packet must contain seven values")
    sequence = _integer(values[0], "packet.seq", minimum=0)
    result, result_gray = values[1], values[2]
    return {
        "seq": sequence,
        "result": _result_payload(result),
        "result_gray_present": result_gray is not None,
        "infer_ms": _finite(values[3], "packet.infer_ms"),
        "worker_ms": _finite(values[4], "packet.worker_ms"),
        "cycle_ms": _finite(values[5], "packet.cycle_ms"),
        "queue_ms": _finite(values[6], "packet.queue_ms"),
    }


def _relative_timestamp(value, started_at, name):
    if value is None:
        return None
    return _finite(value, name) - started_at


def _worker_payload(state, started_at):
    if not isinstance(state, Mapping):
        _error("mp_state must be an object")
    latest = state.get("latest")
    return {
        "seq": _integer(state.get("seq", 0), "worker.seq", minimum=0),
        "input_seq": _integer(state.get("input_seq", 0), "worker.input_seq", minimum=0),
        "overwrites": _integer(state.get("overwrites", 0), "worker.overwrites", minimum=0),
        "error_count": _integer(state.get("error_count", 0), "worker.error_count", minimum=0),
        "alive": bool(state.get("alive", True)),
        "last_success_at": _relative_timestamp(
            state.get("last_success_at"), started_at, "worker.last_success_at"
        ),
        "last_result_input_at": _relative_timestamp(
            state.get("last_result_input_at"), started_at,
            "worker.last_result_input_at",
        ),
        "latest": _packet_payload(latest),
    }


def _motion_payload(motion):
    if motion is None:
        return None
    try:
        dx, dy, magnitude, next_points = (
            motion.dx, motion.dy, motion.magnitude, motion.next_points
        )
    except AttributeError:
        _error("motion must be LKMotion or None")
    if next_points is None:
        _error("motion.next_points must contain tracks")
    return {
        "dx": _finite(dx, "motion.dx"),
        "dy": _finite(dy, "motion.dy"),
        "magnitude": _finite(magnitude, "motion.magnitude"),
        "next_points": _points_payload(next_points, "motion.next_points"),
    }


def _json_line(record):
    _validate_json_values(record)
    try:
        encoded = json.dumps(
            record, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise TraceValidationError(f"trace record is not JSON-safe: {type(exc).__name__}") from exc
    if len(encoded) > MAX_TRACE_LINE_BYTES:
        raise TraceValidationError("trace line exceeds size bound")
    return encoded


def _command(kind, **values):
    command = {"kind": kind, **values}
    _validate_json_values(command)
    return command


class _CursorProxy:
    def __init__(self, recorder, cursor):
        self._recorder = recorder
        self._cursor = cursor

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def position(self):
        value = self._cursor.position()
        self._recorder._record_cursor_read(value)
        return value

    def renew_freshness(self, input_at):
        result = self._cursor.renew_freshness(input_at)
        self._recorder._record_cursor_lease(result is not False)
        return result

    def check_freshness(self, input_at):
        result = self._cursor.check_freshness(input_at)
        self._recorder._record_safety_check(result is not False)
        return result

    @property
    def active(self):
        value = bool(self._cursor.active)
        self._recorder._record_cursor_active(value)
        return value

    def add_delta(self, dx, dy, *args, **kwargs):
        result = self._cursor.add_delta(dx, dy, *args, **kwargs)
        self._recorder._record_command(
            _command("move", dx=_finite(dx, "move.dx"), dy=_finite(dy, "move.dy"))
        )
        return result

    def set_position(self, x, y, *args, **kwargs):
        result = self._cursor.set_position(x, y, *args, **kwargs)
        self._recorder._record_command(
            _command("set_position", x=_finite(x, "set_position.x"),
                     y=_finite(y, "set_position.y"))
        )
        return result


class TraceRecorder:
    """Record completed runtime frames to a bounded JSONL file.

    Recording always requires an explicit path and opens it in exclusive-create
    mode; replay uses a separate in-memory driver.
    """

    def __init__(
        self,
        path,
        *,
        settings,
        screen_size,
        started_at,
        initial_cursor,
        initial_volume=0.5,
        initial_commands_enabled=False,
        camera_target_fps=60,
        max_frames=DEFAULT_TRACE_FRAMES,
        source="recorded",
    ):
        self.path = Path(path)
        self.settings = _settings_payload(settings)
        self.screen_size = [
            _integer(screen_size[0], "screen_size.width", minimum=1, maximum=100000),
            _integer(screen_size[1], "screen_size.height", minimum=1, maximum=100000),
        ]
        self.started_at = _finite(started_at, "started_at")
        self.initial_cursor = _pair(initial_cursor, "initial_cursor")
        self.initial_volume = _finite(initial_volume, "initial_volume")
        if not 0.0 <= self.initial_volume <= 1.0:
            _error("initial_volume outside range")
        if not isinstance(initial_commands_enabled, bool):
            _error("initial_commands_enabled must be boolean")
        self.initial_commands_enabled = initial_commands_enabled
        self.camera_target_fps = _integer(
            camera_target_fps, "camera_target_fps", minimum=1, maximum=1000
        )
        self.max_frames = _integer(
            max_frames, "max_frames", minimum=1, maximum=MAX_TRACE_FRAMES
        )
        if not isinstance(source, str) or source not in _VALID_SOURCES:
            _error("source invalid")
        self.source = source
        self._frame = None
        self._frame_count = 0
        self._last_now = None
        self._recording = True
        self._closed = False
        self._file = None
        self._bytes_written = 0
        header = {
            "type": "header",
            "schemaVersion": SCHEMA_VERSION,
            "source": self.source,
            "settings": self.settings,
            "screen_size": self.screen_size,
            "started_at": 1.0,
            "initial_cursor": self.initial_cursor,
            "initial_volume": self.initial_volume,
            "initial_commands_enabled": self.initial_commands_enabled,
            "camera_target_fps": self.camera_target_fps,
        }
        self._header_bytes = _json_line(header)
        if self._recording:
            if len(self._header_bytes) > MAX_TRACE_BYTES:
                raise TraceValidationError("trace exceeds size bound")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("x", encoding="utf-8", newline="\n")
                self._file.write(self._header_bytes.decode("utf-8"))
                self._file.flush()
                self._bytes_written = len(self._header_bytes)
            except Exception:
                if self._file is not None:
                    self._file.close()
                self._file = None
                raise

    @property
    def recording(self):
        return self._recording and not self._closed

    def _record_command(self, command):
        if self._frame is not None and self.recording:
            self._frame["commands"].append(command)

    def _record_cursor_read(self, value):
        if self._frame is None or not self.recording:
            return
        self._frame["cursor"]["reads"].append(_pair(value, "cursor.position"))

    def _record_cursor_active(self, value):
        if self._frame is not None and self.recording:
            self._frame["cursor"]["active_reads"].append(bool(value))

    def _record_cursor_lease(self, accepted):
        if self._frame is not None and self.recording:
            self._frame["cursor"]["lease_accepted"] = bool(accepted)

    def _record_safety_check(self, accepted):
        if self._frame is not None and self.recording:
            self._frame["cursor"]["safety_checks"].append(bool(accepted))

    def start_frame(self, now, mp_state, cursor_position):
        if not self.recording or self._frame_count >= self.max_frames:
            self._recording = False
            return False
        if self._frame is not None:
            _error("previous trace frame was not completed")
        now = _finite(now, "frame.now")
        if now < self.started_at:
            _error("frame timestamp precedes started_at")
        relative_now = now - self.started_at
        if self._last_now is not None and relative_now <= self._last_now:
            _error("frame timestamps must be strictly monotonic")
        worker = _worker_payload(mp_state, self.started_at)
        for key in ("last_result_input_at",):
            value = worker[key]
            if value is not None and value > relative_now:
                _error(f"worker.{key} is newer than frame timestamp")
        self._last_now = relative_now
        self._frame = {
            "type": "frame",
            "index": self._frame_count,
            "now": relative_now,
            "worker": worker,
            "cursor": {
                "start": _pair(cursor_position, "cursor.start"),
                "reads": [],
                "active_reads": [],
                "lease_accepted": None,
                "safety_checks": [],
            },
            "motion": None,
            "cv": {"reanchors": []},
            "commands": [],
            "volume_reads": [],
            "volume_sets": [],
            "mode": None,
        }
        return True

    def set_motion(self, motion):
        if self._frame is not None and self.recording:
            self._frame["motion"] = _motion_payload(motion)

    def _reanchor(self, session, control_hand, result_gray, gray):
        modes = importlib.import_module("handtracking_modes")
        corrected = modes.reanchor_flow(session, control_hand, result_gray, gray)
        if self._frame is not None and self.recording:
            self._frame["cv"]["reanchors"].append({
                "corrected": _points_payload(corrected, "cv.corrected"),
                "points": _points_payload(session.flow.points, "cv.points"),
                "active": bool(session.flow.active),
                "prev_gray": session.flow.prev_gray is not None,
            })
        return corrected

    def process_packet(self, session, packet, **kwargs):
        frame_processor = importlib.import_module("handtracking_frame")
        process = frame_processor.process_mediapipe_packet
        kwargs = dict(kwargs)
        kwargs.update(
            ctrl_wheel_cb=self.ctrl_wheel,
            execute_radial_action_cb=self.execute_radial_action,
            left_click_cb=self.left_click,
            get_volume_cb=self.get_system_volume,
            set_volume_cb=self.set_system_volume,
        )
        kwargs["reanchor_cb"] = self._reanchor
        return process(session, packet, **kwargs)

    def wrap_cursor(self, cursor):
        return _CursorProxy(self, cursor)

    def execute_swipe(self, direction):
        if direction not in ("LEFT", "RIGHT"):
            _error("swipe direction invalid")
        result = _windows_call("execute_swipe", direction)
        self._record_command(_command("swipe", direction=direction))
        return result

    def mouse_wheel(self, delta):
        delta = _integer(delta, "wheel.delta", minimum=-1000000, maximum=1000000)
        result = _windows_call("mouse_wheel", delta)
        self._record_command(_command("wheel", delta=delta))
        return result

    def ctrl_wheel(self, delta):
        delta = _integer(delta, "ctrl_wheel.delta", minimum=-1000000, maximum=1000000)
        result = _windows_call("ctrl_wheel", delta)
        self._record_command(_command("ctrl_wheel", delta=delta))
        return result

    def execute_radial_action(self, action):
        if not isinstance(action, str) or len(action) > 16:
            _error("radial action invalid")
        result = _windows_call("execute_radial_action", action)
        self._record_command(_command("radial", action=action))
        return result

    def left_click(self):
        result = _windows_call("left_click")
        self._record_command(_command("click"))
        return result

    def get_system_volume(self):
        value = _windows_call("get_system_volume")
        value = _finite(value, "volume.read")
        if not 0.0 <= value <= 1.0:
            _error("volume.read outside range")
        if self._frame is not None and self.recording:
            self._frame["volume_reads"].append(value)
        return value

    def set_system_volume(self, value):
        value = _finite(value, "volume.value")
        if not 0.0 <= value <= 1.0:
            _error("volume.value outside range")
        result = _windows_call("set_system_volume", value)
        success = result is not False
        if self._frame is not None and self.recording:
            self._frame["volume_sets"].append({"value": value, "success": success})
        self._record_command(_command("volume", value=value, success=success))
        return result

    def finish_frame(self, session):
        if self._frame is None:
            return False
        if not self.recording:
            self._frame = None
            return False
        mode = getattr(session, "gesture_mode", None)
        if not isinstance(mode, str) or len(mode) > 32:
            _error("gesture_mode invalid")
        self._frame["mode"] = mode
        data = _json_line(self._frame)
        if self._bytes_written + len(data) > MAX_TRACE_BYTES:
            self._recording = False
            self._frame = None
            return False
        self._file.write(data.decode("utf-8"))
        self._file.flush()
        self._bytes_written += len(data)
        self._frame_count += 1
        self._frame = None
        if self._frame_count >= self.max_frames:
            self._recording = False
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._frame = None
        self._recording = False
        if self._file is not None:
            self._file.close()
            self._file = None


def _windows_call(name, *args):
    # Resolve at call time: tests can replace only the native boundary.
    windows = importlib.import_module("handtracking_windows")
    return getattr(windows, name)(*args)


def _read_trace(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_TRACE_BYTES + 1)
    except OSError:
        raise
    if len(data) > MAX_TRACE_BYTES:
        raise TraceFormatError("trace exceeds size bound")
    lines = data.splitlines()
    if not lines:
        raise TraceFormatError("trace is empty")
    if len(lines) > MAX_TRACE_FRAMES + 1:
        raise TraceFormatError("trace contains too many frames")
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise TraceFormatError(f"trace line {line_number} is empty")
        if len(line) > MAX_TRACE_LINE_BYTES:
            raise TraceFormatError(f"trace line {line_number} exceeds size bound")
        try:
            record = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise TraceFormatError(f"invalid JSON at line {line_number}") from exc
        _validate_json_values(record, f"line {line_number}", format_error=True)
        records.append(record)
    if records and records[0].get("schemaVersion") == 1:
        for frame in records[1:]:
            cursor = frame.get("cursor")
            if isinstance(cursor, dict):
                accepted = cursor.get("lease_accepted")
                checks = [] if accepted is None else [bool(accepted)]
                if accepted is not None and frame.get("motion") is not None:
                    checks.append(bool(accepted))
                callback_checks = len(frame.get("volume_reads", []))
                callback_checks += len(frame.get("volume_sets", []))
                callback_checks += sum(
                    2 if command.get("kind") == "click" else 1
                    for command in frame.get("commands", [])
                    if isinstance(command, dict) and
                    command.get("kind") in {"click", "ctrl_wheel", "radial"}
                )
                checks.extend([bool(accepted)] * callback_checks)
                cursor.setdefault("safety_checks", checks)
    _validate_records(records)
    return records[0], records[1:]


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_settings_payload(settings):
    _exact_keys(settings, _SETTINGS_KEYS, "settings", format_error=True)
    _runtime_settings(settings)


def _validate_hands(value, name):
    if not isinstance(value, list) or len(value) > 2:
        _error(f"{name} invalid", format_error=True)
    for hand_index, hand in enumerate(value):
        if not isinstance(hand, list) or len(hand) != 21:
            _error(f"{name}[{hand_index}] invalid", format_error=True)
        for point_index, point in enumerate(hand):
            if not isinstance(point, list) or len(point) != 3:
                _error(f"{name}[{hand_index}][{point_index}] invalid", format_error=True)
            for axis, item in enumerate(point):
                _finite(item, f"{name}[{hand_index}][{point_index}][{axis}]", format_error=True)


def _validate_result_payload(value, name="result"):
    _exact_keys(value, _RESULT_KEYS, name, format_error=True)
    _validate_hands(value["hand_landmarks"], f"{name}.hand_landmarks")
    _validate_hands(value["hand_world_landmarks"], f"{name}.hand_world_landmarks")
    handedness = value["handedness"]
    if not isinstance(handedness, list) or len(handedness) > 2:
        _error(f"{name}.handedness invalid", format_error=True)
    for entries in handedness:
        if not isinstance(entries, list) or len(entries) > 1:
            _error(f"{name}.handedness invalid", format_error=True)
        for category in entries:
            if not isinstance(category, str) or category not in ("Left", "Right", ""):
                _error(f"{name}.handedness category invalid", format_error=True)


def _validate_packet_payload(value):
    if value is None:
        return
    _exact_keys(value, _PACKET_KEYS, "worker.latest", format_error=True)
    _integer(value["seq"], "packet.seq", minimum=0, format_error=True)
    _validate_result_payload(value["result"])
    if not isinstance(value["result_gray_present"], bool):
        _error("packet.result_gray_present invalid", format_error=True)
    for key in ("infer_ms", "worker_ms", "cycle_ms", "queue_ms"):
        _finite(value[key], f"packet.{key}", format_error=True)


def _validate_points(value, name):
    if value is None:
        return
    _point_array(value, name, format_error=True)


def _validate_records(records):
    if len(records) - 1 > MAX_TRACE_FRAMES:
        _error("trace contains too many frames", format_error=True)
    header = records[0]
    _exact_keys(header, _HEADER_KEYS, "header", format_error=True)
    if header.get("type") != "header":
        _error("unsupported trace schema", format_error=True)
    _integer(header.get("schemaVersion"), "header.schemaVersion", minimum=1, maximum=SCHEMA_VERSION, format_error=True)
    source = header.get("source")
    if not isinstance(source, str) or source not in _VALID_SOURCES:
        _error("header.source invalid", format_error=True)
    _validate_settings_payload(header.get("settings"))
    screen = header.get("screen_size")
    if not isinstance(screen, list) or len(screen) != 2:
        _error("header.screen_size invalid", format_error=True)
    _integer(screen[0], "header.screen_size[0]", minimum=1, maximum=100000, format_error=True)
    _integer(screen[1], "header.screen_size[1]", minimum=1, maximum=100000, format_error=True)
    _finite(header.get("started_at"), "header.started_at", format_error=True)
    _pair(header.get("initial_cursor"), "header.initial_cursor", format_error=True)
    volume = _finite(header.get("initial_volume"), "header.initial_volume", format_error=True)
    if not 0.0 <= volume <= 1.0:
        _error("header.initial_volume outside range", format_error=True)
    if not isinstance(header.get("initial_commands_enabled"), bool):
        _error("header.initial_commands_enabled invalid", format_error=True)
    _integer(header.get("camera_target_fps"), "header.camera_target_fps", minimum=1, maximum=1000, format_error=True)
    previous_now = None
    for expected_index, frame in enumerate(records[1:]):
        _exact_keys(frame, _FRAME_KEYS, f"frame[{expected_index}]", format_error=True)
        if frame.get("type") != "frame":
            _error("frame index/type invalid", format_error=True)
        _integer(frame.get("index"), "frame.index", minimum=expected_index, maximum=expected_index, format_error=True)
        now = _finite(frame.get("now"), f"frame[{expected_index}].now", format_error=True)
        if now < 0.0 or (previous_now is not None and now <= previous_now):
            _error("frame timestamps must be strictly monotonic", format_error=True)
        previous_now = now
        worker = frame.get("worker")
        _exact_keys(worker, _WORKER_KEYS, "frame.worker", format_error=True)
        for key in ("seq", "input_seq", "overwrites", "error_count"):
            _integer(worker[key], f"worker.{key}", minimum=0, format_error=True)
        if not isinstance(worker["alive"], bool):
            _error("worker.alive invalid", format_error=True)
        for key in ("last_success_at", "last_result_input_at"):
            if worker[key] is not None:
                _finite(worker[key], f"worker.{key}", format_error=True)
                if key == "last_result_input_at" and worker[key] > now:
                    _error(f"worker.{key} newer than frame", format_error=True)
        _validate_packet_payload(worker["latest"])
        cursor = frame.get("cursor")
        _exact_keys(cursor, _CURSOR_KEYS, "frame.cursor", format_error=True)
        _pair(cursor["start"], "frame.cursor.start", format_error=True)
        if not isinstance(cursor["reads"], list):
            _error("frame.cursor.reads invalid", format_error=True)
        if len(cursor["reads"]) > MAX_TRACE_READS:
            _error("frame.cursor.reads exceeds bounds", format_error=True)
        for read in cursor["reads"]:
            _pair(read, "frame.cursor.read", format_error=True)
        if not isinstance(cursor["active_reads"], list):
            _error("frame.cursor.active_reads invalid", format_error=True)
        if len(cursor["active_reads"]) > MAX_TRACE_READS:
            _error("frame.cursor.active_reads exceeds bounds", format_error=True)
        if not all(isinstance(value, bool) for value in cursor["active_reads"]):
            _error("frame.cursor.active_reads invalid", format_error=True)
        if cursor["lease_accepted"] is not None and not isinstance(cursor["lease_accepted"], bool):
            _error("frame.cursor.lease_accepted invalid", format_error=True)
        if (not isinstance(cursor["safety_checks"], list) or
                not all(isinstance(value, bool) for value in cursor["safety_checks"])):
            _error("frame.cursor.safety_checks invalid", format_error=True)
        if len(cursor["safety_checks"]) > MAX_TRACE_READS:
            _error("frame.cursor.safety_checks exceeds bounds", format_error=True)
        motion = frame.get("motion")
        if motion is not None:
            _exact_keys(motion, _MOTION_KEYS, "frame.motion", format_error=True)
            for key in ("dx", "dy", "magnitude"):
                _finite(motion[key], f"motion.{key}", format_error=True)
            if motion["next_points"] is None:
                _error("motion.next_points must contain tracks", format_error=True)
            _validate_points(motion["next_points"], "motion.next_points")
        cv = frame.get("cv")
        _exact_keys(cv, _CV_KEYS, "frame.cv", format_error=True)
        if not isinstance(cv["reanchors"], list):
            _error("frame.cv.reanchors invalid", format_error=True)
        if len(cv["reanchors"]) > MAX_TRACE_CV_EVENTS:
            _error("frame.cv.reanchors exceeds bounds", format_error=True)
        for event in cv["reanchors"]:
            _exact_keys(event, _CV_EVENT_KEYS, "frame.cv.reanchor", format_error=True)
            _validate_points(event["corrected"], "cv.corrected")
            _validate_points(event["points"], "cv.points")
            if not isinstance(event["active"], bool) or not isinstance(event["prev_gray"], bool):
                _error("cv state invalid", format_error=True)
        commands = frame.get("commands")
        if not isinstance(commands, list):
            _error("frame.commands invalid", format_error=True)
        if len(commands) > MAX_TRACE_COMMANDS:
            _error("frame.commands exceeds bounds", format_error=True)
        for command in commands:
            if not isinstance(command, Mapping) or "kind" not in command:
                _error("command invalid", format_error=True)
            kind = command["kind"]
            if not isinstance(kind, str):
                _error("command kind invalid", format_error=True)
            allowed = {
                "move": {"kind", "dx", "dy"},
                "set_position": {"kind", "x", "y"},
                "click": {"kind"},
                "wheel": {"kind", "delta"},
                "ctrl_wheel": {"kind", "delta"},
                "swipe": {"kind", "direction"},
                "radial": {"kind", "action"},
                "volume": {"kind", "value", "success"},
            }.get(kind)
            if allowed is None:
                _error("unknown command kind", format_error=True)
            _exact_keys(command, allowed, "command", format_error=True)
            if kind in ("move", "set_position", "volume"):
                for key in (set(command) - {"kind", "success"}):
                    _finite(command[key], f"command.{key}", format_error=True)
            if kind in ("wheel", "ctrl_wheel"):
                _integer(command["delta"], "command.delta", format_error=True)
            if kind == "volume":
                if not 0.0 <= command["value"] <= 1.0 or not isinstance(command["success"], bool):
                    _error("volume command invalid", format_error=True)
            if kind == "swipe" and command["direction"] not in ("LEFT", "RIGHT"):
                _error("swipe command invalid", format_error=True)
            if kind == "radial" and (not isinstance(command["action"], str) or
                                     command["action"] not in _VALID_RADIAL_ACTIONS):
                _error("radial command invalid", format_error=True)
        if not isinstance(frame.get("volume_reads"), list):
            _error("volume_reads invalid", format_error=True)
        if len(frame["volume_reads"]) > MAX_TRACE_READS:
            _error("volume_reads exceeds bounds", format_error=True)
        for value in frame["volume_reads"]:
            value = _finite(value, "volume_read", format_error=True)
            if not 0.0 <= value <= 1.0:
                _error("volume_read outside range", format_error=True)
        if not isinstance(frame.get("volume_sets"), list):
            _error("volume_sets invalid", format_error=True)
        if len(frame["volume_sets"]) > MAX_TRACE_READS:
            _error("volume_sets exceeds bounds", format_error=True)
        for item in frame["volume_sets"]:
            _exact_keys(item, {"value", "success"}, "volume_set", format_error=True)
            value = _finite(item["value"], "volume_set.value", format_error=True)
            if not 0.0 <= value <= 1.0 or not isinstance(item["success"], bool):
                _error("volume_set invalid", format_error=True)
        mode = frame.get("mode")
        if not isinstance(mode, str) or mode not in _VALID_MODES:
            _error("frame.mode invalid", format_error=True)


def _result_from_payload(payload):
    def hands(values):
        return [
            [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in hand]
            for hand in values
        ]

    return SimpleNamespace(
        hand_landmarks=hands(payload["hand_landmarks"]),
        hand_world_landmarks=hands(payload["hand_world_landmarks"]),
        handedness=[
            [SimpleNamespace(category_name=category) for category in entries]
            for entries in payload["handedness"]
        ],
    )


def _packet_from_payload(payload, gray):
    if payload is None:
        return None
    return (
        payload["seq"],
        _result_from_payload(payload["result"]),
        gray if payload["result_gray_present"] else None,
        payload["infer_ms"],
        payload["worker_ms"],
        payload["cycle_ms"],
        payload["queue_ms"],
    )


class _ReplayCamera:
    def __init__(self, count, width, height, target_fps):
        self._remaining = count
        self._width = width
        self._height = height
        self.closed = False
        self.target_fps = target_fps
        self.reported_w = width
        self.reported_h = height
        self.reported_fps = float(target_fps)
        self.codec = "TRACE"
        self._gray = np.zeros((1, 1), dtype=np.uint8)

    def read_frame(self):
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def prepare_detection(self, frame):
        return frame, self._gray

    def show(self, frame):
        return self._remaining > 0

    def close(self):
        self.closed = True


class _ReplayWorker:
    def __init__(self, driver, frames, gray, epoch):
        self.driver = driver
        self.frames = frames
        self.gray = gray
        self.epoch = epoch
        self.index = 0
        self.stopped = False
        self.joined = False

    def snapshot_state(self):
        if self.index >= len(self.frames):
            return {
                "latest": None, "seq": 0, "input_seq": 0, "overwrites": 0,
                "error_count": 0, "last_error": "", "last_success_at": None,
                "last_result_input_at": None, "alive": True,
            }
        frame = self.frames[self.index]
        self.driver.prepare_frame(self.index)
        self.index += 1
        worker = frame["worker"]
        return {
            "latest": _packet_from_payload(worker["latest"], self.gray),
            "seq": worker["seq"],
            "input_seq": worker["input_seq"],
            "overwrites": worker["overwrites"],
            "error_count": worker["error_count"],
            "last_error": "",
            "last_success_at": None if worker["last_success_at"] is None else self.epoch + worker["last_success_at"],
            "last_result_input_at": None if worker["last_result_input_at"] is None else self.epoch + worker["last_result_input_at"],
            "alive": worker["alive"],
        }

    def submit(self, frame, gray, timestamp_ms, enqueued_at):
        return None

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True


class _ReplayCursor:
    def __init__(self, driver, width, height, initial):
        self.driver = driver
        self.width = width
        self.height = height
        self._position = [float(initial[0]), float(initial[1])]
        self._target = list(self._position)
        self._active = False
        self.running = True
        self.last_error = ""
        self.closed = False
        self._deadline = None
        self._input_time = None

    @property
    def active(self):
        if self._deadline is not None and self.driver.current_now >= self._deadline:
            self._active = False
        return self.driver.cursor_active(self._active)

    @property
    def target(self):
        return tuple(self._target)

    def screen_size(self):
        return self.width, self.height

    def position(self):
        return self.driver.cursor_position(self._position)

    def set_position(self, x, y, *args, **kwargs):
        self._position[:] = [float(x), float(y)]
        return True

    def sync(self, active):
        active = bool(active)
        if active and self._deadline is not None and self.driver.current_now >= self._deadline:
            self._active = False
            return None
        self._active = active
        if self._active:
            self._target[:] = self._position
        return None

    def add_delta(self, dx, dy, *, screen_size=None, source_at=None):
        width, height = screen_size or (self.width, self.height)
        self._target[0] = max(0.0, min(float(width) - 1.0, self._target[0] + float(dx)))
        self._target[1] = max(0.0, min(float(height) - 1.0, self._target[1] + float(dy)))
        return True

    def set_input_time(self, value):
        value = float(value)
        if self._input_time is not None and value < self._input_time:
            return False
        self._input_time = value
        return True

    def renew_freshness(self, value):
        value = float(value)
        deadline = value + MP_RESULT_STALE_SECONDS
        accepted = not (
            value > self.driver.current_now or
            self._deadline is not None and value < self._deadline - MP_RESULT_STALE_SECONDS or
            deadline <= self.driver.current_now
        )
        if self.driver.current is not None:
            observed = self.driver.current["expected"]["cursor"]["lease_accepted"]
            if observed is not None:
                accepted = observed
        if not accepted:
            self._active = False
            self._deadline = 0.0
            return False
        self._deadline = deadline
        return True

    def check_freshness(self, value):
        if self.driver.current is None:
            return False
        expected = self.driver.current["expected"]["cursor"]["safety_checks"]
        index = self.driver.current["safety_index"]
        accepted = expected[index] if index < len(expected) else False
        if not accepted:
            self._active = False
            self._deadline = 0.0
        return accepted

    def start(self):
        return None

    def close(self):
        self.closed = True
        self.running = False
        self._active = False


class _ReplayDriver:
    def __init__(self, header, frames, verify):
        self.header = header
        self.frames = frames
        self.verify = bool(verify)
        self.epoch = 1.0
        self.index = 0
        self.current = None
        self.current_now = self.epoch
        self.modes = []
        self.emitted_commands = []
        self.mismatches = []
        self.cursor = None
        self._gray = np.zeros((1, 1), dtype=np.uint8)

    @property
    def current_now(self):
        return self._current_now

    @current_now.setter
    def current_now(self, value):
        self._current_now = value

    def prepare_frame(self, index):
        if index < len(self.frames) and self.cursor is not None:
            start = self.frames[index]["cursor"]["start"]
            self.cursor._position[:] = start
            self.cursor._target[:] = start

    def now(self):
        if self.index >= len(self.frames):
            return self.epoch + (self.frames[-1]["now"] if self.frames else 0.0) + 0.001
        self.current_now = self.epoch + self.frames[self.index]["now"]
        return self.current_now

    def start_frame(self, now, mp_state, cursor_position):
        if self.index >= len(self.frames):
            self.mismatches.append("runtime produced more frames than trace")
            self.current = None
            return False
        expected = self.frames[self.index]
        self.current = {
            "expected": expected,
            "commands": [],
            "cv": [],
            "reads": [],
            "active_reads": [],
            "lease_accepted": None,
            "safety_checks": [],
            "volume_reads": [],
            "volume_sets": [],
            "cv_index": 0,
            "read_index": 0,
            "active_index": 0,
            "safety_index": 0,
            "volume_read_index": 0,
            "volume_set_index": 0,
        }
        expected_now = self.epoch + expected["now"]
        if not math.isclose(float(now), expected_now, rel_tol=0.0, abs_tol=1e-9):
            self.mismatches.append(
                f"frame {self.index}: timestamp mismatch expected {expected_now:g} got {float(now):g}"
            )
        if not _same(cursor_position, expected["cursor"]["start"]):
            self.mismatches.append(f"frame {self.index}: initial cursor observation mismatch")

    def set_motion(self, motion):
        if self.current is not None and self.verify:
            if not _same(_motion_payload(motion), self.current["expected"]["motion"]):
                self.mismatches.append(f"frame {self.index}: motion outcome mismatch")

    def measure_flow(self, *args, **kwargs):
        if self.current is None:
            return None
        motion = self.current["expected"]["motion"]
        if motion is None:
            return None
        from handtracking_flow import LKMotion
        return LKMotion(
            next_points=np.asarray(motion["next_points"], dtype=np.float32)
            if motion["next_points"] is not None else None,
            dx=motion["dx"], dy=motion["dy"], magnitude=motion["magnitude"],
        )

    def _emit(self, command):
        if self.current is not None:
            self.current["commands"].append(command)
            self.emitted_commands.append(command)

    def process_packet(self, session, packet, **kwargs):
        frame_processor = importlib.import_module("handtracking_frame")
        process = frame_processor.process_mediapipe_packet
        kwargs = dict(kwargs)
        kwargs.update(
            ctrl_wheel_cb=self.ctrl_wheel,
            execute_radial_action_cb=self.execute_radial_action,
            left_click_cb=self.left_click,
            get_volume_cb=self.get_system_volume,
            set_volume_cb=self.set_system_volume,
        )
        kwargs["reanchor_cb"] = self.reanchor
        return process(session, packet, **kwargs)

    def reanchor(self, session, control_hand, result_gray, gray):
        if self.current is None:
            return None
        expected = self.current["expected"]["cv"]["reanchors"]
        index = self.current["cv_index"]
        self.current["cv_index"] += 1
        if index >= len(expected):
            self.mismatches.append(f"frame {self.index}: unexpected reanchor call")
            return None
        event = expected[index]
        session.flow.points = (
            None if event["points"] is None
            else np.asarray(event["points"], dtype=np.float32)
        )
        session.flow.active = event["active"]
        session.flow.prev_gray = gray if event["prev_gray"] else None
        self.current["cv"].append(event)
        return (
            None if event["corrected"] is None
            else np.asarray(event["corrected"], dtype=np.float32)
        )

    def wrap_cursor(self, cursor):
        return _CursorProxy(self, cursor)

    def _record_command(self, command):
        self._emit(command)

    def _record_cursor_read(self, value):
        if self.current is None:
            return
        self.current["reads"].append(_pair(value, "cursor.position"))

    def _record_cursor_active(self, value):
        if self.current is not None:
            self.current["active_reads"].append(bool(value))

    def _record_cursor_lease(self, accepted):
        if self.current is not None:
            self.current["lease_accepted"] = bool(accepted)

    def _record_safety_check(self, accepted):
        if self.current is not None:
            self.current["safety_checks"].append(bool(accepted))
            self.current["safety_index"] += 1

    def cursor_position(self, fallback):
        if self.current is None:
            return tuple(fallback)
        expected_reads = self.current["expected"]["cursor"]["reads"]
        index = self.current["read_index"]
        self.current["read_index"] += 1
        if index < len(expected_reads):
            return tuple(expected_reads[index])
        return tuple(fallback)

    def cursor_active(self, fallback):
        if self.current is None:
            return bool(fallback)
        expected = self.current["expected"]["cursor"]["active_reads"]
        index = self.current["active_index"]
        self.current["active_index"] += 1
        if index < len(expected):
            return expected[index]
        return bool(fallback)

    def execute_swipe(self, direction):
        self._emit(_command("swipe", direction=direction))
        return ""

    def mouse_wheel(self, delta):
        self._emit(_command("wheel", delta=int(delta)))

    def ctrl_wheel(self, delta):
        self._emit(_command("ctrl_wheel", delta=int(delta)))

    def execute_radial_action(self, action):
        self._emit(_command("radial", action=action))
        return ""

    def left_click(self):
        self._emit(_command("click"))

    def get_system_volume(self):
        if self.current is None:
            return self.header["initial_volume"]
        expected = self.current["expected"]["volume_reads"]
        index = self.current["volume_read_index"]
        self.current["volume_read_index"] += 1
        if index >= len(expected):
            self.mismatches.append(f"frame {self.index}: unexpected volume read")
            return self.header["initial_volume"]
        value = expected[index]
        self.current["volume_reads"].append(value)
        return value

    def set_system_volume(self, value):
        if self.current is None:
            return False
        expected = self.current["expected"]["volume_sets"]
        index = self.current["volume_set_index"]
        self.current["volume_set_index"] += 1
        if index >= len(expected):
            self.mismatches.append(f"frame {self.index}: unexpected volume write")
            success = False
        else:
            success = expected[index]["success"]
        value = float(value)
        self.current["volume_sets"].append({"value": value, "success": success})
        self._emit(_command("volume", value=value, success=success))
        return success

    def finish_frame(self, session):
        if self.current is None:
            return False
        expected = self.current["expected"]
        frame_index = self.index
        if self.verify:
            if not _same(self.current["commands"], expected["commands"]):
                self.mismatches.append(
                    f"frame {frame_index}: command mismatch expected={expected['commands']!r} got={self.current['commands']!r}"
                )
            if getattr(session, "gesture_mode", None) != expected["mode"]:
                self.mismatches.append(
                    f"frame {frame_index}: mode mismatch expected={expected['mode']!r} got={getattr(session, 'gesture_mode', None)!r}"
                )
            if not _same(self.current["reads"], expected["cursor"]["reads"]):
                self.mismatches.append(f"frame {frame_index}: cursor read mismatch")
            if not _same(self.current["active_reads"], expected["cursor"]["active_reads"]):
                self.mismatches.append(f"frame {frame_index}: cursor active mismatch")
            if self.current["lease_accepted"] != expected["cursor"]["lease_accepted"]:
                self.mismatches.append(f"frame {frame_index}: cursor lease mismatch")
            if self.current["safety_checks"] != expected["cursor"]["safety_checks"]:
                self.mismatches.append(f"frame {frame_index}: cursor safety mismatch")
            if not _same(self.current["volume_reads"], expected["volume_reads"]):
                self.mismatches.append(f"frame {frame_index}: volume read mismatch")
            if not _same(self.current["volume_sets"], expected["volume_sets"]):
                self.mismatches.append(f"frame {frame_index}: volume write mismatch")
            if not _same(self.current["cv"], expected["cv"]["reanchors"]):
                self.mismatches.append(f"frame {frame_index}: CV outcome mismatch")
            if self.current["cv_index"] != len(expected["cv"]["reanchors"]):
                self.mismatches.append(f"frame {frame_index}: CV callback count mismatch")
            if self.current["read_index"] != len(expected["cursor"]["reads"]):
                self.mismatches.append(f"frame {frame_index}: cursor read count mismatch")
            if self.current["active_index"] != len(expected["cursor"]["active_reads"]):
                self.mismatches.append(f"frame {frame_index}: cursor active count mismatch")
            if self.current["safety_index"] != len(expected["cursor"]["safety_checks"]):
                self.mismatches.append(f"frame {frame_index}: cursor safety count mismatch")
        self.modes.append(getattr(session, "gesture_mode", None))
        self.index += 1
        self.current = None
        return True

    def close(self):
        self.current = None


def _same(left, right, *, tolerance=1e-5):
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b, tolerance=tolerance) for a, b in zip(left, right))
    return left == right


def _runtime_settings(values):
    try:
        return RuntimeSettings(**values)
    except (TypeError, ValueError) as exc:
        raise TraceFormatError("trace settings rejected by RuntimeSettings") from exc


def replay_trace(path, *, verify=True):
    """Run trace through real ``handtracking_runtime._run_impl`` headlessly."""
    header, frames = _read_trace(path)
    runtime = importlib.import_module("handtracking_runtime")
    from handtracking_session import RuntimeSession
    from handtracking_state import VolumeState

    driver = _ReplayDriver(header, frames, verify)
    width, height = header["screen_size"]
    camera = _ReplayCamera(len(frames), width, height, header["camera_target_fps"])
    cursor_impl = _ReplayCursor(driver, width, height, header["initial_cursor"])
    driver.cursor = cursor_impl
    worker = _ReplayWorker(driver, frames, camera._gray, driver.epoch)
    session_kwargs = {
        "camera": camera,
        "worker": worker,
        "cursor": driver.wrap_cursor(cursor_impl),
        "screen_w": width,
        "screen_h": height,
        "start_time": driver.epoch,
        "last_hand_seen": driver.epoch,
        "fps_window_start": driver.epoch,
        "mp_fps_window_start": driver.epoch,
        "camera_target_fps": header["camera_target_fps"],
        "volume": VolumeState(level=header["initial_volume"]),
    }
    try:
        session_kwargs["settings"] = _runtime_settings(header["settings"])
        session = RuntimeSession(**session_kwargs)
    except TypeError as exc:
        raise TraceError("RuntimeSession cannot be constructed for replay") from exc
    session.commands_enabled = header["initial_commands_enabled"]
    session.render_enabled = False
    session.trace = driver
    try:
        runtime._run_impl(
            session,
            now_fn=driver.now,
            measure_flow_fn=driver.measure_flow,
            process_packet_fn=driver.process_packet,
            execute_swipe_fn=driver.execute_swipe,
            mouse_wheel_fn=driver.mouse_wheel,
        )
    except Exception as exc:
        driver.mismatches.append(f"runtime failed: {type(exc).__name__}")
    finally:
        session.close()

    if driver.index != len(frames):
        driver.mismatches.append(
            f"runtime processed {driver.index} of {len(frames)} trace frames"
        )
    return {
        "source": header["source"],
        "frames": len(frames),
        "commands": list(driver.emitted_commands),
        "command_count": len(driver.emitted_commands),
        "modes": list(driver.modes),
        "verified": bool(verify and not driver.mismatches),
        "mismatches": list(driver.mismatches),
    }


__all__ = [
    "MAX_TRACE_BYTES", "MAX_TRACE_FRAMES", "MAX_TRACE_LINE_BYTES",
    "SCHEMA_VERSION", "TraceError", "TraceFormatError", "TraceRecorder",
    "TraceValidationError", "replay_trace",
]
