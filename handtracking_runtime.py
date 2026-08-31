"""Hand tracking runtime implementation."""

import ctypes
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities

from handtracking_core import (
    advance_confirmed_hold,
    choose_camera_target_fps,
    choose_control_index,
    fist_evidence_from_hands,
    normalize_flow_delta,
    normalized_points_pixel_distance,
    palm_motion_scale,
    pointer_mode_allowed,
    spock_release_gate_active,
    tracking_result_is_stale,
)
from handtracking_mediapipe import MediaPipeWorker
from handtracking_config import *
from handtracking_gestures import (
    clamp,
    control_point,
    fist_fold_metrics,
    grip_class_scores,
    is_fist,
    is_open_hand,
    is_pointer_pinch_pose,
    is_radial_open_pose,
    is_scroll_gesture,
    is_strong_fist,
    is_volume_release_pose,
    mouse_point,
    normalized_pinch_ratio,
    palm_roll_angle,
    pointer_other_fingers_valid,
    radial_direction,
    resolve_gesture_mode,
    spock_all_fingers_up,
    spock_pose_score,
    swipe_pose_metrics,
    two_hand_geometry,
    wrapped_angle_delta,
)

# L'optical flow usa pochissimi punti e non beneficia di 8 thread OpenCV.
# Limitarlo evita contesa CPU con MediaPipe/XNNPACK sul portatile 4C/8T.
cv2.setNumThreads(1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
# Optical flow sul palmo: punti rigidi che non si spostano quando pieghi le dita.
# Questo evita il salto del cursore durante click/pinch/drag.

user32 = ctypes.windll.user32
try:
    user32.SetProcessDPIAware()
except Exception:
    pass

SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)
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


def smoothstep01(value):
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def cursor_gain_for_speed(speed_px_s):
    # Tre zone: precisione, movimento normale e flick. Il gain cresce molto
    # solo quando la mano accelera davvero, cosi' le micro-correzioni restano fini.
    if speed_px_s <= CURSOR_SPEED_PRECISION_END:
        t = smoothstep01(speed_px_s / max(CURSOR_SPEED_PRECISION_END, 1.0))
        return CURSOR_GAIN_PRECISION + (CURSOR_GAIN_NORMAL - CURSOR_GAIN_PRECISION) * t
    if speed_px_s <= CURSOR_SPEED_NORMAL_END:
        return CURSOR_GAIN_NORMAL
    t = smoothstep01(
        (speed_px_s - CURSOR_SPEED_NORMAL_END) /
        max(CURSOR_SPEED_FLICK_FULL - CURSOR_SPEED_NORMAL_END, 1.0)
    )
    return CURSOR_GAIN_NORMAL + (CURSOR_GAIN_FLICK - CURSOR_GAIN_NORMAL) * t


def mouse_down():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def mouse_up():
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def mouse_wheel(delta):
    wheel_data = ctypes.c_uint32(int(delta) & 0xFFFFFFFF).value
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_data, 0)


def key_down(vk):
    user32.keybd_event(vk, 0, 0, 0)


def key_up(vk):
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def tap_combo(vk, modifiers=()):
    for mod in modifiers:
        key_down(mod)
    key_down(vk)
    key_up(vk)
    for mod in reversed(modifiers):
        key_up(mod)


def ctrl_wheel(delta):
    key_down(VK_CONTROL)
    try:
        mouse_wheel(delta)
    finally:
        key_up(VK_CONTROL)


def foreground_window_title():
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "?"
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(max(length + 1, 2))
    user32.GetWindowTextW(hwnd, buf, len(buf))
    return buf.value or "?"


def execute_swipe(direction):
    # Usa i tasti hardware Browser Back/Forward: sono piu' diretti di Alt+freccia
    # e vengono riconosciuti nativamente da Chrome/Edge/Firefox.
    target = foreground_window_title()
    if direction == "LEFT":
        tap_combo(VK_BROWSER_BACK)
        return f"BACK SENT -> {target[:28]}"
    if direction == "RIGHT":
        tap_combo(VK_BROWSER_FORWARD)
        return f"FORWARD SENT -> {target[:28]}"
    return ""


def execute_radial_action(action):
    if action == "LEFT":
        tap_combo(VK_LEFT, (VK_MENU,))
        return "BACK"
    if action == "RIGHT":
        tap_combo(VK_TAB, (VK_MENU,))
        return "NEXT APP"
    if action == "UP":
        tap_combo(VK_TAB, (VK_LWIN,))
        return "TASK VIEW"
    if action == "DOWN":
        tap_combo(VK_D, (VK_LWIN,))
        return "DESKTOP"
    return ""


try:
    volume_endpoint = AudioUtilities.GetSpeakers().EndpointVolume
except Exception:
    volume_endpoint = None


def get_system_volume():
    if volume_endpoint is None:
        return 0.5
    return float(volume_endpoint.GetMasterVolumeLevelScalar())


def set_system_volume(value):
    if volume_endpoint is not None:
        volume_endpoint.SetMasterVolumeLevelScalar(clamp(value, 0.0, 1.0), None)


def left_click():
    mouse_down()
    mouse_up()


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def cursor_position():
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


cursor_lock = threading.Lock()
cursor_target = [0.0, 0.0]
cursor_active = False
cursor_stop = False


def sync_cursor_target(active):
    global cursor_active
    x, y = cursor_position()
    with cursor_lock:
        cursor_target[0] = float(x)
        cursor_target[1] = float(y)
        cursor_active = active


def add_cursor_delta(dx, dy):
    with cursor_lock:
        cursor_target[0] = clamp(cursor_target[0] + dx, 0, SCREEN_W - 1)
        cursor_target[1] = clamp(cursor_target[1] + dy, 0, SCREEN_H - 1)

def cursor_worker():
    last = time.perf_counter()
    while not cursor_stop:
        now = time.perf_counter()
        dt = max(now - last, 1.0 / 500.0)
        last = now
        with cursor_lock:
            active = cursor_active
            tx, ty = cursor_target
        if active:
            x, y = cursor_position()
            alpha = 1.0 - math.exp(-dt / CURSOR_INTERP_TAU)
            nx = x + (tx - x) * alpha
            ny = y + (ty - y) * alpha
            if abs(tx - x) > 0.5 or abs(ty - y) > 0.5:
                user32.SetCursorPos(int(round(nx)), int(round(ny)))
        time.sleep(1.0 / CURSOR_OUTPUT_HZ)


class RuntimeCleanup:
    """Own runtime resources so cleanup is guaranteed by the public wrapper."""

    def __init__(self):
        self.capture = None
        self.mediapipe_worker = None
        self.cursor_thread = None

    def close(self):
        global cursor_stop

        try:
            sync_cursor_target(False)
        except Exception:
            pass

        worker = self.mediapipe_worker
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
            try:
                worker.join(timeout=2.0)
            except Exception:
                pass

        cursor_stop = True
        if self.cursor_thread is not None:
            try:
                self.cursor_thread.join(timeout=0.25)
            except Exception:
                pass

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def draw_hand(frame, hand, pinch_active=False, paused=False, scrolling=False, volume_control=False):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand]
    if volume_control:
        line_color = point_color = (255, 0, 255)
    elif paused:
        line_color = point_color = (0, 165, 255)
    elif scrolling:
        line_color = point_color = (255, 180, 0)
    elif pinch_active:
        line_color = point_color = (0, 0, 255)
    else:
        line_color, point_color = (255, 255, 255), (0, 255, 0)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 3, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 8 if i in (4, 8, 12) else 5, point_color, -1, cv2.LINE_AA)
    if pinch_active:
        mx, my = mouse_point(hand)
        cv2.circle(frame, (int(mx * w), int(my * h)), 13, (0, 255, 255), 3, cv2.LINE_AA)


def draw_radial_menu(frame, center, selected):
    h, w = frame.shape[:2]
    cx, cy = int(center[0] * w), int(center[1] * h)
    radius = int(min(w, h) * 0.115)
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(radius * 0.35), (180, 180, 180), 1, cv2.LINE_AA)
    items = {
        "UP": ((cx, cy - radius), "TASK"),
        "RIGHT": ((cx + radius, cy), "APP"),
        "DOWN": ((cx, cy + radius), "DESKTOP"),
        "LEFT": ((cx - radius, cy), "BACK"),
    }
    for direction, (pt, label) in items.items():
        active = direction == selected
        color = (0, 255, 255) if active else (255, 255, 255)
        thickness = 3 if active else 1
        cv2.circle(frame, pt, 24 if active else 19, color, thickness, cv2.LINE_AA)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(frame, label, (pt[0] - size[0] // 2, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_two_hand_transform(frame, point_a, point_b):
    h, w = frame.shape[:2]
    a = (int(point_a[0] * w), int(point_a[1] * h))
    b = (int(point_b[0] * w), int(point_b[1] * h))
    cv2.line(frame, a, b, (255, 255, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, a, 14, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, b, 14, (255, 255, 0), 3, cv2.LINE_AA)
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    cv2.putText(frame, "ZOOM", (mx - 42, my - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


def flow_points_from_hand(hand):
    pts = []
    for idx in FLOW_LANDMARK_IDS:
        pts.append([hand[idx].x * DETECTION_W, hand[idx].y * DETECTION_H])
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def propagate_points(old_gray, new_gray, points):
    if old_gray is None or new_gray is None or points is None:
        return None
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray, new_gray, points, None,
        winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if new_pts is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if good.sum() < 3:
        return None
    if err is not None and np.median(err.reshape(-1)[good]) > FLOW_MAX_ERROR:
        return None
    return new_pts


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

def _run_impl(cleanup):
    global cursor_stop
    cursor_stop = False
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cleanup.capture = cap
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
        cleanup.capture = cap
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire la webcam")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    reported_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    reported_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    camera_codec = "".join(
        chr((reported_fourcc >> (8 * i)) & 0xFF) for i in range(4)
    ).replace("\x00", "") or "?"
    camera_target_fps = choose_camera_target_fps(
        reported_fps, TARGET_FPS, FALLBACK_FPS,
    )
    cv2.namedWindow("Hands", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.perf_counter()

    def build_mediapipe_image(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    mp_worker = MediaPipeWorker(
        factory=HandLandmarker,
        options=options,
        image_builder=build_mediapipe_image,
    )
    cleanup.mediapipe_worker = mp_worker
    mp_worker.start()
    sync_cursor_target(False)
    cursor_thread = threading.Thread(target=cursor_worker, daemon=True)
    cleanup.cursor_thread = cursor_thread
    cursor_thread.start()

    flow_prev_gray = None
    flow_points = None
    flow_virtual = np.zeros(2, dtype=np.float64)
    flow_filtered = np.zeros(2, dtype=np.float64)
    flow_prev_filtered = np.zeros(2, dtype=np.float64)
    flow_time = None
    flow_active = False
    flow_motion_scale = 1.0
    last_flow_success = 0.0

    latest_result = None
    latest_result_seq = -1
    control_index = 0
    control_handedness = None
    mp_control_ref = None
    fist_states = []
    scroll_states = []
    paused_by_fist = False
    scroll_active = False
    scroll_candidate_at = None
    scroll_release_at = None
    scroll_residual = 0.0
    volume_active = False
    volume_candidate_at = None
    volume_candidate_last_seen = None
    volume_release_at = None
    volume_pose_lost_at = None
    volume_last_angle = None
    volume_delta_history = deque(maxlen=VOLUME_DELTA_MEDIAN_FRAMES)
    volume_level = get_system_volume()
    fist_vote_history = deque(maxlen=FIST_VOTE_WINDOW)
    volume_vote_history = deque(maxlen=VOLUME_VOTE_WINDOW)
    debug_fist_score = 0.0
    debug_volume_score = 0.0
    debug_grip_gap = 0.0
    debug_fist_folded = 0
    debug_fist_tightness = 2.0
    debug_strong_fist = False

    # Stato del Gesture Engine centrale.
    gesture_mode = "MOUSE"
    gesture_event = ""
    gesture_event_until = 0.0
    gesture_input_block_until = 0.0
    swipe_tracking = False
    swipe_cooldown_until = 0.0
    swipe_pose_history = deque(maxlen=SWIPE_POSE_HISTORY)
    swipe_pose_last_seen = None
    swipe_flow_started_at = None
    swipe_flow_accum_x = 0.0
    swipe_flow_accum_y = 0.0
    debug_swipe_score = 0.0
    debug_swipe_stable = 0.0
    debug_swipe_gap = 9.0
    debug_swipe_extended = 0
    radial_candidate_at = None
    radial_anchor = None
    radial_active = False
    radial_center = None
    radial_selected = None
    radial_selection_candidate = None
    radial_selection_since = None
    radial_release_at = None
    radial_pinch_latched = False
    radial_pinch_candidate_at = None
    two_hand_candidate_at = None
    two_hand_active = False
    two_hand_release_at = None
    two_hand_last_distance = None
    two_hand_distance_history = deque(maxlen=TWO_HAND_DISTANCE_HISTORY)
    two_hand_zoom_residual = 0.0
    two_hand_points = None

    # Gate globale dei comandi. Il tracking resta sempre acceso.
    commands_enabled = False
    spock_candidate_at = None
    spock_last_seen = None
    spock_release_at = None
    spock_latched = False
    spock_release_required = False
    spock_blocking = False
    spock_progress = 0.0
    spock_confirmed_seconds = 0.0
    debug_spock_score = 0.0
    debug_spock_stable_score = 0.0
    spock_score_history = deque(maxlen=SPOCK_SCORE_WINDOW)
    spock_upright_invalid_frames = 0

    last_click_at = None

    # Puntatore pinch-only.
    pointer_pinch_held = False
    pointer_move_active = False
    pointer_pinch_started_at = None
    pointer_release_at = None
    pointer_release_braking = False
    # Spostamento netto del palmo dall'inizio del pinch. Non somma il jitter.
    pointer_motion_accum = np.zeros(2, dtype=np.float64)
    pointer_flow_travel = 0.0
    pointer_cursor_origin = None

    # Precision snap: stabilizza il cursore quando la mano rallenta.
    snap_anchor = None
    snap_started_at = None
    precision_snap_active = False
    last_hand_seen = time.perf_counter()
    fps_window_start = time.perf_counter()
    fps_frames = 0
    actual_fps = 0.0
    mp_fps_window_start = time.perf_counter()
    mp_fps_last_seq = 0
    actual_mp_fps = 0.0
    mp_infer_ms_ema = 0.0
    mp_worker_ms_ema = 0.0
    mp_cycle_ms_ema = 0.0
    mp_queue_ms_ema = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        frame = cv2.flip(frame, 1)
        detect_frame = cv2.resize(frame, (DETECTION_W, DETECTION_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

        ts = int((now - start_time) * 1000)
        # resize/cvtColor allocano array nuovi a ogni frame: il worker puo'
        # mantenere questi riferimenti senza costose copie aggiuntive.
        mp_worker.submit(detect_frame, gray, ts, now)
        mp_state = mp_worker.snapshot_state()
        packet = mp_state["latest"]
        mp_stats = mp_state
        if not mp_state["alive"]:
            detail = mp_state["last_error"] or "unknown worker failure"
            raise RuntimeError(f"MediaPipe worker stopped: {detail}")
        mp_input_seq = mp_stats["input_seq"]
        mp_overwrites = mp_stats["overwrites"]
        mp_error_count = mp_stats["error_count"]
        mp_last_error = mp_stats["last_error"]
        mp_result_stale = tracking_result_is_stale(
            mp_stats["last_result_input_at"], now, MP_RESULT_STALE_SECONDS,
        )

        if mp_result_stale:
            spock_release_required = spock_release_required or spock_latched
            spock_candidate_at = None
            spock_last_seen = None
            spock_release_at = None
            spock_latched = False
            spock_blocking = spock_release_required
            spock_progress = 0.0
            spock_confirmed_seconds = 0.0
            spock_score_history.clear()
            spock_upright_invalid_frames = 0
            paused_by_fist = False
            fist_vote_history.clear()
            volume_active = False
            volume_candidate_at = None
            volume_candidate_last_seen = None
            volume_release_at = None
            volume_pose_lost_at = None
            volume_last_angle = None
            volume_delta_history.clear()
            volume_vote_history.clear()
            scroll_active = False
            scroll_candidate_at = None
            scroll_release_at = None
            scroll_residual = 0.0
            two_hand_active = False
            two_hand_candidate_at = None
            two_hand_release_at = None
            two_hand_last_distance = None
            two_hand_distance_history.clear()
            two_hand_zoom_residual = 0.0
            two_hand_points = None
            radial_active = False
            radial_candidate_at = None
            radial_anchor = None
            radial_center = None
            radial_selected = None
            radial_selection_candidate = None
            radial_selection_since = None
            radial_release_at = None
            radial_pinch_latched = False
            radial_pinch_candidate_at = None
            swipe_tracking = False
            swipe_flow_started_at = None
            swipe_flow_accum_x = 0.0
            swipe_flow_accum_y = 0.0
            swipe_pose_last_seen = None
            pointer_pinch_held = False
            pointer_move_active = False
            pointer_pinch_started_at = None
            pointer_release_at = None
            pointer_release_braking = False
            pointer_motion_accum[:] = 0.0
            pointer_flow_travel = 0.0
            pointer_cursor_origin = None
            mp_control_ref = None
            control_handedness = None
            flow_points = None
            flow_active = False
            flow_virtual[:] = 0.0
            flow_filtered[:] = 0.0
            flow_prev_filtered[:] = 0.0
            flow_time = None
            latest_result = None
            fist_states = []
            scroll_states = []
            debug_fist_score = 0.0
            debug_volume_score = 0.0
            debug_grip_gap = 0.0
            debug_fist_folded = 0
            debug_fist_tightness = 2.0
            debug_strong_fist = False
            debug_spock_score = 0.0
            debug_spock_stable_score = 0.0
            debug_swipe_score = 0.0
            debug_swipe_stable = 0.0
            debug_swipe_gap = 9.0
            debug_swipe_extended = 0
            snap_anchor = None
            snap_started_at = None
            precision_snap_active = False
            sync_cursor_target(False)

        # Optical flow: misura il movimento su ogni frame della webcam.
        if flow_prev_gray is not None and flow_points is not None:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                flow_prev_gray, gray, flow_points, None,
                winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
            )
            if next_pts is not None and status is not None:
                back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, flow_prev_gray, next_pts, None,
                    winSize=(25, 25), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                )
                good = status.reshape(-1).astype(bool)
                if back_pts is not None and back_status is not None:
                    good &= back_status.reshape(-1).astype(bool)
                    fb_error = np.linalg.norm(
                        back_pts.reshape(-1, 2) - flow_points.reshape(-1, 2), axis=1
                    )
                    good &= fb_error <= FLOW_FB_MAX
                if good.sum() >= 3:
                    old_good = flow_points.reshape(-1, 2)[good]
                    new_good = next_pts.reshape(-1, 2)[good]
                    deltas = new_good - old_good
                    mdx, mdy = np.median(deltas, axis=0)
                    spread = np.median(np.linalg.norm(deltas - [mdx, mdy], axis=1))
                    med_err = 0.0 if err is None else float(np.median(err.reshape(-1)[good]))
                    raw_mag = math.hypot(float(mdx), float(mdy))
                    valid = (spread <= FLOW_MAX_SPREAD and med_err <= FLOW_MAX_ERROR and
                             raw_mag <= FLOW_MAX_CAMERA_STEP)
                    if valid:
                        motion_dx, motion_dy = normalize_flow_delta(
                            float(mdx), float(mdy), flow_motion_scale,
                        )
                        motion_mag = math.hypot(motion_dx, motion_dy)
                        flow_points = next_pts
                        flow_active = True
                        last_flow_success = now
                        if (not mp_result_stale and not paused_by_fist and
                                commands_enabled and not spock_blocking and
                                now >= gesture_input_block_until):
                            # Swipe dinamico al frame-rate camera. MediaPipe decide solo
                            # se la posa e' valida; il movimento viene letto dallo stesso
                            # optical flow usato dal mouse, quindi non c'e' piu' competizione.
                            swipe_motion_consumed = False
                            swipe_base_gate = (
                                not pointer_pinch_held and
                                not volume_active and volume_candidate_at is None and
                                not two_hand_active and not radial_active and
                                not scroll_active and two_hand_candidate_at is None and
                                now >= swipe_cooldown_until
                            )
                            swipe_pose_recent = (
                                swipe_pose_last_seen is not None and
                                now - swipe_pose_last_seen <= SWIPE_MISS_GRACE
                            )
                            if swipe_base_gate:
                                flow_dx = motion_dx
                                flow_dy = motion_dy
                                horizontal_flow = (
                                    abs(flow_dx) >= SWIPE_FLOW_INTENT_PX and
                                    abs(flow_dx) >= abs(flow_dy) * SWIPE_FLOW_AXIS_DOMINANCE
                                )
                                # MediaPipe deve validare la posa solo all'innesco. Dopo
                                # l'inizio del flick l'optical flow continua da solo per
                                # pochi millisecondi, cosi' il motion blur non annulla il gesto.
                                if not swipe_tracking and swipe_pose_recent and horizontal_flow:
                                    swipe_tracking = True
                                    swipe_flow_started_at = now
                                    swipe_flow_accum_x = 0.0
                                    swipe_flow_accum_y = 0.0
                                    sync_cursor_target(False)
                                    flow_virtual[:] = 0.0
                                    flow_filtered[:] = 0.0
                                    flow_prev_filtered[:] = 0.0
                                    flow_time = None

                                if swipe_tracking:
                                    swipe_motion_consumed = True
                                    swipe_flow_accum_x += flow_dx
                                    swipe_flow_accum_y += flow_dy
                                    swipe_elapsed = now - (swipe_flow_started_at or now)
                                    total_horizontal = (
                                        abs(swipe_flow_accum_x) >=
                                        abs(swipe_flow_accum_y) * SWIPE_FLOW_AXIS_DOMINANCE
                                    )
                                    if (swipe_elapsed <= SWIPE_FLOW_MAX_SECONDS and
                                            total_horizontal and
                                            abs(swipe_flow_accum_x) >= SWIPE_FLOW_TRIGGER_PX):
                                        direction = "RIGHT" if swipe_flow_accum_x > 0 else "LEFT"
                                        action_label = execute_swipe(direction)
                                        gesture_event = f"SWIPE {direction}: {action_label}"
                                        gesture_event_until = now + GESTURE_EVENT_SHOW_SECONDS
                                        swipe_cooldown_until = now + SWIPE_COOLDOWN
                                        gesture_input_block_until = now + 0.10
                                        swipe_tracking = False
                                        swipe_pose_history.clear()
                                        swipe_pose_last_seen = None
                                        swipe_flow_started_at = None
                                        swipe_flow_accum_x = 0.0
                                        swipe_flow_accum_y = 0.0
                                        sync_cursor_target(True)
                                        flow_virtual[:] = 0.0
                                        flow_filtered[:] = 0.0
                                        flow_prev_filtered[:] = 0.0
                                        flow_time = None
                                    elif swipe_elapsed > SWIPE_FLOW_MAX_SECONDS:
                                        swipe_tracking = False
                                        swipe_flow_started_at = None
                                        swipe_flow_accum_x = 0.0
                                        swipe_flow_accum_y = 0.0
                                        sync_cursor_target(True)
                                        flow_virtual[:] = 0.0
                                        flow_filtered[:] = 0.0
                                        flow_prev_filtered[:] = 0.0
                                        flow_time = None
                            elif swipe_tracking:
                                # La posa e' sparita durante il flick: annulla subito.
                                swipe_motion_consumed = True
                                swipe_tracking = False
                                swipe_flow_started_at = None
                                swipe_flow_accum_x = 0.0
                                swipe_flow_accum_y = 0.0
                                sync_cursor_target(True)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None

                            if swipe_motion_consumed:
                                pass
                            elif (volume_active or two_hand_active or radial_active or
                                    two_hand_candidate_at is not None):
                                pass
                            elif scroll_active:
                                if abs(motion_dy) >= SCROLL_DEADZONE_PX:
                                    scroll_residual += motion_dy * SCROLL_GAIN
                                    if abs(scroll_residual) >= SCROLL_EVENT_STEP:
                                        steps = int(scroll_residual / SCROLL_EVENT_STEP)
                                        wheel_delta = int(clamp(
                                            steps * SCROLL_EVENT_STEP,
                                            -SCROLL_MAX_EVENT, SCROLL_MAX_EVENT,
                                        ))
                                        mouse_wheel(wheel_delta)
                                        scroll_residual -= wheel_delta
                            elif pointer_pinch_held and not pointer_release_braking:
                                # Il click non deve trasformarsi in movimento per jitter:
                                # misura lo spostamento NETTO del palmo dall'inizio del pinch.
                                pointer_motion_accum += np.array(
                                    [motion_dx, motion_dy], dtype=np.float64,
                                )
                                pointer_flow_travel = max(
                                    pointer_flow_travel,
                                    float(np.linalg.norm(pointer_motion_accum)),
                                )
                                if (not pointer_move_active and
                                        pointer_flow_travel >= POINTER_MOVE_TRIGGER_PX):
                                    pointer_move_active = True
                                    sync_cursor_target(True)
                                    # Il movimento che ha superato la soglia serve solo a
                                    # distinguere move da click: riancora qui per evitare salti.
                                    flow_virtual[:] = 0.0
                                    flow_filtered[:] = 0.0
                                    flow_prev_filtered[:] = 0.0
                                    flow_time = None

                                if pointer_move_active:
                                    flow_virtual += np.array(
                                        [motion_dx, motion_dy], dtype=np.float64,
                                    )
                                    if flow_time is None:
                                        flow_filtered[:] = flow_virtual
                                        flow_prev_filtered[:] = flow_filtered
                                        flow_time = now
                                    else:
                                        dt = max(now - flow_time, 1.0 / 240.0)
                                        speed = motion_mag / dt
                                        mix = clamp(speed / FLOW_FAST_SPEED, 0.0, 1.0)
                                        tau = FLOW_TAU_SLOW + (FLOW_TAU_FAST - FLOW_TAU_SLOW) * mix
                                        alpha = 1.0 - math.exp(-dt / tau)
                                        flow_filtered += (flow_virtual - flow_filtered) * alpha
                                        out = flow_filtered - flow_prev_filtered
                                        flow_prev_filtered[:] = flow_filtered
                                        flow_time = now
                                        out_mag = float(np.linalg.norm(out))
                                        if out_mag >= FLOW_SOFT_DEADZONE_PX:
                                            if out_mag < FLOW_DEADZONE_PX:
                                                ramp = clamp(
                                                    (out_mag - FLOW_SOFT_DEADZONE_PX) /
                                                    (FLOW_DEADZONE_PX - FLOW_SOFT_DEADZONE_PX),
                                                    0.0, 1.0,
                                                )
                                                out = out * (ramp * ramp)
                                            dynamic_gain = cursor_gain_for_speed(speed)
                                            dx = (out[0] / DETECTION_W * SCREEN_W * MOVE_GAIN *
                                                  MOVEMENT_MULTIPLIER * dynamic_gain)
                                            dy = (out[1] / DETECTION_H * SCREEN_H * MOVE_GAIN *
                                                  MOVEMENT_MULTIPLIER * dynamic_gain)
                                            screen_step = math.hypot(float(dx), float(dy))
                                            if precision_snap_active:
                                                if screen_step <= SNAP_BREAK_DELTA_PX:
                                                    dx *= SNAP_HOLD_GAIN
                                                    dy *= SNAP_HOLD_GAIN
                                                else:
                                                    precision_snap_active = False
                                                    snap_anchor = None
                                                    snap_started_at = None
                                            add_cursor_delta(float(dx), float(dy))
                            else:
                                # Mano aperta o qualsiasi posa senza pinch: cursore fermo.
                                if cursor_active:
                                    sync_cursor_target(False)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                    else:
                        flow_active = False
                else:
                    flow_active = False
            else:
                flow_active = False
        flow_prev_gray = gray
        # Consuma solo risultati MediaPipe nuovi; l'inferenza non blocca il loop camera.
        new_mp = False
        if packet is not None and packet[0] != latest_result_seq:
            (latest_result_seq, latest_result, result_gray,
             mp_infer_ms, mp_worker_ms, mp_cycle_ms, mp_queue_ms) = packet
            new_mp = True

        if new_mp:
            if mp_infer_ms_ema <= 0.0:
                mp_infer_ms_ema = mp_infer_ms
                mp_worker_ms_ema = mp_worker_ms
                mp_cycle_ms_ema = mp_cycle_ms
                mp_queue_ms_ema = mp_queue_ms
            else:
                mp_infer_ms_ema = mp_infer_ms_ema * 0.85 + mp_infer_ms * 0.15
                mp_worker_ms_ema = mp_worker_ms_ema * 0.85 + mp_worker_ms * 0.15
                if mp_cycle_ms > 0.0:
                    if mp_cycle_ms_ema <= 0.0:
                        mp_cycle_ms_ema = mp_cycle_ms
                    else:
                        mp_cycle_ms_ema = mp_cycle_ms_ema * 0.85 + mp_cycle_ms * 0.15
                mp_queue_ms_ema = mp_queue_ms_ema * 0.85 + mp_queue_ms * 0.15
            if latest_result.hand_landmarks:
                hands = latest_result.hand_landmarks
                world_hands = getattr(latest_result, "hand_world_landmarks", None)
                if not world_hands or len(world_hands) != len(hands):
                    class_hands = hands
                else:
                    class_hands = world_hands

                # Spock e' il gate globale: il movimento assoluto della mano NON conta.
                # Usiamo solo la geometria relativa delle dita e una memoria temporale
                # robusta: tremolio, motion blur o 2-4 frame sbagliati non interrompono
                # l'aggancio ne' azzerano il secondo di mantenimento.
                spock_scores_now = [spock_pose_score(hand) for hand in hands]
                debug_spock_score = max(spock_scores_now, default=0.0)
                upright_now = any(spock_all_fingers_up(hand) for hand in hands)
                if upright_now:
                    spock_upright_invalid_frames = 0
                    spock_score_history.append(debug_spock_score)
                else:
                    spock_upright_invalid_frames += 1
                    # Un frame storto puo' essere jitter. Una posa realmente non verticale
                    # per piu' frame annulla invece subito la candidatura Spock.
                    if spock_upright_invalid_frames >= SPOCK_UP_INVALID_FRAMES:
                        spock_score_history.clear()
                        if not spock_latched:
                            spock_candidate_at = None
                            spock_last_seen = None
                            spock_blocking = False
                            spock_progress = 0.0
                            spock_confirmed_seconds = 0.0

                ranked_spock = sorted(spock_score_history, reverse=True)
                keep_index = min(SPOCK_SCORE_KEEP_BEST - 1, len(ranked_spock) - 1)
                history_evidence = ranked_spock[keep_index] if ranked_spock else 0.0
                debug_spock_stable_score = max(debug_spock_score, history_evidence)
                if spock_upright_invalid_frames >= SPOCK_UP_INVALID_FRAMES:
                    debug_spock_stable_score = 0.0
                spock_threshold = (
                    SPOCK_SCORE_HOLD
                    if (spock_candidate_at is not None or spock_latched or spock_blocking)
                    else SPOCK_SCORE_ON
                )
                spock_now = debug_spock_stable_score >= spock_threshold
                spock_toggled = False
                spock_sample_seconds = min(
                    SPOCK_SAMPLE_MAX_SECONDS,
                    max(
                        (mp_cycle_ms / 1000.0) if mp_cycle_ms > 0.0
                        else 1.0 / max(camera_target_fps, 1),
                        1.0 / 120.0,
                    ),
                )
                if spock_release_required:
                    spock_blocking = True
                    spock_progress = 1.0
                    if spock_now:
                        spock_release_at = None
                    else:
                        if spock_release_at is None:
                            spock_release_at = now
                        release_elapsed = now - spock_release_at
                        if not spock_release_gate_active(
                                required=True,
                                detected=False,
                                release_elapsed=release_elapsed,
                                release_seconds=SPOCK_RELEASE_SECONDS):
                            spock_release_required = False
                            spock_blocking = False
                            spock_release_at = None
                            spock_progress = 0.0
                            spock_confirmed_seconds = 0.0
                            gesture_input_block_until = max(
                                gesture_input_block_until,
                                now + SPOCK_POST_RELEASE_BLOCK,
                            )
                elif spock_now:
                    spock_release_at = None
                    spock_last_seen = now
                    spock_blocking = True
                    if spock_latched:
                        spock_progress = 1.0
                    else:
                        if spock_candidate_at is None:
                            spock_candidate_at = now
                            spock_confirmed_seconds = 0.0
                        spock_confirmed_seconds = advance_confirmed_hold(
                            spock_confirmed_seconds,
                            upright_now,
                            spock_sample_seconds,
                            SPOCK_HOLD_SECONDS,
                        )
                        spock_progress = clamp(
                            spock_confirmed_seconds / SPOCK_HOLD_SECONDS, 0.0, 1.0
                        )
                        if spock_progress >= 1.0:
                            commands_enabled = not commands_enabled
                            spock_latched = True
                            spock_candidate_at = None
                            spock_progress = 1.0
                            spock_toggled = True
                            gesture_event = (
                                "CONTROLLI ATTIVI" if commands_enabled else "CONTROLLI BLOCCATI"
                            )
                            gesture_event_until = now + 1.20
                else:
                    if spock_latched:
                        spock_blocking = True
                        if spock_release_at is None:
                            spock_release_at = now
                        elif now - spock_release_at >= SPOCK_RELEASE_SECONDS:
                            spock_latched = False
                            spock_blocking = False
                            spock_release_at = None
                            spock_progress = 0.0
                            spock_confirmed_seconds = 0.0
                            gesture_input_block_until = max(
                                gesture_input_block_until,
                                now + SPOCK_POST_RELEASE_BLOCK,
                            )
                            mp_control_ref = None
                            control_handedness = None
                            flow_points = None
                            flow_active = False
                            sync_cursor_target(False)
                    elif (spock_candidate_at is not None and spock_last_seen is not None and
                          now - spock_last_seen <= SPOCK_MISS_GRACE):
                        spock_blocking = True
                        spock_progress = clamp(
                            spock_confirmed_seconds / SPOCK_HOLD_SECONDS, 0.0, 1.0
                        )
                    else:
                        spock_candidate_at = None
                        spock_last_seen = None
                        spock_release_at = None
                        spock_blocking = False
                        spock_progress = 0.0
                        spock_confirmed_seconds = 0.0
                        spock_score_history.clear()
                        debug_spock_stable_score = 0.0

                if spock_toggled:
                    paused_by_fist = False
                    fist_vote_history.clear()
                    volume_active = False
                    volume_candidate_at = None
                    volume_candidate_last_seen = None
                    volume_release_at = None
                    volume_pose_lost_at = None
                    volume_last_angle = None
                    volume_delta_history.clear()
                    volume_vote_history.clear()
                    scroll_active = False
                    scroll_candidate_at = None
                    scroll_release_at = None
                    scroll_residual = 0.0
                    two_hand_active = False
                    two_hand_candidate_at = None
                    two_hand_release_at = None
                    two_hand_last_distance = None
                    two_hand_distance_history.clear()
                    two_hand_zoom_residual = 0.0
                    two_hand_points = None
                    radial_active = False
                    radial_candidate_at = None
                    radial_anchor = None
                    radial_center = None
                    radial_selected = None
                    radial_selection_candidate = None
                    radial_selection_since = None
                    radial_release_at = None
                    radial_pinch_latched = False
                    radial_pinch_candidate_at = None
                    swipe_tracking = False
                    mp_control_ref = None
                    control_handedness = None
                    flow_points = None
                    flow_active = False
                    flow_virtual[:] = 0.0
                    flow_filtered[:] = 0.0
                    flow_prev_filtered[:] = 0.0
                    flow_time = None
                    sync_cursor_target(False)

                class_metrics = [grip_class_scores(hand) for hand in class_hands]
                norm_metrics = [grip_class_scores(hand) for hand in hands]
                fist_scores_now = [m[0] for m in class_metrics]
                volume_scores_now = [
                    max(class_metrics[i][1], norm_metrics[i][1])
                    for i in range(len(hands))
                ]
                gap_scores_now = [norm_metrics[i][2] for i in range(len(hands))]

                points = [control_point(hand) for hand in hands]
                handedness_result = getattr(latest_result, "handedness", None) or []
                handedness_labels = []
                for i in range(len(hands)):
                    try:
                        handedness_labels.append(
                            handedness_result[i][0].category_name or ""
                        )
                    except (IndexError, AttributeError, TypeError):
                        handedness_labels.append("")
                old_mp_ref = mp_control_ref
                control_index = choose_control_index(
                    points,
                    handedness_labels,
                    previous_point=mp_control_ref,
                    previous_label=control_handedness,
                )
                control_distance = (
                    0.0 if mp_control_ref is None else
                    math.hypot(
                        points[control_index][0] - mp_control_ref[0],
                        points[control_index][1] - mp_control_ref[1],
                    )
                )
                selected_handedness = handedness_labels[control_index]

                # Il pugno e' un clutch globale: qualunque mano puo' metterci in pausa,
                # mentre VOLUME LOCK richiede comunque un pugno forte per interrompersi.
                raw_fist_states = [is_fist(hand) for hand in hands]
                strong_fist_states = [is_strong_fist(hand) for hand in hands]
                fist_evidence = fist_evidence_from_hands(
                    raw_fists=raw_fist_states,
                    strong_fists=strong_fist_states,
                    volume_scores=volume_scores_now,
                    gap_scores=gap_scores_now,
                    volume_active=volume_active,
                    volume_score_on=VOLUME_SCORE_ON,
                    suppress_gap=VOLUME_FIST_SUPPRESS_GAP,
                )
                old_pause = paused_by_fist
                fist_vote_history.append(1.0 if fist_evidence else 0.0)
                fist_votes_on = sum(v > 0.5 for v in fist_vote_history)
                fist_pending = fist_votes_on >= FIST_PENDING_VOTES
                if not paused_by_fist:
                    if fist_votes_on >= FIST_VOTE_ON:
                        paused_by_fist = True
                else:
                    if sum(v < 0.5 for v in fist_vote_history) >= FIST_VOTE_OFF:
                        paused_by_fist = False

                # Durante VOLUME LOCK non consentire un salto improvviso sull'altra mano.
                if (volume_active and mp_control_ref is not None and
                        control_distance > VOLUME_HAND_SWITCH_MAX):
                    if now - last_hand_seen > VOLUME_TRACKING_LOSS_GRACE:
                        volume_active = False
                        volume_candidate_at = None
                        volume_candidate_last_seen = None
                        volume_release_at = None
                        volume_pose_lost_at = None
                        volume_last_angle = None
                        volume_delta_history.clear()
                        volume_vote_history.clear()
                        mp_control_ref = None
                        control_handedness = None
                        sync_cursor_target(False)
                    continue

                last_hand_seen = now
                if selected_handedness:
                    control_handedness = selected_handedness
                control_hand = hands[control_index]
                control_class_hand = class_hands[control_index]
                palm_width_px = normalized_points_pixel_distance(
                    (control_hand[5].x, control_hand[5].y),
                    (control_hand[17].x, control_hand[17].y),
                    DETECTION_W,
                    DETECTION_H,
                )
                target_motion_scale = palm_motion_scale(
                    palm_width_px,
                    reference_width_px=PALM_REFERENCE_WIDTH_PX,
                    minimum=PALM_SCALE_MIN,
                    maximum=PALM_SCALE_MAX,
                )
                flow_motion_scale = flow_motion_scale * 0.82 + target_motion_scale * 0.18
                debug_fist_score = max(fist_scores_now, default=0.0)
                debug_volume_score = volume_scores_now[control_index]
                debug_grip_gap = gap_scores_now[control_index]
                debug_fist_folded, debug_fist_tightness = fist_fold_metrics(control_hand)
                debug_strong_fist = strong_fist_states[control_index]

                # Un singolo frame compatto non deve rendere difficile il volume.
                # Solo quando il pugno e' coerente per almeno due frame congela
                # temporaneamente il volume; al terzo voto entra nel clutch.
                if fist_pending:
                    volume_vote_history.clear()
                    volume_last_angle = None
                    volume_delta_history.clear()
                else:
                    volume_vote_history.append(debug_volume_score)

                volume_gesture_now = (
                    sum(s >= VOLUME_SCORE_ON for s in volume_vote_history) >= VOLUME_VOTE_ON
                    and not paused_by_fist
                    and not fist_pending
                )
                volume_candidate_now = (
                    debug_volume_score >= VOLUME_SCORE_CANDIDATE
                    and not paused_by_fist
                    and not fist_pending
                )
                fist_states = [
                    (paused_by_fist and raw_fist_states[i])
                    for i in range(len(hands))
                ]
                scroll_states = [
                    is_scroll_gesture(hand)
                    and not paused_by_fist
                    and not volume_active
                    and not volume_candidate_now
                    for hand in hands
                ]
                scroll_gesture_now = scroll_states[control_index] and not volume_gesture_now
                mp_control_ref = points[control_index]

                # --------------------------------------------------------------
                # Gesture Engine - priorita': pugno > volume > due mani >
                # menu radiale > scroll > swipe > puntatore.
                # --------------------------------------------------------------
                pair_hands = None
                two_hand_held = False
                pair_geometry = None
                if (len(hands) >= 2 and commands_enabled and not spock_blocking and
                        not paused_by_fist and not volume_active and
                        volume_candidate_at is None and not volume_candidate_now):
                    pair_hands = sorted(hands[:2], key=lambda h: control_point(h)[0])
                    pair_geometry = two_hand_geometry(pair_hands[0], pair_hands[1])
                    pinch_limit = TWO_HAND_PINCH_OFF if two_hand_active else TWO_HAND_PINCH_ON
                    two_hand_held = (
                        pair_geometry[0] >= TWO_HAND_MIN_SEPARATION and
                        all(normalized_pinch_ratio(hand, 8) < pinch_limit
                            for hand in pair_hands)
                    )

                if two_hand_held:
                    if two_hand_candidate_at is None:
                        two_hand_candidate_at = now
                    two_hand_release_at = None
                    if (two_hand_active or
                            now - two_hand_candidate_at >= TWO_HAND_CONFIRM_SECONDS):
                        distance_now, point_a, point_b = pair_geometry
                        if not two_hand_active:
                            two_hand_active = True
                            two_hand_distance_history.clear()
                            two_hand_distance_history.append(distance_now)
                            two_hand_last_distance = distance_now
                            two_hand_zoom_residual = 0.0
                            radial_active = False
                            radial_candidate_at = None
                            radial_release_at = None
                            radial_pinch_latched = False
                            swipe_tracking = False
                            scroll_active = False
                            scroll_candidate_at = None
                            scroll_release_at = None
                            scroll_residual = 0.0
                            volume_candidate_at = None
                            volume_candidate_last_seen = None
                            volume_vote_history.clear()
                            sync_cursor_target(False)
                            flow_virtual[:] = 0.0
                            flow_filtered[:] = 0.0
                            flow_prev_filtered[:] = 0.0
                            flow_time = None
                        else:
                            two_hand_distance_history.append(distance_now)
                            stable_distance = sorted(two_hand_distance_history)[
                                len(two_hand_distance_history) // 2
                            ]
                            if (two_hand_last_distance is not None and
                                    len(two_hand_distance_history) >= 3):
                                distance_delta = stable_distance - two_hand_last_distance
                                if abs(distance_delta) > TWO_HAND_MAX_DISTANCE_DELTA:
                                    # Salto di tracking: riancora senza emettere zoom.
                                    two_hand_last_distance = stable_distance
                                    two_hand_zoom_residual = 0.0
                                elif abs(distance_delta) >= TWO_HAND_DISTANCE_DEADZONE:
                                    two_hand_zoom_residual += distance_delta * TWO_HAND_ZOOM_GAIN
                                    two_hand_last_distance = stable_distance
                                    zoom_steps = int(two_hand_zoom_residual / TWO_HAND_WHEEL_STEP)
                                    if zoom_steps != 0:
                                        ctrl_wheel(zoom_steps * int(TWO_HAND_WHEEL_STEP))
                                        two_hand_zoom_residual -= zoom_steps * TWO_HAND_WHEEL_STEP
                        two_hand_points = (point_a, point_b)
                elif two_hand_active:
                    if two_hand_release_at is None:
                        two_hand_release_at = now
                    elif now - two_hand_release_at >= TWO_HAND_RELEASE_GRACE:
                        two_hand_active = False
                        two_hand_candidate_at = None
                        two_hand_release_at = None
                        two_hand_last_distance = None
                        two_hand_distance_history.clear()
                        two_hand_zoom_residual = 0.0
                        two_hand_points = None
                        gesture_input_block_until = now + 0.16
                        sync_cursor_target(True)
                        flow_virtual[:] = 0.0
                        flow_filtered[:] = 0.0
                        flow_prev_filtered[:] = 0.0
                        flow_time = None
                else:
                    two_hand_candidate_at = None
                    two_hand_release_at = None

                # Menu radiale: mano aperta e quasi ferma per ~1 s. Il centro
                # resta fisso; sposta la mano verso una voce e fai pinch per selezionare.
                radial_priority_block = (
                    not commands_enabled or spock_blocking or paused_by_fist or
                    volume_active or two_hand_active or
                    two_hand_candidate_at is not None or len(hands) != 1
                )
                if radial_priority_block:
                    radial_candidate_at = None
                    radial_anchor = None
                    if radial_active:
                        radial_active = False
                        radial_center = None
                        radial_selected = None
                        radial_selection_candidate = None
                        radial_selection_since = None
                        radial_release_at = None
                        radial_pinch_latched = False
                        radial_pinch_candidate_at = None
                elif radial_active:
                    raw_selection = radial_direction(control_hand, radial_center)
                    if raw_selection != radial_selection_candidate:
                        radial_selection_candidate = raw_selection
                        radial_selection_since = now
                        radial_selected = None
                        radial_pinch_candidate_at = None
                    elif (radial_selection_since is not None and
                          now - radial_selection_since >= RADIAL_SELECTION_HOLD):
                        radial_selected = raw_selection

                    radial_pinch = normalized_pinch_ratio(control_hand, 8)
                    if radial_pinch < RADIAL_PINCH_ON:
                        if radial_pinch_candidate_at is None:
                            radial_pinch_candidate_at = now
                        if (not radial_pinch_latched and radial_selected is not None and
                                now - radial_pinch_candidate_at >= RADIAL_PINCH_CONFIRM):
                            radial_pinch_latched = True
                            action_label = execute_radial_action(radial_selected)
                            gesture_event = f"RADIAL: {action_label}"
                            gesture_event_until = now + GESTURE_EVENT_SHOW_SECONDS
                            gesture_input_block_until = now + 0.20
                            radial_active = False
                            radial_candidate_at = None
                            radial_anchor = None
                            radial_center = None
                            radial_selected = None
                            radial_selection_candidate = None
                            radial_selection_since = None
                            radial_release_at = None
                            radial_pinch_candidate_at = None
                            sync_cursor_target(True)
                            flow_virtual[:] = 0.0
                            flow_filtered[:] = 0.0
                            flow_prev_filtered[:] = 0.0
                            flow_time = None
                    elif radial_pinch > RADIAL_PINCH_OFF:
                        radial_pinch_latched = False
                        radial_pinch_candidate_at = None

                    if radial_active:
                        open_now = is_radial_open_pose(control_class_hand)
                        if open_now or radial_pinch < RADIAL_PINCH_OFF:
                            radial_release_at = None
                        else:
                            if radial_release_at is None:
                                radial_release_at = now
                            elif now - radial_release_at >= RADIAL_RELEASE_GRACE:
                                radial_active = False
                                radial_candidate_at = None
                                radial_anchor = None
                                radial_center = None
                                radial_selected = None
                                radial_selection_candidate = None
                                radial_selection_since = None
                                radial_release_at = None
                                radial_pinch_latched = False
                                radial_pinch_candidate_at = None
                                gesture_input_block_until = now + 0.12
                                sync_cursor_target(True)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                else:
                    open_now = is_radial_open_pose(control_class_hand)
                    radial_can_arm = (
                        open_now and not scroll_active and not swipe_tracking and
                        not volume_candidate_now and volume_candidate_at is None and
                        now >= gesture_input_block_until
                    )
                    if radial_can_arm:
                        current_anchor = points[control_index]
                        if radial_candidate_at is None or radial_anchor is None:
                            radial_candidate_at = now
                            radial_anchor = current_anchor
                        else:
                            drift = math.hypot(
                                current_anchor[0] - radial_anchor[0],
                                current_anchor[1] - radial_anchor[1],
                            )
                            if drift > RADIAL_STILL_MAX:
                                radial_candidate_at = now
                                radial_anchor = current_anchor
                            elif now - radial_candidate_at >= RADIAL_OPEN_HOLD:
                                radial_active = True
                                radial_center = current_anchor
                                radial_selected = None
                                radial_selection_candidate = None
                                radial_selection_since = None
                                radial_release_at = None
                                radial_pinch_latched = False
                                radial_pinch_candidate_at = None
                                swipe_tracking = False
                                scroll_active = False
                                scroll_candidate_at = None
                                scroll_release_at = None
                                scroll_residual = 0.0
                                sync_cursor_target(False)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                    else:
                        radial_candidate_at = None
                        radial_anchor = None

                # Spazzata naturale: score temporale della mano piatta/unita +
                # movimento laterale. Non richiede piu' una posa perfetta frame per frame.
                swipe_allowed = (
                    commands_enabled and not spock_blocking and len(hands) == 1 and
                    not paused_by_fist and not volume_active and
                    not volume_candidate_now and volume_candidate_at is None and
                    not two_hand_active and two_hand_candidate_at is None and
                    not radial_active and not scroll_active and
                    now >= gesture_input_block_until
                )
                if swipe_allowed:
                    (debug_swipe_score, debug_swipe_gap,
                     debug_swipe_extended) = swipe_pose_metrics(control_hand)
                    swipe_pose_history.append(debug_swipe_score)
                    best_scores = sorted(swipe_pose_history, reverse=True)[:SWIPE_POSE_KEEP_BEST]
                    debug_swipe_stable = sum(best_scores) / max(len(best_scores), 1)
                    if (debug_swipe_score >= SWIPE_POSE_SCORE_ON or
                            debug_swipe_stable >= SWIPE_POSE_SCORE_ON):
                        swipe_pose_last_seen = now
                    elif (swipe_tracking and
                          debug_swipe_score >= SWIPE_POSE_SCORE_HOLD):
                        # Solo uno swipe gia' partito usa la soglia bassa per
                        # tollerare motion blur senza armare falsamente il mouse.
                        swipe_pose_last_seen = now
                else:
                    debug_swipe_score = 0.0
                    debug_swipe_stable = 0.0
                    debug_swipe_gap = 9.0
                    debug_swipe_extended = 0
                    swipe_pose_history.clear()
                    swipe_pose_last_seen = None

                # MediaPipe ri-ancora i punti rigidi del palmo; LK li porta al frame corrente.
                # Puntatore pinch-only. La mano aperta non muove mai il cursore.
                pointer_allowed = pointer_mode_allowed(
                    commands_enabled=commands_enabled,
                    spock_blocking=spock_blocking,
                    hand_count=len(hands),
                    paused=paused_by_fist,
                    volume_active=volume_active,
                    two_hand_active=two_hand_active,
                    two_hand_candidate=two_hand_candidate_at is not None,
                    radial_active=radial_active,
                    scroll_active=scroll_active,
                    swipe_tracking=swipe_tracking,
                    input_blocked=now < gesture_input_block_until,
                    volume_candidate=(
                        volume_candidate_now or volume_candidate_at is not None
                    ),
                )
                pointer_ratio = normalized_pinch_ratio(control_hand, 8)
                pointer_fingers_valid = pointer_other_fingers_valid(control_hand)
                pointer_pose_on = is_pointer_pinch_pose(control_hand, POINTER_PINCH_ON)

                if pointer_pinch_held:
                    # Se medio/anulare/mignolo si chiudono a pugno, annulla il
                    # puntatore: non deve diventare ne' movimento ne' click.
                    if not pointer_allowed or not pointer_fingers_valid:
                        pointer_pinch_held = False
                        pointer_move_active = False
                        pointer_pinch_started_at = None
                        pointer_release_at = None
                        pointer_release_braking = False
                        pointer_motion_accum[:] = 0.0
                        pointer_flow_travel = 0.0
                        pointer_cursor_origin = None
                        sync_cursor_target(False)
                    elif pointer_ratio > POINTER_PINCH_OFF:
                        # Congela immediatamente il cursore appena il pinch e' chiaramente
                        # aperto. La grace sotto serve solo a confermare il rilascio.
                        pointer_release_braking = True
                        pointer_move_active = False
                        sync_cursor_target(False)
                        flow_virtual[:] = 0.0
                        flow_filtered[:] = 0.0
                        flow_prev_filtered[:] = 0.0
                        flow_time = None
                        if pointer_release_at is None:
                            pointer_release_at = now
                        elif now - pointer_release_at >= POINTER_RELEASE_GRACE:
                            pinch_duration = now - (pointer_pinch_started_at or now)
                            quick_click = (
                                pinch_duration <= POINTER_CLICK_MAX_SECONDS and
                                pointer_flow_travel <= POINTER_CLICK_MAX_TRAVEL_PX
                            )
                            if quick_click:
                                # Un click rapido non deve spostare il cursore nemmeno
                                # se i primi frame del pinch hanno prodotto un po' di jitter.
                                if pointer_cursor_origin is not None:
                                    user32.SetCursorPos(
                                        int(pointer_cursor_origin[0]), int(pointer_cursor_origin[1])
                                    )
                                    sync_cursor_target(False)
                                is_double_pinch = (
                                    last_click_at is not None and
                                    now - last_click_at <= DOUBLE_PINCH_WINDOW
                                )
                                left_click()
                                if is_double_pinch:
                                    gesture_event = "DOPPIO PINCH: DOPPIO CLICK"
                                    last_click_at = None
                                else:
                                    gesture_event = "PINCH RAPIDO: CLICK"
                                    last_click_at = now
                                gesture_event_until = now + GESTURE_EVENT_SHOW_SECONDS
                            pointer_pinch_held = False
                            pointer_move_active = False
                            pointer_pinch_started_at = None
                            pointer_release_at = None
                            pointer_release_braking = False
                            pointer_motion_accum[:] = 0.0
                            pointer_flow_travel = 0.0
                            pointer_cursor_origin = None
                            precision_snap_active = False
                            snap_anchor = None
                            snap_started_at = None
                            sync_cursor_target(False)
                            flow_virtual[:] = 0.0
                            flow_filtered[:] = 0.0
                            flow_prev_filtered[:] = 0.0
                            flow_time = None
                    elif pointer_ratio > POINTER_RELEASE_BRAKE_RATIO:
                        # Pre-rilascio: blocca il cursore prima ancora di raggiungere
                        # la soglia OFF, cosi' l'apertura delle dita non trascina il mouse.
                        pointer_release_braking = True
                        pointer_move_active = False
                        pointer_release_at = None
                        sync_cursor_target(False)
                        flow_virtual[:] = 0.0
                        flow_filtered[:] = 0.0
                        flow_prev_filtered[:] = 0.0
                        flow_time = None
                    else:
                        pointer_release_braking = False
                        pointer_release_at = None
                elif pointer_allowed and pointer_pose_on:
                    pointer_pinch_held = True
                    pointer_move_active = False
                    pointer_pinch_started_at = now
                    pointer_release_at = None
                    pointer_release_braking = False
                    pointer_motion_accum[:] = 0.0
                    pointer_flow_travel = 0.0
                    pointer_cursor_origin = cursor_position()
                    swipe_tracking = False
                    swipe_flow_started_at = None
                    swipe_flow_accum_x = 0.0
                    swipe_flow_accum_y = 0.0
                    swipe_pose_last_seen = None
                    sync_cursor_target(False)
                    flow_virtual[:] = 0.0
                    flow_filtered[:] = 0.0
                    flow_prev_filtered[:] = 0.0
                    flow_time = None

                # Il pinch abilita il puntatore, ma la traslazione viene sempre
                # misurata sulla parte rigida del palmo. Cosi' chiudere/aprire indice
                # e pollice non introduce movimento spurio del cursore.
                mp_pts = flow_points_from_hand(control_hand)
                corrected = propagate_points(result_gray, gray, mp_pts)
                if corrected is not None:
                    flow_points = corrected
                    flow_prev_gray = gray
                    flow_active = True

                if paused_by_fist or not commands_enabled or spock_blocking:
                    volume_active = False
                    volume_candidate_at = None
                    volume_candidate_last_seen = None
                    volume_release_at = None
                    volume_pose_lost_at = None
                    volume_last_angle = None
                    volume_delta_history.clear()
                    volume_vote_history.clear()
                    scroll_active = False
                    scroll_candidate_at = None
                    scroll_release_at = None
                    scroll_residual = 0.0
                    two_hand_active = False
                    two_hand_candidate_at = None
                    two_hand_release_at = None
                    two_hand_last_distance = None
                    two_hand_distance_history.clear()
                    two_hand_zoom_residual = 0.0
                    two_hand_points = None
                    radial_active = False
                    radial_candidate_at = None
                    radial_anchor = None
                    radial_center = None
                    radial_selected = None
                    radial_selection_candidate = None
                    radial_selection_since = None
                    radial_release_at = None
                    radial_pinch_latched = False
                    radial_pinch_candidate_at = None
                    swipe_tracking = False
                else:
                    if not volume_active:
                        dedicated_mode_block = (
                            pointer_pinch_held or
                            two_hand_active or two_hand_candidate_at is not None or
                            radial_active or swipe_tracking
                        )
                        if dedicated_mode_block:
                            volume_candidate_at = None
                            volume_candidate_last_seen = None
                            volume_vote_history.clear()
                        elif volume_gesture_now or volume_candidate_now:
                            if volume_candidate_at is None:
                                volume_candidate_at = now
                            volume_candidate_last_seen = now
                            if (volume_gesture_now and
                                    now - volume_candidate_at >= VOLUME_CONFIRM_SECONDS):
                                volume_active = True
                                volume_candidate_last_seen = None
                                volume_release_at = None
                                volume_pose_lost_at = None
                                volume_last_angle = palm_roll_angle(control_hand)
                                volume_delta_history.clear()
                                volume_level = get_system_volume()
                                scroll_active = False
                                scroll_candidate_at = None
                                scroll_release_at = None
                                scroll_residual = 0.0
                                sync_cursor_target(False)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                        elif volume_candidate_at is not None:
                            # Un singolo frame incerto non annulla l'aggancio e non lascia
                            # partire click/pugno/scroll nello stesso istante.
                            if (volume_candidate_last_seen is None or
                                    now - volume_candidate_last_seen > VOLUME_ENTRY_MISS_GRACE):
                                volume_candidate_at = None
                                volume_candidate_last_seen = None
                    else:
                        # VOLUME LOCK: appena la mano mostra una chiara apertura,
                        # congela SUBITO il volume. La conferma temporale serve solo
                        # a decidere quando uscire definitivamente dalla modalita'.
                        release_pose = is_volume_release_pose(control_class_hand)
                        fully_open = is_open_hand(control_class_hand)
                        if fist_pending:
                            # Due frame compatti consecutivi: congela il volume ma
                            # aspetta il voto successivo prima di dichiarare PUGNO.
                            volume_pose_lost_at = None
                            volume_release_at = None
                            volume_last_angle = None
                            volume_delta_history.clear()
                        elif release_pose or fully_open:
                            volume_pose_lost_at = None
                            if volume_release_at is None:
                                volume_release_at = now
                            # Nessuna rotazione viene letta durante il rilascio.
                            volume_last_angle = None
                            volume_delta_history.clear()
                            if now - volume_release_at >= VOLUME_RELEASE_GRACE:
                                volume_active = False
                                volume_candidate_at = None
                                volume_candidate_last_seen = None
                                volume_release_at = None
                                volume_pose_lost_at = None
                                volume_last_angle = None
                                volume_delta_history.clear()
                                volume_vote_history.clear()
                                sync_cursor_target(True)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                        elif debug_volume_score < VOLUME_HOLD_MIN_SCORE:
                            # La posa non e' piu' credibile: congela immediatamente
                            # il volume e aspetta una breve grazia prima di sganciare.
                            volume_release_at = None
                            if volume_pose_lost_at is None:
                                volume_pose_lost_at = now
                            volume_last_angle = None
                            volume_delta_history.clear()
                            if now - volume_pose_lost_at >= VOLUME_POSE_LOSS_GRACE:
                                volume_active = False
                                volume_candidate_at = None
                                volume_candidate_last_seen = None
                                volume_pose_lost_at = None
                                volume_last_angle = None
                                volume_vote_history.clear()
                                sync_cursor_target(True)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                        else:
                            volume_release_at = None
                            volume_pose_lost_at = None
                            angle = palm_roll_angle(control_hand)
                            if volume_last_angle is None:
                                # Dopo un falso rilascio/rientro, questo frame diventa
                                # il nuovo zero: niente salto di volume.
                                volume_last_angle = angle
                                volume_delta_history.clear()
                            else:
                                raw_delta = wrapped_angle_delta(angle, volume_last_angle)
                                volume_last_angle = angle
                                raw_delta = clamp(raw_delta, -VOLUME_MAX_DELTA_RAD, VOLUME_MAX_DELTA_RAD)
                                volume_delta_history.append(raw_delta)
                                stable_delta = sorted(volume_delta_history)[len(volume_delta_history) // 2]
                                if abs(stable_delta) >= VOLUME_DEADZONE_RAD:
                                    volume_level = clamp(
                                        volume_level + stable_delta * VOLUME_GAIN * VOLUME_DIRECTION,
                                        0.0, 1.0,
                                    )
                                    set_system_volume(volume_level)

                    if (pointer_pinch_held or volume_active or two_hand_active or radial_active or
                            swipe_tracking or two_hand_candidate_at is not None):
                        scroll_active = False
                        scroll_candidate_at = None
                        scroll_release_at = None
                        scroll_residual = 0.0
                    elif not scroll_active:
                        if scroll_gesture_now:
                            if scroll_candidate_at is None:
                                scroll_candidate_at = now
                            elif now - scroll_candidate_at >= SCROLL_CONFIRM_SECONDS:
                                scroll_active = True
                                scroll_release_at = None
                                scroll_residual = 0.0
                                sync_cursor_target(False)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None
                        else:
                            scroll_candidate_at = None
                    else:
                        if scroll_gesture_now:
                            scroll_release_at = None
                        else:
                            if scroll_release_at is None:
                                scroll_release_at = now
                            elif now - scroll_release_at >= SCROLL_RELEASE_GRACE:
                                scroll_active = False
                                scroll_candidate_at = None
                                scroll_release_at = None
                                scroll_residual = 0.0
                                sync_cursor_target(True)
                                flow_virtual[:] = 0.0
                                flow_filtered[:] = 0.0
                                flow_prev_filtered[:] = 0.0
                                flow_time = None

                if paused_by_fist or not commands_enabled or spock_blocking:
                    sync_cursor_target(False)
                    flow_virtual[:] = 0.0
                    flow_filtered[:] = 0.0
                    flow_prev_filtered[:] = 0.0
                    flow_time = None
                else:
                    exclusive_cursor_block = (
                        scroll_active or volume_active or two_hand_active or radial_active or
                        swipe_tracking or two_hand_candidate_at is not None
                    )
                    if (pointer_move_active and not exclusive_cursor_block and
                            now >= gesture_input_block_until and
                            (old_pause or not cursor_active)):
                        sync_cursor_target(True)
                        flow_virtual[:] = 0.0
                        flow_filtered[:] = 0.0
                        flow_prev_filtered[:] = 0.0
                        flow_time = None
                    elif not pointer_move_active and cursor_active:
                        sync_cursor_target(False)

                    if (pointer_pinch_held and pointer_move_active and
                            not exclusive_cursor_block and corrected is None and
                            old_mp_ref is not None and mp_control_ref is not None and
                            not old_pause and now >= gesture_input_block_until):
                        # Fallback MediaPipe: anche qui usa il centro rigido del palmo,
                        # mai il contatto indice-pollice, per evitare salti durante il gesto.
                        fdx = mp_control_ref[0] - old_mp_ref[0]
                        fdy = mp_control_ref[1] - old_mp_ref[1]
                        fdx, fdy = normalize_flow_delta(fdx, fdy, flow_motion_scale)
                        fmag = math.hypot(fdx, fdy)
                        pointer_flow_travel += fmag * max(DETECTION_W, DETECTION_H)
                        if 0.0005 < fmag < 0.055:
                            fallback_dt = (
                                max(mp_cycle_ms_ema / 1000.0, 1.0 / 60.0)
                                if mp_cycle_ms_ema > 0.0 else 1.0 / 30.0
                            )
                            fallback_speed = fmag * max(DETECTION_W, DETECTION_H) / fallback_dt
                            dynamic_gain = cursor_gain_for_speed(fallback_speed)
                            dx = fdx * SCREEN_W * MOVE_GAIN * MOVEMENT_MULTIPLIER * dynamic_gain
                            dy = fdy * SCREEN_H * MOVE_GAIN * MOVEMENT_MULTIPLIER * dynamic_gain
                            screen_step = math.hypot(float(dx), float(dy))
                            if precision_snap_active:
                                if screen_step <= SNAP_BREAK_DELTA_PX:
                                    dx *= SNAP_HOLD_GAIN
                                    dy *= SNAP_HOLD_GAIN
                                else:
                                    precision_snap_active = False
                                    snap_anchor = None
                                    snap_started_at = None
                            add_cursor_delta(float(dx), float(dy))
            else:
                debug_spock_score = 0.0
                debug_fist_score = 0.0
                debug_volume_score = 0.0
                debug_grip_gap = 0.0
                debug_fist_folded = 0
                debug_fist_tightness = 2.0
                debug_strong_fist = False
                debug_swipe_score = 0.0
                debug_swipe_stable = 0.0
                debug_swipe_gap = 9.0
                debug_swipe_extended = 0
                if spock_latched:
                    if spock_release_at is None:
                        spock_release_at = now
                    elif now - spock_release_at >= SPOCK_RELEASE_SECONDS:
                        spock_latched = False
                        spock_blocking = False
                        spock_release_at = None
                        spock_progress = 0.0
                        spock_confirmed_seconds = 0.0
                        spock_score_history.clear()
                        debug_spock_stable_score = 0.0
                        gesture_input_block_until = max(
                            gesture_input_block_until,
                            now + SPOCK_POST_RELEASE_BLOCK,
                        )
                elif (spock_candidate_at is not None and spock_last_seen is not None and
                      now - spock_last_seen > SPOCK_MISS_GRACE):
                    spock_candidate_at = None
                    spock_last_seen = None
                    spock_blocking = False
                    spock_progress = 0.0
                    spock_confirmed_seconds = 0.0
                    spock_score_history.clear()
                    debug_spock_stable_score = 0.0

                fist_states = []
                scroll_states = []

                # Il vecchio grace di 280 ms e' utile per non perdere lo stato delle
                # gesture, ma e' troppo lungo per un cursore: se la mano sparisce,
                # LK puo' iniziare a seguire lo sfondo. Congela quindi il puntatore
                # dopo circa due frame e lo riancora quando MediaPipe rivede la mano.
                if (pointer_move_active and
                        now - last_hand_seen > POINTER_TRACKING_LOSS_GRACE):
                    sync_cursor_target(False)
                    flow_points = None
                    flow_active = False
                    flow_virtual[:] = 0.0
                    flow_filtered[:] = 0.0
                    flow_prev_filtered[:] = 0.0
                    flow_time = None

                loss_grace = VOLUME_TRACKING_LOSS_GRACE if volume_active else TRACKING_LOSS_GRACE
                if now - last_hand_seen > loss_grace:
                    paused_by_fist = False
                    volume_active = False
                    volume_candidate_at = None
                    volume_candidate_last_seen = None
                    volume_release_at = None
                    volume_pose_lost_at = None
                    volume_last_angle = None
                    volume_delta_history.clear()
                    volume_vote_history.clear()
                    fist_vote_history.clear()
                    scroll_active = False
                    scroll_candidate_at = None
                    scroll_release_at = None
                    scroll_residual = 0.0
                    two_hand_active = False
                    two_hand_candidate_at = None
                    two_hand_release_at = None
                    two_hand_last_distance = None
                    two_hand_distance_history.clear()
                    two_hand_zoom_residual = 0.0
                    two_hand_points = None
                    radial_active = False
                    radial_candidate_at = None
                    radial_anchor = None
                    radial_center = None
                    radial_selected = None
                    radial_selection_candidate = None
                    radial_selection_since = None
                    radial_release_at = None
                    radial_pinch_latched = False
                    radial_pinch_candidate_at = None
                    swipe_tracking = False
                    pointer_pinch_held = False
                    pointer_move_active = False
                    pointer_pinch_started_at = None
                    pointer_release_at = None
                    pointer_motion_accum[:] = 0.0
                    pointer_flow_travel = 0.0
                    pointer_cursor_origin = None
                    gesture_input_block_until = 0.0
                    mp_control_ref = None
                    control_handedness = None
                    flow_points = None
                    flow_active = False
                    sync_cursor_target(False)
        # Se l'optical flow perde i punti, MediaPipe li riaggancia al prossimo risultato valido.
        if not flow_active and now - last_flow_success > TRACKING_LOSS_GRACE:
            flow_points = None

        # Precision snap: quando il cursore resta stabile lo trattiene leggermente.
        snap_allowed = (
            pointer_pinch_held and pointer_move_active and
            commands_enabled and not spock_blocking and not paused_by_fist and
            latest_result is not None and bool(latest_result.hand_landmarks) and
            not volume_active and volume_candidate_at is None and not scroll_active and
            not two_hand_active and two_hand_candidate_at is None and
            not radial_active and not swipe_tracking and
            now >= gesture_input_block_until
        )
        if snap_allowed:
            cursor_x, cursor_y = cursor_position()
            if snap_anchor is None or snap_started_at is None:
                snap_anchor = (float(cursor_x), float(cursor_y))
                snap_started_at = now
                precision_snap_active = False
            else:
                snap_drift = math.hypot(
                    cursor_x - snap_anchor[0], cursor_y - snap_anchor[1]
                )
                allowed_radius = SNAP_HOLD_RADIUS_PX if precision_snap_active else SNAP_RADIUS_PX
                if snap_drift <= allowed_radius:
                    stable_for = now - snap_started_at
                    if stable_for >= SNAP_ARM_SECONDS:
                        precision_snap_active = True
                else:
                    snap_anchor = (float(cursor_x), float(cursor_y))
                    snap_started_at = now
                    precision_snap_active = False
        else:
            snap_anchor = None
            snap_started_at = None
            precision_snap_active = False

        pinch_now = pointer_pinch_held
        if spock_blocking:
            gesture_mode = "SPOCK"
        elif not commands_enabled:
            gesture_mode = "LOCKED"
        elif paused_by_fist:
            gesture_mode = "FIST"
        elif pointer_move_active:
            gesture_mode = "POINTER"
        elif pointer_pinch_held:
            gesture_mode = "PINCH"
        else:
            gesture_mode = resolve_gesture_mode(
                paused_by_fist, volume_active, two_hand_active, radial_active,
                scroll_active, swipe_tracking,
            )
        if latest_result is not None and latest_result.hand_landmarks:
            for i, hand in enumerate(latest_result.hand_landmarks):
                paused = fist_states[i] if i < len(fist_states) else False
                draw_hand(frame, hand,
                          pinch_active=(i == control_index and pinch_now),
                          paused=paused,
                          scrolling=(i == control_index and scroll_active),
                          volume_control=(i == control_index and volume_active))

        if radial_active and radial_center is not None:
            draw_radial_menu(frame, radial_center, radial_selected)
        if two_hand_active and two_hand_points is not None:
            draw_two_hand_transform(
                frame, two_hand_points[0], two_hand_points[1],
            )

        fps_frames += 1
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            actual_fps = fps_frames / elapsed
            camera_target_fps = choose_camera_target_fps(
                actual_fps, TARGET_FPS, FALLBACK_FPS,
            )
            fps_frames = 0
            fps_window_start = now

        mp_elapsed = now - mp_fps_window_start
        if mp_elapsed >= 0.5:
            completed_seq = mp_worker.snapshot_state()["seq"]
            actual_mp_fps = (completed_seq - mp_fps_last_seq) / mp_elapsed
            mp_fps_last_seq = completed_seq
            mp_fps_window_start = now

        if gesture_mode == "SPOCK":
            status = f"SPOCK: {int(round(spock_progress * 100))}% | tieni 1 s per TOGGLE"
        elif gesture_mode == "LOCKED":
            status = "COMANDI BLOCCATI | fai SPOCK per 1 s"
        elif gesture_mode == "FIST":
            status = "PUGNO = PAUSA TRACKING"
        elif gesture_mode == "VOLUME":
            status = f"VOLUME: {int(round(volume_level * 100))}%"
        elif gesture_mode == "TWO_HAND":
            status = "2 MANI: ZOOM"
        elif gesture_mode == "RADIAL":
            choice = radial_selected if radial_selected is not None else "CENTRO"
            status = f"MENU RADIALE: {choice} | PINCH = OK"
        elif gesture_mode == "SCROLL":
            status = "SCROLL: INDICE + MEDIO"
        elif gesture_mode == "SWIPE":
            status = "SWIPE: 4 DITA UNITE | SX=BACK DX=FORWARD"
        elif gesture_mode == "POINTER":
            status = "PUNTATORE: PINCH INDICE+POLLICE | MUOVI PER SPOSTARE"
        elif gesture_mode == "PINCH":
            status = "PINCH: RILASCIA RAPIDO = CLICK | MUOVI = PUNTATORE"
        else:
            status = "CURSORE FERMO | PINCH INDICE+POLLICE PER PUNTARE"
        flow_label = "FLOW ON" if flow_active else "FLOW WAIT"
        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Camera {actual_fps:.1f} | MP {actual_mp_fps:.1f} FPS | call {mp_infer_ms_ema:.0f} ms | worker {mp_worker_ms_ema:.0f} ms | cycle {mp_cycle_ms_ema:.0f} ms | {flow_label}",
            (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.66, (255, 255, 255), 2, cv2.LINE_AA,
        )
        drop_pct = 100.0 * mp_overwrites / max(mp_input_seq, 1)
        cv2.putText(
            frame,
            f"MP queue {mp_queue_ms_ema:.1f} ms | drop {drop_pct:.0f}% | {camera_codec} {reported_w}x{reported_h}@{reported_fps:.0f} | target {camera_target_fps} | ESC",
            (30, 125), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FIST {debug_fist_score:.2f} | VOL {debug_volume_score:.2f} | GAP {debug_grip_gap:.2f}",
            (30, 158), cv2.FONT_HERSHEY_SIMPLEX,
            0.64, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FOLD {debug_fist_folded}/4 | TIGHT {debug_fist_tightness:.2f} | STRONG {int(debug_strong_fist)}",
            (30, 190), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"ENGINE: {gesture_mode} | SPOCK raw {debug_spock_score:.2f} stable {debug_spock_stable_score:.2f}",
            (30, 222), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (200, 255, 200), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"SWIPE raw {debug_swipe_score:.2f} stable {debug_swipe_stable:.2f} | JOIN {debug_swipe_gap:.2f} | EXT {debug_swipe_extended}/4",
            (30, 254), cv2.FONT_HERSHEY_SIMPLEX,
            0.56, (255, 255, 255), 2, cv2.LINE_AA,
        )
        if gesture_event and now < gesture_event_until:
            cv2.putText(
                frame, gesture_event,
                (30, 286), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (0, 255, 255), 2, cv2.LINE_AA,
            )
        if mp_error_count:
            cv2.putText(
                frame,
                f"MP ERR {mp_error_count}: {mp_last_error}",
                (30, 318), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 80, 255), 2, cv2.LINE_AA,
            )

        # Simula a schermo il LED che poi potra' stare sopra la TV.
        frame_w = frame.shape[1]
        if spock_blocking:
            led_color = (0, 180, 255)      # ambra = Spock in lettura
        elif commands_enabled:
            led_color = (0, 255, 0)        # verde = comandi attivi
        else:
            led_color = (0, 0, 255)        # rosso = comandi bloccati
        led_x, led_y = frame_w - 58, 45
        cv2.circle(frame, (led_x, led_y), 17, led_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (led_x, led_y), 20, (255, 255, 255), 2, cv2.LINE_AA)
        led_label = "CMD ON" if commands_enabled else "CMD OFF"
        cv2.putText(frame, led_label, (frame_w - 155, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, led_color, 2, cv2.LINE_AA)

        if spock_blocking and not spock_latched:
            bar_w, bar_h = 260, 16
            bar_x, bar_y = frame_w - bar_w - 30, 108
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (255, 255, 255), 2)
            fill_w = int(bar_w * clamp(spock_progress, 0.0, 1.0))
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h), led_color, -1)
        cv2.imshow("Hands", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    return None


def run():
    cleanup = RuntimeCleanup()
    try:
        return _run_impl(cleanup)
    finally:
        cleanup.close()


if __name__ == "__main__":
    run()
