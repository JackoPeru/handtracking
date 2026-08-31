"""Hand tracking runtime implementation."""

import math
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

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
from handtracking_engine import resolve_runtime_mode
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
    normalized_pinch_ratio,
    palm_roll_angle,
    pointer_other_fingers_valid,
    radial_direction,
    spock_all_fingers_up,
    spock_pose_score,
    swipe_pose_metrics,
    two_hand_geometry,
    wrapped_angle_delta,
)
from handtracking_render import draw_hand, draw_radial_menu, draw_two_hand_transform
from handtracking_state import (
    FlowState,
    PointerState,
    RadialState,
    ScrollState,
    SpockState,
    SwipeState,
    TwoHandState,
    VolumeState,
)
from handtracking_windows import (
    CursorController,
    ctrl_wheel,
    execute_radial_action,
    execute_swipe,
    get_system_volume,
    left_click,
    set_system_volume,
)

# L'optical flow usa pochissimi punti e non beneficia di 8 thread OpenCV.
# Limitarlo evita contesa CPU con MediaPipe/XNNPACK sul portatile 4C/8T.
cv2.setNumThreads(1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Optical flow sul palmo: punti rigidi che non si spostano quando pieghi le dita.
# Questo evita il salto del cursore durante click/pinch/drag.


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


class RuntimeCleanup:
    """Own runtime resources so cleanup is guaranteed by the public wrapper."""

    def __init__(self):
        self.capture = None
        self.mediapipe_worker = None
        self.cursor = None

    def close(self):
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

        if self.cursor is not None:
            try:
                self.cursor.close()
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
    cursor = CursorController()
    cleanup.cursor = cursor
    screen_w, screen_h = cursor.screen_size()
    cursor.sync(False)
    cursor.start()

    flow = FlowState()
    pointer = PointerState()
    scroll = ScrollState()
    volume = VolumeState(level=get_system_volume())
    swipe = SwipeState()
    radial = RadialState()
    two_hand = TwoHandState()
    spock = SpockState()

    latest_result = None
    latest_result_seq = -1
    control_index = 0
    control_handedness = None
    mp_control_ref = None
    fist_states = []
    scroll_states = []
    paused_by_fist = False
    fist_vote_history = deque(maxlen=FIST_VOTE_WINDOW)
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

    # Gate globale dei comandi. Il tracking resta sempre acceso.
    commands_enabled = False

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
            spock.release_required = spock.release_required or spock.latched
            spock.reset(preserve_release_required=True)
            paused_by_fist = False
            fist_vote_history.clear()
            volume.reset()
            scroll.reset()
            two_hand.reset()
            radial.reset()
            swipe.cancel_tracking()
            pointer.reset(preserve_last_click=True)
            mp_control_ref = None
            control_handedness = None
            flow.points = None
            flow.active = False
            flow.clear_motion()
            latest_result = None
            fist_states = []
            scroll_states = []
            debug_fist_score = 0.0
            debug_volume_score = 0.0
            debug_grip_gap = 0.0
            debug_fist_folded = 0
            debug_fist_tightness = 2.0
            debug_strong_fist = False
            spock.debug_score = 0.0
            spock.debug_stable_score = 0.0
            swipe.debug_score = 0.0
            swipe.debug_stable = 0.0
            swipe.debug_gap = 9.0
            swipe.debug_extended = 0
            snap_anchor = None
            snap_started_at = None
            precision_snap_active = False
            cursor.sync(False)

        # Optical flow: misura il movimento su ogni frame della webcam.
        if flow.prev_gray is not None and flow.points is not None:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                flow.prev_gray, gray, flow.points, None,
                winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
            )
            if next_pts is not None and status is not None:
                back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, flow.prev_gray, next_pts, None,
                    winSize=(25, 25), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                )
                good = status.reshape(-1).astype(bool)
                if back_pts is not None and back_status is not None:
                    good &= back_status.reshape(-1).astype(bool)
                    fb_error = np.linalg.norm(
                        back_pts.reshape(-1, 2) - flow.points.reshape(-1, 2), axis=1
                    )
                    good &= fb_error <= FLOW_FB_MAX
                if good.sum() >= 3:
                    old_good = flow.points.reshape(-1, 2)[good]
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
                            float(mdx), float(mdy), flow.motion_scale,
                        )
                        motion_mag = math.hypot(motion_dx, motion_dy)
                        flow.points = next_pts
                        flow.active = True
                        flow.last_success = now
                        if (not mp_result_stale and not paused_by_fist and
                                commands_enabled and not spock.blocking and
                                now >= gesture_input_block_until):
                            # Swipe dinamico al frame-rate camera. MediaPipe decide solo
                            # se la posa e' valida; il movimento viene letto dallo stesso
                            # optical flow usato dal mouse, quindi non c'e' piu' competizione.
                            swipe_motion_consumed = False
                            swipe_base_gate = (
                                not pointer.pinch_held and
                                not volume.active and volume.candidate_at is None and
                                not two_hand.active and not radial.active and
                                not scroll.active and two_hand.candidate_at is None and
                                now >= swipe.cooldown_until
                            )
                            swipe_pose_recent = (
                                swipe.pose_last_seen is not None and
                                now - swipe.pose_last_seen <= SWIPE_MISS_GRACE
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
                                if not swipe.tracking and swipe_pose_recent and horizontal_flow:
                                    swipe.tracking = True
                                    swipe.flow_started_at = now
                                    swipe.flow_accum_x = 0.0
                                    swipe.flow_accum_y = 0.0
                                    cursor.sync(False)
                                    flow.clear_motion()

                                if swipe.tracking:
                                    swipe_motion_consumed = True
                                    swipe.flow_accum_x += flow_dx
                                    swipe.flow_accum_y += flow_dy
                                    swipe_elapsed = now - (swipe.flow_started_at or now)
                                    total_horizontal = (
                                        abs(swipe.flow_accum_x) >=
                                        abs(swipe.flow_accum_y) * SWIPE_FLOW_AXIS_DOMINANCE
                                    )
                                    if (swipe_elapsed <= SWIPE_FLOW_MAX_SECONDS and
                                            total_horizontal and
                                            abs(swipe.flow_accum_x) >= SWIPE_FLOW_TRIGGER_PX):
                                        direction = "RIGHT" if swipe.flow_accum_x > 0 else "LEFT"
                                        action_label = execute_swipe(direction)
                                        gesture_event = f"SWIPE {direction}: {action_label}"
                                        gesture_event_until = now + GESTURE_EVENT_SHOW_SECONDS
                                        swipe.cooldown_until = now + SWIPE_COOLDOWN
                                        gesture_input_block_until = now + 0.10
                                        swipe.tracking = False
                                        swipe.pose_history.clear()
                                        swipe.pose_last_seen = None
                                        swipe.flow_started_at = None
                                        swipe.flow_accum_x = 0.0
                                        swipe.flow_accum_y = 0.0
                                        cursor.sync(True)
                                        flow.clear_motion()
                                    elif swipe_elapsed > SWIPE_FLOW_MAX_SECONDS:
                                        swipe.tracking = False
                                        swipe.flow_started_at = None
                                        swipe.flow_accum_x = 0.0
                                        swipe.flow_accum_y = 0.0
                                        cursor.sync(True)
                                        flow.clear_motion()
                            elif swipe.tracking:
                                # La posa e' sparita durante il flick: annulla subito.
                                swipe_motion_consumed = True
                                swipe.tracking = False
                                swipe.flow_started_at = None
                                swipe.flow_accum_x = 0.0
                                swipe.flow_accum_y = 0.0
                                cursor.sync(True)
                                flow.clear_motion()

                            if swipe_motion_consumed:
                                pass
                            elif (volume.active or two_hand.active or radial.active or
                                    two_hand.candidate_at is not None):
                                pass
                            elif scroll.active:
                                if abs(motion_dy) >= SCROLL_DEADZONE_PX:
                                    scroll.residual += motion_dy * SCROLL_GAIN
                                    if abs(scroll.residual) >= SCROLL_EVENT_STEP:
                                        steps = int(scroll.residual / SCROLL_EVENT_STEP)
                                        wheel_delta = int(clamp(
                                            steps * SCROLL_EVENT_STEP,
                                            -SCROLL_MAX_EVENT, SCROLL_MAX_EVENT,
                                        ))
                                        mouse_wheel(wheel_delta)
                                        scroll.residual -= wheel_delta
                            elif pointer.pinch_held and not pointer.release_braking:
                                # Il click non deve trasformarsi in movimento per jitter:
                                # misura lo spostamento NETTO del palmo dall'inizio del pinch.
                                pointer.motion_accum += np.array(
                                    [motion_dx, motion_dy], dtype=np.float64,
                                )
                                pointer.flow_travel = max(
                                    pointer.flow_travel,
                                    float(np.linalg.norm(pointer.motion_accum)),
                                )
                                if (not pointer.move_active and
                                        pointer.flow_travel >= POINTER_MOVE_TRIGGER_PX):
                                    pointer.move_active = True
                                    cursor.sync(True)
                                    # Il movimento che ha superato la soglia serve solo a
                                    # distinguere move da click: riancora qui per evitare salti.
                                    flow.clear_motion()

                                if pointer.move_active:
                                    flow.virtual += np.array(
                                        [motion_dx, motion_dy], dtype=np.float64,
                                    )
                                    if flow.time is None:
                                        flow.filtered[:] = flow.virtual
                                        flow.prev_filtered[:] = flow.filtered
                                        flow.time = now
                                    else:
                                        dt = max(now - flow.time, 1.0 / 240.0)
                                        speed = motion_mag / dt
                                        mix = clamp(speed / FLOW_FAST_SPEED, 0.0, 1.0)
                                        tau = FLOW_TAU_SLOW + (FLOW_TAU_FAST - FLOW_TAU_SLOW) * mix
                                        alpha = 1.0 - math.exp(-dt / tau)
                                        flow.filtered += (flow.virtual - flow.filtered) * alpha
                                        out = flow.filtered - flow.prev_filtered
                                        flow.prev_filtered[:] = flow.filtered
                                        flow.time = now
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
                                            dx = (out[0] / DETECTION_W * screen_w * MOVE_GAIN *
                                                  MOVEMENT_MULTIPLIER * dynamic_gain)
                                            dy = (out[1] / DETECTION_H * screen_h * MOVE_GAIN *
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
                                            cursor.add_delta(float(dx), float(dy), screen_size=(screen_w, screen_h))
                            else:
                                # Mano aperta o qualsiasi posa senza pinch: cursore fermo.
                                if cursor.active:
                                    cursor.sync(False)
                                flow.clear_motion()
                    else:
                        flow.active = False
                else:
                    flow.active = False
            else:
                flow.active = False
        flow.prev_gray = gray
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
                spock.debug_score = max(spock_scores_now, default=0.0)
                upright_now = any(spock_all_fingers_up(hand) for hand in hands)
                if upright_now:
                    spock.upright_invalid_frames = 0
                    spock.score_history.append(spock.debug_score)
                else:
                    spock.upright_invalid_frames += 1
                    # Un frame storto puo' essere jitter. Una posa realmente non verticale
                    # per piu' frame annulla invece subito la candidatura Spock.
                    if spock.upright_invalid_frames >= SPOCK_UP_INVALID_FRAMES:
                        spock.score_history.clear()
                        if not spock.latched:
                            spock.candidate_at = None
                            spock.last_seen = None
                            spock.blocking = False
                            spock.progress = 0.0
                            spock.confirmed_seconds = 0.0

                ranked_spock = sorted(spock.score_history, reverse=True)
                keep_index = min(SPOCK_SCORE_KEEP_BEST - 1, len(ranked_spock) - 1)
                history_evidence = ranked_spock[keep_index] if ranked_spock else 0.0
                spock.debug_stable_score = max(spock.debug_score, history_evidence)
                if spock.upright_invalid_frames >= SPOCK_UP_INVALID_FRAMES:
                    spock.debug_stable_score = 0.0
                spock_threshold = (
                    SPOCK_SCORE_HOLD
                    if (spock.candidate_at is not None or spock.latched or spock.blocking)
                    else SPOCK_SCORE_ON
                )
                spock_now = spock.debug_stable_score >= spock_threshold
                spock_toggled = False
                spock_sample_seconds = min(
                    SPOCK_SAMPLE_MAX_SECONDS,
                    max(
                        (mp_cycle_ms / 1000.0) if mp_cycle_ms > 0.0
                        else 1.0 / max(camera_target_fps, 1),
                        1.0 / 120.0,
                    ),
                )
                if spock.release_required:
                    spock.blocking = True
                    spock.progress = 1.0
                    if spock_now:
                        spock.release_at = None
                    else:
                        if spock.release_at is None:
                            spock.release_at = now
                        release_elapsed = now - spock.release_at
                        if not spock_release_gate_active(
                                required=True,
                                detected=False,
                                release_elapsed=release_elapsed,
                                release_seconds=SPOCK_RELEASE_SECONDS):
                            spock.release_required = False
                            spock.blocking = False
                            spock.release_at = None
                            spock.progress = 0.0
                            spock.confirmed_seconds = 0.0
                            gesture_input_block_until = max(
                                gesture_input_block_until,
                                now + SPOCK_POST_RELEASE_BLOCK,
                            )
                elif spock_now:
                    spock.release_at = None
                    spock.last_seen = now
                    spock.blocking = True
                    if spock.latched:
                        spock.progress = 1.0
                    else:
                        if spock.candidate_at is None:
                            spock.candidate_at = now
                            spock.confirmed_seconds = 0.0
                        spock.confirmed_seconds = advance_confirmed_hold(
                            spock.confirmed_seconds,
                            upright_now,
                            spock_sample_seconds,
                            SPOCK_HOLD_SECONDS,
                        )
                        spock.progress = clamp(
                            spock.confirmed_seconds / SPOCK_HOLD_SECONDS, 0.0, 1.0
                        )
                        if spock.progress >= 1.0:
                            commands_enabled = not commands_enabled
                            spock.latched = True
                            spock.candidate_at = None
                            spock.progress = 1.0
                            spock_toggled = True
                            gesture_event = (
                                "CONTROLLI ATTIVI" if commands_enabled else "CONTROLLI BLOCCATI"
                            )
                            gesture_event_until = now + 1.20
                else:
                    if spock.latched:
                        spock.blocking = True
                        if spock.release_at is None:
                            spock.release_at = now
                        elif now - spock.release_at >= SPOCK_RELEASE_SECONDS:
                            spock.latched = False
                            spock.blocking = False
                            spock.release_at = None
                            spock.progress = 0.0
                            spock.confirmed_seconds = 0.0
                            gesture_input_block_until = max(
                                gesture_input_block_until,
                                now + SPOCK_POST_RELEASE_BLOCK,
                            )
                            mp_control_ref = None
                            control_handedness = None
                            flow.points = None
                            flow.active = False
                            cursor.sync(False)
                    elif (spock.candidate_at is not None and spock.last_seen is not None and
                          now - spock.last_seen <= SPOCK_MISS_GRACE):
                        spock.blocking = True
                        spock.progress = clamp(
                            spock.confirmed_seconds / SPOCK_HOLD_SECONDS, 0.0, 1.0
                        )
                    else:
                        spock.candidate_at = None
                        spock.last_seen = None
                        spock.release_at = None
                        spock.blocking = False
                        spock.progress = 0.0
                        spock.confirmed_seconds = 0.0
                        spock.score_history.clear()
                        spock.debug_stable_score = 0.0

                if spock_toggled:
                    paused_by_fist = False
                    fist_vote_history.clear()
                    volume.reset()
                    scroll.reset()
                    two_hand.reset()
                    radial.reset()
                    swipe.tracking = False
                    mp_control_ref = None
                    control_handedness = None
                    flow.points = None
                    flow.active = False
                    flow.clear_motion()
                    cursor.sync(False)

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
                    volume_active=volume.active,
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
                if (volume.active and mp_control_ref is not None and
                        control_distance > VOLUME_HAND_SWITCH_MAX):
                    if now - last_hand_seen > VOLUME_TRACKING_LOSS_GRACE:
                        volume.reset()
                        mp_control_ref = None
                        control_handedness = None
                        cursor.sync(False)
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
                flow.motion_scale = flow.motion_scale * 0.82 + target_motion_scale * 0.18
                debug_fist_score = max(fist_scores_now, default=0.0)
                debug_volume_score = volume_scores_now[control_index]
                debug_grip_gap = gap_scores_now[control_index]
                debug_fist_folded, debug_fist_tightness = fist_fold_metrics(control_hand)
                debug_strong_fist = strong_fist_states[control_index]

                # Un singolo frame compatto non deve rendere difficile il volume.
                # Solo quando il pugno e' coerente per almeno due frame congela
                # temporaneamente il volume; al terzo voto entra nel clutch.
                if fist_pending:
                    volume.vote_history.clear()
                    volume.last_angle = None
                    volume.delta_history.clear()
                else:
                    volume.vote_history.append(debug_volume_score)

                volume_gesture_now = (
                    sum(s >= VOLUME_SCORE_ON for s in volume.vote_history) >= VOLUME_VOTE_ON
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
                    and not volume.active
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
                if (len(hands) >= 2 and commands_enabled and not spock.blocking and
                        not paused_by_fist and not volume.active and
                        volume.candidate_at is None and not volume_candidate_now):
                    pair_hands = sorted(hands[:2], key=lambda h: control_point(h)[0])
                    pair_geometry = two_hand_geometry(pair_hands[0], pair_hands[1])
                    pinch_limit = TWO_HAND_PINCH_OFF if two_hand.active else TWO_HAND_PINCH_ON
                    two_hand_held = (
                        pair_geometry[0] >= TWO_HAND_MIN_SEPARATION and
                        all(normalized_pinch_ratio(hand, 8) < pinch_limit
                            for hand in pair_hands)
                    )

                if two_hand_held:
                    if two_hand.candidate_at is None:
                        two_hand.candidate_at = now
                    two_hand.release_at = None
                    if (two_hand.active or
                            now - two_hand.candidate_at >= TWO_HAND_CONFIRM_SECONDS):
                        distance_now, point_a, point_b = pair_geometry
                        if not two_hand.active:
                            two_hand.active = True
                            two_hand.distance_history.clear()
                            two_hand.distance_history.append(distance_now)
                            two_hand.last_distance = distance_now
                            two_hand.zoom_residual = 0.0
                            radial.reset()
                            swipe.tracking = False
                            scroll.reset()
                            volume.candidate_at = None
                            volume.candidate_last_seen = None
                            volume.vote_history.clear()
                            cursor.sync(False)
                            flow.clear_motion()
                        else:
                            two_hand.distance_history.append(distance_now)
                            stable_distance = sorted(two_hand.distance_history)[
                                len(two_hand.distance_history) // 2
                            ]
                            if (two_hand.last_distance is not None and
                                    len(two_hand.distance_history) >= 3):
                                distance_delta = stable_distance - two_hand.last_distance
                                if abs(distance_delta) > TWO_HAND_MAX_DISTANCE_DELTA:
                                    # Salto di tracking: riancora senza emettere zoom.
                                    two_hand.last_distance = stable_distance
                                    two_hand.zoom_residual = 0.0
                                elif abs(distance_delta) >= TWO_HAND_DISTANCE_DEADZONE:
                                    two_hand.zoom_residual += distance_delta * TWO_HAND_ZOOM_GAIN
                                    two_hand.last_distance = stable_distance
                                    zoom_steps = int(two_hand.zoom_residual / TWO_HAND_WHEEL_STEP)
                                    if zoom_steps != 0:
                                        ctrl_wheel(zoom_steps * int(TWO_HAND_WHEEL_STEP))
                                        two_hand.zoom_residual -= zoom_steps * TWO_HAND_WHEEL_STEP
                        two_hand.points = (point_a, point_b)
                elif two_hand.active:
                    if two_hand.release_at is None:
                        two_hand.release_at = now
                    elif now - two_hand.release_at >= TWO_HAND_RELEASE_GRACE:
                        two_hand.reset()
                        gesture_input_block_until = now + 0.16
                        cursor.sync(True)
                        flow.clear_motion()
                else:
                    two_hand.candidate_at = None
                    two_hand.release_at = None

                # Menu radiale: mano aperta e quasi ferma per ~1 s. Il centro
                # resta fisso; sposta la mano verso una voce e fai pinch per selezionare.
                radial_priority_block = (
                    not commands_enabled or spock.blocking or paused_by_fist or
                    volume.active or two_hand.active or
                    two_hand.candidate_at is not None or len(hands) != 1
                )
                if radial_priority_block:
                    radial.candidate_at = None
                    radial.anchor = None
                    if radial.active:
                        radial.reset()
                elif radial.active:
                    raw_selection = radial_direction(control_hand, radial.center)
                    if raw_selection != radial.selection_candidate:
                        radial.selection_candidate = raw_selection
                        radial.selection_since = now
                        radial.selected = None
                        radial.pinch_candidate_at = None
                    elif (radial.selection_since is not None and
                          now - radial.selection_since >= RADIAL_SELECTION_HOLD):
                        radial.selected = raw_selection

                    radial_pinch = normalized_pinch_ratio(control_hand, 8)
                    if radial_pinch < RADIAL_PINCH_ON:
                        if radial.pinch_candidate_at is None:
                            radial.pinch_candidate_at = now
                        if (not radial.pinch_latched and radial.selected is not None and
                                now - radial.pinch_candidate_at >= RADIAL_PINCH_CONFIRM):
                            radial.pinch_latched = True
                            action_label = execute_radial_action(radial.selected)
                            gesture_event = f"RADIAL: {action_label}"
                            gesture_event_until = now + GESTURE_EVENT_SHOW_SECONDS
                            gesture_input_block_until = now + 0.20
                            radial.reset()
                            cursor.sync(True)
                            flow.clear_motion()
                    elif radial_pinch > RADIAL_PINCH_OFF:
                        radial.pinch_latched = False
                        radial.pinch_candidate_at = None

                    if radial.active:
                        open_now = is_radial_open_pose(control_class_hand)
                        if open_now or radial_pinch < RADIAL_PINCH_OFF:
                            radial.release_at = None
                        else:
                            if radial.release_at is None:
                                radial.release_at = now
                            elif now - radial.release_at >= RADIAL_RELEASE_GRACE:
                                radial.reset()
                                gesture_input_block_until = now + 0.12
                                cursor.sync(True)
                                flow.clear_motion()
                else:
                    open_now = is_radial_open_pose(control_class_hand)
                    radial_can_arm = (
                        open_now and not scroll.active and not swipe.tracking and
                        not volume_candidate_now and volume.candidate_at is None and
                        now >= gesture_input_block_until
                    )
                    if radial_can_arm:
                        current_anchor = points[control_index]
                        if radial.candidate_at is None or radial.anchor is None:
                            radial.candidate_at = now
                            radial.anchor = current_anchor
                        else:
                            drift = math.hypot(
                                current_anchor[0] - radial.anchor[0],
                                current_anchor[1] - radial.anchor[1],
                            )
                            if drift > RADIAL_STILL_MAX:
                                radial.candidate_at = now
                                radial.anchor = current_anchor
                            elif now - radial.candidate_at >= RADIAL_OPEN_HOLD:
                                radial.active = True
                                radial.center = current_anchor
                                radial.selected = None
                                radial.selection_candidate = None
                                radial.selection_since = None
                                radial.release_at = None
                                radial.pinch_latched = False
                                radial.pinch_candidate_at = None
                                swipe.tracking = False
                                scroll.reset()
                                cursor.sync(False)
                                flow.clear_motion()
                    else:
                        radial.candidate_at = None
                        radial.anchor = None

                # Spazzata naturale: score temporale della mano piatta/unita +
                # movimento laterale. Non richiede piu' una posa perfetta frame per frame.
                swipe_allowed = (
                    commands_enabled and not spock.blocking and len(hands) == 1 and
                    not paused_by_fist and not volume.active and
                    not volume_candidate_now and volume.candidate_at is None and
                    not two_hand.active and two_hand.candidate_at is None and
                    not radial.active and not scroll.active and
                    now >= gesture_input_block_until
                )
                if swipe_allowed:
                    (swipe.debug_score, swipe.debug_gap,
                     swipe.debug_extended) = swipe_pose_metrics(control_hand)
                    swipe.pose_history.append(swipe.debug_score)
                    best_scores = sorted(swipe.pose_history, reverse=True)[:SWIPE_POSE_KEEP_BEST]
                    swipe.debug_stable = sum(best_scores) / max(len(best_scores), 1)
                    if (swipe.debug_score >= SWIPE_POSE_SCORE_ON or
                            swipe.debug_stable >= SWIPE_POSE_SCORE_ON):
                        swipe.pose_last_seen = now
                    elif (swipe.tracking and
                          swipe.debug_score >= SWIPE_POSE_SCORE_HOLD):
                        # Solo uno swipe gia' partito usa la soglia bassa per
                        # tollerare motion blur senza armare falsamente il mouse.
                        swipe.pose_last_seen = now
                else:
                    swipe.debug_score = 0.0
                    swipe.debug_stable = 0.0
                    swipe.debug_gap = 9.0
                    swipe.debug_extended = 0
                    swipe.pose_history.clear()
                    swipe.pose_last_seen = None

                # MediaPipe ri-ancora i punti rigidi del palmo; LK li porta al frame corrente.
                # Puntatore pinch-only. La mano aperta non muove mai il cursore.
                pointer_allowed = pointer_mode_allowed(
                    commands_enabled=commands_enabled,
                    spock_blocking=spock.blocking,
                    hand_count=len(hands),
                    paused=paused_by_fist,
                    volume_active=volume.active,
                    two_hand_active=two_hand.active,
                    two_hand_candidate=two_hand.candidate_at is not None,
                    radial_active=radial.active,
                    scroll_active=scroll.active,
                    swipe_tracking=swipe.tracking,
                    input_blocked=now < gesture_input_block_until,
                    volume_candidate=(
                        volume_candidate_now or volume.candidate_at is not None
                    ),
                )
                pointer_ratio = normalized_pinch_ratio(control_hand, 8)
                pointer_fingers_valid = pointer_other_fingers_valid(control_hand)
                pointer_pose_on = is_pointer_pinch_pose(control_hand, POINTER_PINCH_ON)

                if pointer.pinch_held:
                    # Se medio/anulare/mignolo si chiudono a pugno, annulla il
                    # puntatore: non deve diventare ne' movimento ne' click.
                    if not pointer_allowed or not pointer_fingers_valid:
                        pointer.reset(preserve_last_click=True)
                        cursor.sync(False)
                    elif pointer_ratio > POINTER_PINCH_OFF:
                        # Congela immediatamente il cursore appena il pinch e' chiaramente
                        # aperto. La grace sotto serve solo a confermare il rilascio.
                        pointer.release_braking = True
                        pointer.move_active = False
                        cursor.sync(False)
                        flow.clear_motion()
                        if pointer.release_at is None:
                            pointer.release_at = now
                        elif now - pointer.release_at >= POINTER_RELEASE_GRACE:
                            pinch_duration = now - (pointer.pinch_started_at or now)
                            quick_click = (
                                pinch_duration <= POINTER_CLICK_MAX_SECONDS and
                                pointer.flow_travel <= POINTER_CLICK_MAX_TRAVEL_PX
                            )
                            if quick_click:
                                # Un click rapido non deve spostare il cursore nemmeno
                                # se i primi frame del pinch hanno prodotto un po' di jitter.
                                if pointer.cursor_origin is not None:
                                    cursor.set_position(
                                        pointer.cursor_origin[0], pointer.cursor_origin[1]
                                    )
                                    cursor.sync(False)
                                is_double_pinch = (
                                    pointer.last_click_at is not None and
                                    now - pointer.last_click_at <= DOUBLE_PINCH_WINDOW
                                )
                                left_click()
                                if is_double_pinch:
                                    gesture_event = "DOPPIO PINCH: DOPPIO CLICK"
                                    pointer.last_click_at = None
                                else:
                                    gesture_event = "PINCH RAPIDO: CLICK"
                                    pointer.last_click_at = now
                                gesture_event_until = now + GESTURE_EVENT_SHOW_SECONDS
                            pointer.reset(preserve_last_click=True)
                            precision_snap_active = False
                            snap_anchor = None
                            snap_started_at = None
                            cursor.sync(False)
                            flow.clear_motion()
                    elif pointer_ratio > POINTER_RELEASE_BRAKE_RATIO:
                        # Pre-rilascio: blocca il cursore prima ancora di raggiungere
                        # la soglia OFF, cosi' l'apertura delle dita non trascina il mouse.
                        pointer.release_braking = True
                        pointer.move_active = False
                        pointer.release_at = None
                        cursor.sync(False)
                        flow.clear_motion()
                    else:
                        pointer.release_braking = False
                        pointer.release_at = None
                elif pointer_allowed and pointer_pose_on:
                    pointer.pinch_held = True
                    pointer.move_active = False
                    pointer.pinch_started_at = now
                    pointer.release_at = None
                    pointer.release_braking = False
                    pointer.motion_accum[:] = 0.0
                    pointer.flow_travel = 0.0
                    pointer.cursor_origin = cursor.position()
                    swipe.cancel_tracking()
                    cursor.sync(False)
                    flow.clear_motion()

                # Il pinch abilita il puntatore, ma la traslazione viene sempre
                # misurata sulla parte rigida del palmo. Cosi' chiudere/aprire indice
                # e pollice non introduce movimento spurio del cursore.
                mp_pts = flow_points_from_hand(control_hand)
                corrected = propagate_points(result_gray, gray, mp_pts)
                if corrected is not None:
                    flow.points = corrected
                    flow.prev_gray = gray
                    flow.active = True

                if paused_by_fist or not commands_enabled or spock.blocking:
                    volume.reset()
                    scroll.reset()
                    two_hand.reset()
                    radial.reset()
                    swipe.tracking = False
                else:
                    if not volume.active:
                        dedicated_mode_block = (
                            pointer.pinch_held or
                            two_hand.active or two_hand.candidate_at is not None or
                            radial.active or swipe.tracking
                        )
                        if dedicated_mode_block:
                            volume.candidate_at = None
                            volume.candidate_last_seen = None
                            volume.vote_history.clear()
                        elif volume_gesture_now or volume_candidate_now:
                            if volume.candidate_at is None:
                                volume.candidate_at = now
                            volume.candidate_last_seen = now
                            if (volume_gesture_now and
                                    now - volume.candidate_at >= VOLUME_CONFIRM_SECONDS):
                                volume.active = True
                                volume.candidate_last_seen = None
                                volume.release_at = None
                                volume.pose_lost_at = None
                                volume.last_angle = palm_roll_angle(control_hand)
                                volume.delta_history.clear()
                                volume.level = get_system_volume()
                                scroll.reset()
                                cursor.sync(False)
                                flow.clear_motion()
                        elif volume.candidate_at is not None:
                            # Un singolo frame incerto non annulla l'aggancio e non lascia
                            # partire click/pugno/scroll nello stesso istante.
                            if (volume.candidate_last_seen is None or
                                    now - volume.candidate_last_seen > VOLUME_ENTRY_MISS_GRACE):
                                volume.candidate_at = None
                                volume.candidate_last_seen = None
                    else:
                        # VOLUME LOCK: appena la mano mostra una chiara apertura,
                        # congela SUBITO il volume. La conferma temporale serve solo
                        # a decidere quando uscire definitivamente dalla modalita'.
                        release_pose = is_volume_release_pose(control_class_hand)
                        fully_open = is_open_hand(control_class_hand)
                        if fist_pending:
                            # Due frame compatti consecutivi: congela il volume ma
                            # aspetta il voto successivo prima di dichiarare PUGNO.
                            volume.pose_lost_at = None
                            volume.release_at = None
                            volume.last_angle = None
                            volume.delta_history.clear()
                        elif release_pose or fully_open:
                            volume.pose_lost_at = None
                            if volume.release_at is None:
                                volume.release_at = now
                            # Nessuna rotazione viene letta durante il rilascio.
                            volume.last_angle = None
                            volume.delta_history.clear()
                            if now - volume.release_at >= VOLUME_RELEASE_GRACE:
                                volume.reset()
                                cursor.sync(True)
                                flow.clear_motion()
                        elif debug_volume_score < VOLUME_HOLD_MIN_SCORE:
                            # La posa non e' piu' credibile: congela immediatamente
                            # il volume e aspetta una breve grazia prima di sganciare.
                            volume.release_at = None
                            if volume.pose_lost_at is None:
                                volume.pose_lost_at = now
                            volume.last_angle = None
                            volume.delta_history.clear()
                            if now - volume.pose_lost_at >= VOLUME_POSE_LOSS_GRACE:
                                volume.reset()
                                cursor.sync(True)
                                flow.clear_motion()
                        else:
                            volume.release_at = None
                            volume.pose_lost_at = None
                            angle = palm_roll_angle(control_hand)
                            if volume.last_angle is None:
                                # Dopo un falso rilascio/rientro, questo frame diventa
                                # il nuovo zero: niente salto di volume.
                                volume.last_angle = angle
                                volume.delta_history.clear()
                            else:
                                raw_delta = wrapped_angle_delta(angle, volume.last_angle)
                                volume.last_angle = angle
                                raw_delta = clamp(raw_delta, -VOLUME_MAX_DELTA_RAD, VOLUME_MAX_DELTA_RAD)
                                volume.delta_history.append(raw_delta)
                                stable_delta = sorted(volume.delta_history)[len(volume.delta_history) // 2]
                                if abs(stable_delta) >= VOLUME_DEADZONE_RAD:
                                    volume.level = clamp(
                                        volume.level + stable_delta * VOLUME_GAIN * VOLUME_DIRECTION,
                                        0.0, 1.0,
                                    )
                                    set_system_volume(volume.level)

                    if (pointer.pinch_held or volume.active or two_hand.active or radial.active or
                            swipe.tracking or two_hand.candidate_at is not None):
                        scroll.reset()
                    elif not scroll.active:
                        if scroll_gesture_now:
                            if scroll.candidate_at is None:
                                scroll.candidate_at = now
                            elif now - scroll.candidate_at >= SCROLL_CONFIRM_SECONDS:
                                scroll.active = True
                                scroll.release_at = None
                                scroll.residual = 0.0
                                cursor.sync(False)
                                flow.clear_motion()
                        else:
                            scroll.candidate_at = None
                    else:
                        if scroll_gesture_now:
                            scroll.release_at = None
                        else:
                            if scroll.release_at is None:
                                scroll.release_at = now
                            elif now - scroll.release_at >= SCROLL_RELEASE_GRACE:
                                scroll.reset()
                                cursor.sync(True)
                                flow.clear_motion()

                if paused_by_fist or not commands_enabled or spock.blocking:
                    cursor.sync(False)
                    flow.clear_motion()
                else:
                    exclusive_cursor_block = (
                        scroll.active or volume.active or two_hand.active or radial.active or
                        swipe.tracking or two_hand.candidate_at is not None
                    )
                    if (pointer.move_active and not exclusive_cursor_block and
                            now >= gesture_input_block_until and
                            (old_pause or not cursor.active)):
                        cursor.sync(True)
                        flow.clear_motion()
                    elif not pointer.move_active and cursor.active:
                        cursor.sync(False)

                    if (pointer.pinch_held and pointer.move_active and
                            not exclusive_cursor_block and corrected is None and
                            old_mp_ref is not None and mp_control_ref is not None and
                            not old_pause and now >= gesture_input_block_until):
                        # Fallback MediaPipe: anche qui usa il centro rigido del palmo,
                        # mai il contatto indice-pollice, per evitare salti durante il gesto.
                        fdx = mp_control_ref[0] - old_mp_ref[0]
                        fdy = mp_control_ref[1] - old_mp_ref[1]
                        fdx, fdy = normalize_flow_delta(fdx, fdy, flow.motion_scale)
                        fmag = math.hypot(fdx, fdy)
                        pointer.flow_travel += fmag * max(DETECTION_W, DETECTION_H)
                        if 0.0005 < fmag < 0.055:
                            fallback_dt = (
                                max(mp_cycle_ms_ema / 1000.0, 1.0 / 60.0)
                                if mp_cycle_ms_ema > 0.0 else 1.0 / 30.0
                            )
                            fallback_speed = fmag * max(DETECTION_W, DETECTION_H) / fallback_dt
                            dynamic_gain = cursor_gain_for_speed(fallback_speed)
                            dx = fdx * screen_w * MOVE_GAIN * MOVEMENT_MULTIPLIER * dynamic_gain
                            dy = fdy * screen_h * MOVE_GAIN * MOVEMENT_MULTIPLIER * dynamic_gain
                            screen_step = math.hypot(float(dx), float(dy))
                            if precision_snap_active:
                                if screen_step <= SNAP_BREAK_DELTA_PX:
                                    dx *= SNAP_HOLD_GAIN
                                    dy *= SNAP_HOLD_GAIN
                                else:
                                    precision_snap_active = False
                                    snap_anchor = None
                                    snap_started_at = None
                            cursor.add_delta(float(dx), float(dy), screen_size=(screen_w, screen_h))
            else:
                spock.debug_score = 0.0
                debug_fist_score = 0.0
                debug_volume_score = 0.0
                debug_grip_gap = 0.0
                debug_fist_folded = 0
                debug_fist_tightness = 2.0
                debug_strong_fist = False
                swipe.debug_score = 0.0
                swipe.debug_stable = 0.0
                swipe.debug_gap = 9.0
                swipe.debug_extended = 0
                if spock.latched:
                    if spock.release_at is None:
                        spock.release_at = now
                    elif now - spock.release_at >= SPOCK_RELEASE_SECONDS:
                        spock.latched = False
                        spock.blocking = False
                        spock.release_at = None
                        spock.progress = 0.0
                        spock.confirmed_seconds = 0.0
                        spock.score_history.clear()
                        spock.debug_stable_score = 0.0
                        gesture_input_block_until = max(
                            gesture_input_block_until,
                            now + SPOCK_POST_RELEASE_BLOCK,
                        )
                elif (spock.candidate_at is not None and spock.last_seen is not None and
                      now - spock.last_seen > SPOCK_MISS_GRACE):
                    spock.candidate_at = None
                    spock.last_seen = None
                    spock.blocking = False
                    spock.progress = 0.0
                    spock.confirmed_seconds = 0.0
                    spock.score_history.clear()
                    spock.debug_stable_score = 0.0

                fist_states = []
                scroll_states = []

                # Il vecchio grace di 280 ms e' utile per non perdere lo stato delle
                # gesture, ma e' troppo lungo per un cursore: se la mano sparisce,
                # LK puo' iniziare a seguire lo sfondo. Congela quindi il puntatore
                # dopo circa due frame e lo riancora quando MediaPipe rivede la mano.
                if (pointer.move_active and
                        now - last_hand_seen > POINTER_TRACKING_LOSS_GRACE):
                    cursor.sync(False)
                    flow.points = None
                    flow.active = False
                    flow.clear_motion()

                loss_grace = VOLUME_TRACKING_LOSS_GRACE if volume.active else TRACKING_LOSS_GRACE
                if now - last_hand_seen > loss_grace:
                    paused_by_fist = False
                    volume.reset()
                    fist_vote_history.clear()
                    scroll.reset()
                    two_hand.reset()
                    radial.reset()
                    swipe.cancel_tracking()
                    pointer.reset(preserve_last_click=True)
                    gesture_input_block_until = 0.0
                    mp_control_ref = None
                    control_handedness = None
                    flow.points = None
                    flow.active = False
                    flow.clear_motion()
                    cursor.sync(False)
        # Se l'optical flow perde i punti, MediaPipe li riaggancia al prossimo risultato valido.
        if not flow.active and now - flow.last_success > TRACKING_LOSS_GRACE:
            flow.points = None

        # Precision snap: quando il cursore resta stabile lo trattiene leggermente.
        snap_allowed = (
            pointer.pinch_held and pointer.move_active and
            commands_enabled and not spock.blocking and not paused_by_fist and
            latest_result is not None and bool(latest_result.hand_landmarks) and
            not volume.active and volume.candidate_at is None and not scroll.active and
            not two_hand.active and two_hand.candidate_at is None and
            not radial.active and not swipe.tracking and
            now >= gesture_input_block_until
        )
        if snap_allowed:
            cursor_x, cursor_y = cursor.position()
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

        pinch_now = pointer.pinch_held
        gesture_mode = resolve_runtime_mode(
            commands_enabled=commands_enabled,
            spock_blocking=spock.blocking,
            paused=paused_by_fist,
            volume=volume.active,
            two_hand=two_hand.active,
            radial=radial.active,
            scrolling=scroll.active,
            swipe=swipe.tracking,
            pointer_move=pointer.move_active,
            pointer_pinch=pointer.pinch_held,
        )
        if latest_result is not None and latest_result.hand_landmarks:
            for i, hand in enumerate(latest_result.hand_landmarks):
                paused = fist_states[i] if i < len(fist_states) else False
                draw_hand(frame, hand,
                          pinch_active=(i == control_index and pinch_now),
                          paused=paused,
                          scrolling=(i == control_index and scroll.active),
                          volume_control=(i == control_index and volume.active))

        if radial.active and radial.center is not None:
            draw_radial_menu(frame, radial.center, radial.selected)
        if two_hand.active and two_hand.points is not None:
            draw_two_hand_transform(
                frame, two_hand.points[0], two_hand.points[1],
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
            status = f"SPOCK: {int(round(spock.progress * 100))}% | tieni 1 s per TOGGLE"
        elif gesture_mode == "LOCKED":
            status = "COMANDI BLOCCATI | fai SPOCK per 1 s"
        elif gesture_mode == "FIST":
            status = "PUGNO = PAUSA TRACKING"
        elif gesture_mode == "VOLUME":
            status = f"VOLUME: {int(round(volume.level * 100))}%"
        elif gesture_mode == "TWO_HAND":
            status = "2 MANI: ZOOM"
        elif gesture_mode == "RADIAL":
            choice = radial.selected if radial.selected is not None else "CENTRO"
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
        flow_label = "FLOW ON" if flow.active else "FLOW WAIT"
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
            frame, f"ENGINE: {gesture_mode} | SPOCK raw {spock.debug_score:.2f} stable {spock.debug_stable_score:.2f}",
            (30, 222), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (200, 255, 200), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"SWIPE raw {swipe.debug_score:.2f} stable {swipe.debug_stable:.2f} | JOIN {swipe.debug_gap:.2f} | EXT {swipe.debug_extended}/4",
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
        if spock.blocking:
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

        if spock.blocking and not spock.latched:
            bar_w, bar_h = 260, 16
            bar_x, bar_y = frame_w - bar_w - 30, 108
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (255, 255, 255), 2)
            fill_w = int(bar_w * clamp(spock.progress, 0.0, 1.0))
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
