"""Hand tracking runtime implementation."""

import math
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from handtracking_core import (
    choose_camera_target_fps,
    choose_control_index,
    fist_evidence_from_hands,
    normalize_flow_delta,
    normalized_points_pixel_distance,
    palm_motion_scale,
    pointer_mode_allowed,
    tracking_result_is_stale,
)
from handtracking_engine import resolve_runtime_mode
from handtracking_flow import (
    cursor_gain_for_speed,
    dispatch_flow_motion,
    flow_points_from_hand,
    measure_optical_flow,
    propagate_points,
)
from handtracking_hud import draw_runtime_hud
from handtracking_mediapipe import MediaPipeWorker
from handtracking_processing import update_ema_metrics, update_spock_state
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
    mouse_wheel,
    set_system_volume,
)

# L'optical flow usa pochissimi punti e non beneficia di 8 thread OpenCV.
# Limitarlo evita contesa CPU con MediaPipe/XNNPACK sul portatile 4C/8T.
cv2.setNumThreads(1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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
        flow_motion = measure_optical_flow(
            flow.prev_gray, gray, flow.points, flow.motion_scale
        )
        if flow_motion is not None:
            flow.points = flow_motion.next_points
            flow.active = True
            flow.last_success = now
            flow_result = dispatch_flow_motion(
                motion_dx=flow_motion.dx,
                motion_dy=flow_motion.dy,
                motion_mag=flow_motion.magnitude,
                now=now,
                mp_result_stale=mp_result_stale,
                paused_by_fist=paused_by_fist,
                commands_enabled=commands_enabled,
                spock_blocking=spock.blocking,
                gesture_input_block_until=gesture_input_block_until,
                pointer=pointer,
                volume=volume,
                two_hand=two_hand,
                radial=radial,
                scroll=scroll,
                swipe=swipe,
                flow=flow,
                cursor=cursor,
                screen_w=screen_w,
                screen_h=screen_h,
                precision_snap_active=precision_snap_active,
                snap_anchor=snap_anchor,
                snap_started_at=snap_started_at,
                execute_swipe_cb=execute_swipe,
                mouse_wheel_cb=mouse_wheel,
            )
            if flow_result.gesture_event is not None:
                gesture_event = flow_result.gesture_event
                gesture_event_until = flow_result.gesture_event_until
            gesture_input_block_until = flow_result.gesture_input_block_until
            precision_snap_active = flow_result.precision_snap_active
            snap_anchor = flow_result.snap_anchor
            snap_started_at = flow_result.snap_started_at
        elif flow.prev_gray is not None and flow.points is not None:
            flow.active = False
        flow.prev_gray = gray
        # Consuma solo risultati MediaPipe nuovi; l'inferenza non blocca il loop camera.
        new_mp = False
        if packet is not None and packet[0] != latest_result_seq:
            (latest_result_seq, latest_result, result_gray,
             mp_infer_ms, mp_worker_ms, mp_cycle_ms, mp_queue_ms) = packet
            new_mp = True

        if new_mp:
            (mp_infer_ms_ema, mp_worker_ms_ema,
             mp_cycle_ms_ema, mp_queue_ms_ema) = update_ema_metrics(
                (mp_infer_ms_ema, mp_worker_ms_ema, mp_cycle_ms_ema, mp_queue_ms_ema),
                (mp_infer_ms, mp_worker_ms, mp_cycle_ms, mp_queue_ms),
            )
            if latest_result.hand_landmarks:
                hands = latest_result.hand_landmarks
                world_hands = getattr(latest_result, "hand_world_landmarks", None)
                if not world_hands or len(world_hands) != len(hands):
                    class_hands = hands
                else:
                    class_hands = world_hands

                spock_scores_now = [spock_pose_score(hand) for hand in hands]
                upright_now = any(spock_all_fingers_up(hand) for hand in hands)
                spock_sample_seconds = min(
                    SPOCK_SAMPLE_MAX_SECONDS,
                    max(
                        (mp_cycle_ms / 1000.0) if mp_cycle_ms > 0.0
                        else 1.0 / max(camera_target_fps, 1),
                        1.0 / 120.0,
                    ),
                )
                spock_update = update_spock_state(
                    spock,
                    raw_score=max(spock_scores_now, default=0.0),
                    upright_now=upright_now,
                    now=now,
                    sample_seconds=spock_sample_seconds,
                    commands_enabled=commands_enabled,
                    input_block_until=gesture_input_block_until,
                )
                commands_enabled = spock_update.commands_enabled
                gesture_input_block_until = spock_update.input_block_until
                if spock_update.event is not None:
                    gesture_event = spock_update.event
                    gesture_event_until = spock_update.event_until
                if spock_update.released:
                    mp_control_ref = None
                    control_handedness = None
                    flow.points = None
                    flow.active = False
                    cursor.sync(False)

                if spock_update.toggled:
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

        draw_runtime_hud(
            frame,
            gesture_mode=gesture_mode,
            gesture_event=gesture_event,
            gesture_event_until=gesture_event_until,
            now=now,
            flow_active=flow.active,
            commands_enabled=commands_enabled,
            spock_blocking=spock.blocking,
            spock_latched=spock.latched,
            spock_progress=spock.progress,
            volume_level=volume.level,
            radial_selected=radial.selected,
            actual_fps=actual_fps,
            actual_mp_fps=actual_mp_fps,
            mp_infer_ms=mp_infer_ms_ema,
            mp_worker_ms=mp_worker_ms_ema,
            mp_cycle_ms=mp_cycle_ms_ema,
            mp_queue_ms=mp_queue_ms_ema,
            mp_overwrites=mp_overwrites,
            mp_input_seq=mp_input_seq,
            camera_codec=camera_codec,
            reported_w=reported_w,
            reported_h=reported_h,
            reported_fps=reported_fps,
            camera_target_fps=camera_target_fps,
            debug_fist_score=debug_fist_score,
            debug_volume_score=debug_volume_score,
            debug_grip_gap=debug_grip_gap,
            debug_fist_folded=debug_fist_folded,
            debug_fist_tightness=debug_fist_tightness,
            debug_strong_fist=debug_strong_fist,
            spock_debug_score=spock.debug_score,
            spock_debug_stable=spock.debug_stable_score,
            swipe_debug_score=swipe.debug_score,
            swipe_debug_stable=swipe.debug_stable,
            swipe_debug_gap=swipe.debug_gap,
            swipe_debug_extended=swipe.debug_extended,
            mp_error_count=mp_error_count,
            mp_last_error=mp_last_error,
        )
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
