"""Hand tracking runtime implementation."""

import math
import time

import cv2

from handtracking_camera import CameraRuntime
from handtracking_core import (
    choose_camera_target_fps,
    normalize_flow_delta,
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
from handtracking_handlers import (
    update_pointer_state,
    update_radial_state,
    update_two_hand_state,
)
from handtracking_processing import (
    analyze_hand_frame,
    update_hand_mode_metrics,
    update_ema_metrics,
    update_precision_snap,
    update_spock_state,
    update_swipe_pose,
)
from handtracking_scroll import update_scroll_state
from handtracking_config import *
from handtracking_gestures import (
    control_point,
    normalized_pinch_ratio,
    spock_all_fingers_up,
    spock_pose_score,
    two_hand_geometry,
)
from handtracking_render import draw_runtime_overlays
from handtracking_session import RuntimeSession
from handtracking_tracking import (
    apply_stale_fail_safe,
    expire_lost_flow,
    handle_missing_hands,
)
from handtracking_volume import update_volume_state
from handtracking_windows import (
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

def _run_impl(session):
    camera = session.camera
    reported_fps = camera.reported_fps
    reported_w = camera.reported_w
    reported_h = camera.reported_h
    camera_codec = camera.codec
    camera_target_fps = camera.target_fps

    start_time = session.start_time
    mp_worker = session.worker
    cursor = session.cursor
    screen_w, screen_h = session.screen_w, session.screen_h

    flow = session.flow
    pointer = session.pointer
    scroll = session.scroll
    volume = session.volume
    swipe = session.swipe
    radial = session.radial
    two_hand = session.two_hand
    spock = session.spock

    latest_result = session.latest_result
    latest_result_seq = session.latest_result_seq
    control_index = session.control_index
    control_handedness = session.control_handedness
    mp_control_ref = session.mp_control_ref
    fist_states = session.fist_states
    paused_by_fist = session.paused_by_fist
    fist_vote_history = session.fist_vote_history
    debug_fist_score = session.debug_fist_score
    debug_volume_score = session.debug_volume_score
    debug_grip_gap = session.debug_grip_gap
    debug_fist_folded = session.debug_fist_folded
    debug_fist_tightness = session.debug_fist_tightness
    debug_strong_fist = session.debug_strong_fist

    # Stato del Gesture Engine centrale.
    gesture_mode = session.gesture_mode
    gesture_event = session.gesture_event
    gesture_event_until = session.gesture_event_until
    gesture_input_block_until = session.gesture_input_block_until

    # Gate globale dei comandi. Il tracking resta sempre acceso.
    commands_enabled = session.commands_enabled

    # Precision snap: stabilizza il cursore quando la mano rallenta.
    snap_anchor = session.snap_anchor
    snap_started_at = session.snap_started_at
    precision_snap_active = session.precision_snap_active
    last_hand_seen = session.last_hand_seen
    fps_window_start = session.fps_window_start
    fps_frames = session.fps_frames
    actual_fps = session.actual_fps
    mp_fps_window_start = session.mp_fps_window_start
    mp_fps_last_seq = session.mp_fps_last_seq
    actual_mp_fps = session.actual_mp_fps
    mp_infer_ms_ema = session.mp_infer_ms_ema
    mp_worker_ms_ema = session.mp_worker_ms_ema
    mp_cycle_ms_ema = session.mp_cycle_ms_ema
    mp_queue_ms_ema = session.mp_queue_ms_ema
    while True:
        prepared = camera.read_prepared()
        if prepared is None:
            break
        now = time.perf_counter()
        frame = prepared.frame
        detect_frame = prepared.detect_frame
        gray = prepared.gray

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
            stale = apply_stale_fail_safe(
                spock=spock,
                fist_vote_history=fist_vote_history,
                volume=volume,
                scroll=scroll,
                two_hand=two_hand,
                radial=radial,
                swipe=swipe,
                pointer=pointer,
                flow=flow,
                cursor=cursor,
            )
            paused_by_fist = stale.paused_by_fist
            mp_control_ref = stale.mp_control_ref
            control_handedness = stale.control_handedness
            latest_result = stale.latest_result
            fist_states = stale.fist_states
            debug_fist_score = stale.debug_fist_score
            debug_volume_score = stale.debug_volume_score
            debug_grip_gap = stale.debug_grip_gap
            debug_fist_folded = stale.debug_fist_folded
            debug_fist_tightness = stale.debug_fist_tightness
            debug_strong_fist = stale.debug_strong_fist
            snap_anchor = stale.snap_anchor
            snap_started_at = stale.snap_started_at
            precision_snap_active = stale.precision_snap_active

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

                old_mp_ref = mp_control_ref
                analysis = analyze_hand_frame(
                    latest_result=latest_result,
                    hands=hands,
                    class_hands=class_hands,
                    previous_point=mp_control_ref,
                    previous_label=control_handedness,
                    paused_by_fist=paused_by_fist,
                    fist_vote_history=fist_vote_history,
                    volume_active=volume.active,
                )
                paused_by_fist = analysis.paused_by_fist
                old_pause = analysis.old_pause
                fist_pending = analysis.fist_pending
                control_index = analysis.control_index
                control_distance = analysis.control_distance
                selected_handedness = analysis.selected_handedness

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
                control_hand = analysis.control_hand
                control_class_hand = analysis.control_class_hand
                points = analysis.points
                mode_metrics = update_hand_mode_metrics(
                    analysis,
                    hands=hands,
                    volume=volume,
                    flow=flow,
                )
                volume_gesture_now = mode_metrics.volume_gesture_now
                volume_candidate_now = mode_metrics.volume_candidate_now
                fist_states = mode_metrics.fist_states
                scroll_gesture_now = mode_metrics.scroll_gesture_now
                debug_fist_score = mode_metrics.debug_fist_score
                debug_volume_score = mode_metrics.debug_volume_score
                debug_grip_gap = mode_metrics.debug_grip_gap
                debug_fist_folded = mode_metrics.debug_fist_folded
                debug_fist_tightness = mode_metrics.debug_fist_tightness
                debug_strong_fist = mode_metrics.debug_strong_fist
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

                gesture_input_block_until = update_two_hand_state(
                    two_hand,
                    held=two_hand_held,
                    pair_geometry=pair_geometry,
                    now=now,
                    input_block_until=gesture_input_block_until,
                    radial=radial,
                    swipe=swipe,
                    scroll=scroll,
                    volume=volume,
                    cursor=cursor,
                    flow=flow,
                    ctrl_wheel_cb=ctrl_wheel,
                )

                # Menu radiale: mano aperta e quasi ferma per ~1 s. Il centro
                # resta fisso; sposta la mano verso una voce e fai pinch per selezionare.
                radial_priority_block = (
                    not commands_enabled or spock.blocking or paused_by_fist or
                    volume.active or two_hand.active or
                    two_hand.candidate_at is not None or len(hands) != 1
                )
                radial_result = update_radial_state(
                    radial,
                    now=now,
                    priority_block=radial_priority_block,
                    control_hand=control_hand,
                    control_class_hand=control_class_hand,
                    current_anchor=points[control_index],
                    volume_candidate_now=volume_candidate_now,
                    volume_candidate_at=volume.candidate_at,
                    input_block_until=gesture_input_block_until,
                    scroll=scroll,
                    swipe=swipe,
                    cursor=cursor,
                    flow=flow,
                    execute_action_cb=execute_radial_action,
                )
                gesture_input_block_until = radial_result.input_block_until
                if radial_result.event is not None:
                    gesture_event = radial_result.event
                    gesture_event_until = radial_result.event_until

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
                update_swipe_pose(
                    swipe,
                    allowed=swipe_allowed,
                    control_hand=control_hand,
                    now=now,
                )

                pointer_result = update_pointer_state(
                    pointer,
                    control_hand=control_hand,
                    now=now,
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
                    cursor=cursor,
                    flow=flow,
                    swipe=swipe,
                    precision_snap_active=precision_snap_active,
                    snap_anchor=snap_anchor,
                    snap_started_at=snap_started_at,
                    left_click_cb=left_click,
                )
                if pointer_result.event is not None:
                    gesture_event = pointer_result.event
                    gesture_event_until = pointer_result.event_until
                precision_snap_active = pointer_result.precision_snap_active
                snap_anchor = pointer_result.snap_anchor
                snap_started_at = pointer_result.snap_started_at

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
                    dedicated_mode_block = (
                        pointer.pinch_held or
                        two_hand.active or two_hand.candidate_at is not None or
                        radial.active or swipe.tracking
                    )
                    update_volume_state(
                        volume,
                        now=now,
                        dedicated_mode_block=dedicated_mode_block,
                        volume_gesture_now=volume_gesture_now,
                        volume_candidate_now=volume_candidate_now,
                        control_hand=control_hand,
                        control_class_hand=control_class_hand,
                        fist_pending=fist_pending,
                        debug_volume_score=debug_volume_score,
                        scroll=scroll,
                        cursor=cursor,
                        flow=flow,
                        get_volume_cb=get_system_volume,
                        set_volume_cb=set_system_volume,
                    )
                    update_scroll_state(
                        scroll,
                        now=now,
                        gesture_now=scroll_gesture_now,
                        blocked=(
                            pointer.pinch_held or volume.active or
                            two_hand.active or radial.active or swipe.tracking or
                            two_hand.candidate_at is not None
                        ),
                        cursor=cursor,
                        flow=flow,
                    )

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

                missing = handle_missing_hands(
                    now=now,
                    last_hand_seen=last_hand_seen,
                    pointer=pointer,
                    volume=volume,
                    scroll=scroll,
                    two_hand=two_hand,
                    radial=radial,
                    swipe=swipe,
                    flow=flow,
                    cursor=cursor,
                    fist_vote_history=fist_vote_history,
                    paused_by_fist=paused_by_fist,
                    gesture_input_block_until=gesture_input_block_until,
                    mp_control_ref=mp_control_ref,
                    control_handedness=control_handedness,
                )
                paused_by_fist = missing.paused_by_fist
                gesture_input_block_until = missing.gesture_input_block_until
                mp_control_ref = missing.mp_control_ref
                control_handedness = missing.control_handedness
                fist_states = missing.fist_states
        expire_lost_flow(flow, now=now)

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
        snap_update = update_precision_snap(
            allowed=snap_allowed,
            cursor_position=cursor.position() if snap_allowed else None,
            now=now,
            active=precision_snap_active,
            anchor=snap_anchor,
            started_at=snap_started_at,
        )
        precision_snap_active = snap_update.active
        snap_anchor = snap_update.anchor
        snap_started_at = snap_update.started_at

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
        draw_runtime_overlays(
            frame,
            latest_result=latest_result,
            fist_states=fist_states,
            control_index=control_index,
            pinch_active=pointer.pinch_held,
            scroll_active=scroll.active,
            volume_active=volume.active,
            radial_active=radial.active,
            radial_center=radial.center,
            radial_selected=radial.selected,
            two_hand_active=two_hand.active,
            two_hand_points=two_hand.points,
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
        if not camera.show(frame):
            break

    return None


def run():
    camera = CameraRuntime.open()
    session = None
    try:
        session = RuntimeSession.create(camera=camera)
        return _run_impl(session)
    finally:
        if session is not None:
            session.close()
        else:
            camera.close()


if __name__ == "__main__":
    run()
