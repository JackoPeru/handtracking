"""Hand tracking runtime implementation."""

import time

import cv2

from handtracking_camera import CameraRuntime
from handtracking_core import (
    choose_camera_target_fps,
    tracking_result_is_stale,
)
from handtracking_engine import resolve_runtime_mode
from handtracking_frame import process_mediapipe_packet
from handtracking_flow import (
    dispatch_flow_motion,
    measure_optical_flow,
)
from handtracking_hud import draw_runtime_hud
from handtracking_processing import update_precision_snap
from handtracking_config import *
from handtracking_render import draw_runtime_overlays
from handtracking_session import RuntimeSession
from handtracking_tracking import (
    apply_stale_fail_safe,
    expire_lost_flow,
)
from handtracking_windows import (
    execute_swipe,
    mouse_wheel,
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

        session.latest_result = latest_result
        session.latest_result_seq = latest_result_seq
        session.control_index = control_index
        session.control_handedness = control_handedness
        session.mp_control_ref = mp_control_ref
        session.fist_states = fist_states
        session.paused_by_fist = paused_by_fist
        session.debug_fist_score = debug_fist_score
        session.debug_volume_score = debug_volume_score
        session.debug_grip_gap = debug_grip_gap
        session.debug_fist_folded = debug_fist_folded
        session.debug_fist_tightness = debug_fist_tightness
        session.debug_strong_fist = debug_strong_fist
        session.gesture_event = gesture_event
        session.gesture_event_until = gesture_event_until
        session.gesture_input_block_until = gesture_input_block_until
        session.commands_enabled = commands_enabled
        session.snap_anchor = snap_anchor
        session.snap_started_at = snap_started_at
        session.precision_snap_active = precision_snap_active
        session.last_hand_seen = last_hand_seen
        session.mp_infer_ms_ema = mp_infer_ms_ema
        session.mp_worker_ms_ema = mp_worker_ms_ema
        session.mp_cycle_ms_ema = mp_cycle_ms_ema
        session.mp_queue_ms_ema = mp_queue_ms_ema

        frame_result = process_mediapipe_packet(
            session,
            packet,
            gray=gray,
            now=now,
            camera_target_fps=camera_target_fps,
        )
        if frame_result.processed:
            latest_result = session.latest_result
            latest_result_seq = session.latest_result_seq
            control_index = session.control_index
            control_handedness = session.control_handedness
            mp_control_ref = session.mp_control_ref
            fist_states = session.fist_states
            paused_by_fist = session.paused_by_fist
            debug_fist_score = session.debug_fist_score
            debug_volume_score = session.debug_volume_score
            debug_grip_gap = session.debug_grip_gap
            debug_fist_folded = session.debug_fist_folded
            debug_fist_tightness = session.debug_fist_tightness
            debug_strong_fist = session.debug_strong_fist
            gesture_event = session.gesture_event
            gesture_event_until = session.gesture_event_until
            gesture_input_block_until = session.gesture_input_block_until
            commands_enabled = session.commands_enabled
            snap_anchor = session.snap_anchor
            snap_started_at = session.snap_started_at
            precision_snap_active = session.precision_snap_active
            last_hand_seen = session.last_hand_seen
            mp_infer_ms_ema = session.mp_infer_ms_ema
            mp_worker_ms_ema = session.mp_worker_ms_ema
            mp_cycle_ms_ema = session.mp_cycle_ms_ema
            mp_queue_ms_ema = session.mp_queue_ms_ema
        if frame_result.skip_frame:
            continue

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
