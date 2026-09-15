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
    commit_flow_measurement,
    dispatch_flow_motion,
    measure_optical_flow,
    should_measure_optical_flow,
)
from handtracking_hud import draw_runtime_hud
from handtracking_processing import update_precision_snap
from handtracking_config import *
from handtracking_render import draw_runtime_overlays
from handtracking_session import RuntimeSession
from handtracking_settings import RuntimeSettings
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

def _run_impl(session, *, now_fn=None, measure_flow_fn=None,
              process_packet_fn=None, execute_swipe_fn=None, mouse_wheel_fn=None):
    now_fn = now_fn or time.perf_counter
    measure_flow_fn = measure_flow_fn or measure_optical_flow
    process_packet_fn = process_packet_fn or (
        session.trace.process_packet if session.trace else process_mediapipe_packet
    )
    execute_swipe_fn = execute_swipe_fn or (
        session.trace.execute_swipe if session.trace else execute_swipe
    )
    mouse_wheel_fn = mouse_wheel_fn or (
        session.trace.mouse_wheel if session.trace else mouse_wheel
    )
    camera = session.camera
    mp_worker = session.worker
    cursor = session.cursor
    flow = session.flow
    pointer = session.pointer
    scroll = session.scroll
    volume = session.volume
    swipe = session.swipe
    radial = session.radial
    two_hand = session.two_hand
    spock = session.spock
    perf = session.perf

    while True:
        loop_started = perf.now_ns()
        camera_started = perf.now_ns()
        frame = camera.read_frame()
        perf.observe_ns("camera", camera_started)
        if frame is None:
            break
        now = now_fn()
        cursor.set_input_time(now)
        mp_state = mp_worker.snapshot_state()
        packet = mp_state["latest"]
        if not mp_state["alive"]:
            detail = mp_state["last_error"] or "unknown worker failure"
            raise RuntimeError(f"MediaPipe worker stopped: {detail}")
        if not cursor.running:
            detail = cursor.last_error or "unknown cursor worker failure"
            raise RuntimeError(f"Cursor worker stopped: {detail}")
        session.mp_input_seq = mp_state["input_seq"]
        session.mp_overwrites = mp_state["overwrites"]
        session.mp_error_count = mp_state["error_count"]
        session.mp_last_error = mp_state["last_error"]
        mp_result_stale = tracking_result_is_stale(
            mp_state["last_result_input_at"], now, MP_RESULT_STALE_SECONDS,
        )
        if session.trace:
            session.trace.start_frame(now, mp_state, cursor.position())
        if not mp_result_stale:
            if cursor.renew_freshness(mp_state["last_result_input_at"]) is False:
                mp_result_stale = True

        if mp_result_stale:
            stale = apply_stale_fail_safe(
                spock=spock,
                fist_vote_history=session.fist_vote_history,
                volume=volume,
                scroll=scroll,
                two_hand=two_hand,
                radial=radial,
                swipe=swipe,
                pointer=pointer,
                flow=flow,
                cursor=cursor,
            )
            session.paused_by_fist = stale.paused_by_fist
            session.mp_control_ref = stale.mp_control_ref
            session.control_handedness = stale.control_handedness
            session.latest_result = stale.latest_result
            session.fist_states = stale.fist_states
            session.debug_fist_score = stale.debug_fist_score
            session.debug_volume_score = stale.debug_volume_score
            session.debug_grip_gap = stale.debug_grip_gap
            session.debug_fist_folded = stale.debug_fist_folded
            session.debug_fist_tightness = stale.debug_fist_tightness
            session.debug_strong_fist = stale.debug_strong_fist
            session.snap_anchor = stale.snap_anchor
            session.snap_started_at = stale.snap_started_at
            session.precision_snap_active = stale.precision_snap_active

        submit_due = session.mp_scheduler.should_submit(
            now,
            cycle_ms=session.mp_cycle_ms_ema,
            target_fps=session.camera_target_fps or TARGET_FPS,
        )
        flow_due = should_measure_optical_flow(
            now=now,
            mp_result_stale=mp_result_stale,
            paused_by_fist=session.paused_by_fist,
            commands_enabled=session.commands_enabled,
            spock_blocking=spock.blocking,
            gesture_input_block_until=session.gesture_input_block_until,
            pointer=pointer,
            volume=volume,
            two_hand=two_hand,
            radial=radial,
            scroll=scroll,
            swipe=swipe,
            flow=flow,
        )
        packet_new = packet is not None and packet[0] != session.latest_result_seq
        detect_frame = None
        gray = None
        if (submit_due or flow_due or
                (packet_new and not mp_result_stale and session.commands_enabled)):
            preprocess_started = perf.now_ns()
            detect_frame, gray = camera.prepare_detection(frame)
            perf.observe_ns("preprocess", preprocess_started)

        if submit_due:
            ts = int((now - session.start_time) * 1000)
            # Il worker mantiene i riferimenti ai frame senza copie aggiuntive.
            mp_worker.submit(detect_frame, gray, ts, now)

        flow_motion = None
        if flow_due:
            flow_started = perf.now_ns()
            flow_motion = measure_flow_fn(
                flow.prev_gray, gray, flow.points, flow.motion_scale
            )
            perf.observe_ns("flow", flow_started)
            commit_flow_measurement(flow, gray, flow_motion, now=now)
        if session.trace:
            session.trace.set_motion(flow_motion)
        # LK measurement has no OS side effects; fresh safety state must be
        # applied before dispatch consumes the measured motion.
        mp_process_started = perf.now_ns()
        frame_result = process_packet_fn(
            session,
            packet,
            gray=gray,
            now=now,
            camera_target_fps=session.camera_target_fps,
            mp_result_stale=mp_result_stale,
            allow_pointer_fallback=flow_motion is None,
        )
        if frame_result.processed:
            perf.observe_ns("mp_process", mp_process_started)
        if frame_result.skip_frame:
            if session.trace:
                session.trace.finish_frame(session)
            perf.observe_ns("loop", loop_started)
            continue

        if flow_motion is not None:
            flow_result = dispatch_flow_motion(
                motion_dx=flow_motion.dx,
                motion_dy=flow_motion.dy,
                motion_mag=flow_motion.magnitude,
                now=now,
                mp_result_stale=mp_result_stale,
                paused_by_fist=session.paused_by_fist,
                commands_enabled=session.commands_enabled,
                spock_blocking=spock.blocking,
                gesture_input_block_until=session.gesture_input_block_until,
                pointer=pointer,
                volume=volume,
                two_hand=two_hand,
                radial=radial,
                scroll=scroll,
                swipe=swipe,
                flow=flow,
                cursor=cursor,
                screen_w=session.screen_w,
                screen_h=session.screen_h,
                precision_snap_active=session.precision_snap_active,
                snap_anchor=session.snap_anchor,
                snap_started_at=session.snap_started_at,
                execute_swipe_cb=execute_swipe_fn,
                mouse_wheel_cb=mouse_wheel_fn,
                move_gain=session.settings.move_gain,
            )
            if flow_result.gesture_event is not None:
                session.gesture_event = flow_result.gesture_event
                session.gesture_event_until = flow_result.gesture_event_until
            session.gesture_input_block_until = flow_result.gesture_input_block_until
            session.precision_snap_active = flow_result.precision_snap_active
            session.snap_anchor = flow_result.snap_anchor
            session.snap_started_at = flow_result.snap_started_at

        expire_lost_flow(flow, now=now)

        snap_allowed = (
            pointer.pinch_held and pointer.move_active and
            session.commands_enabled and not spock.blocking and not session.paused_by_fist and
            session.latest_result is not None and bool(session.latest_result.hand_landmarks) and
            not volume.active and volume.candidate_at is None and not scroll.active and
            not two_hand.active and two_hand.candidate_at is None and
            not radial.active and not swipe.tracking and
            now >= session.gesture_input_block_until
        )
        snap_update = update_precision_snap(
            allowed=snap_allowed,
            cursor_position=cursor.position() if snap_allowed else None,
            now=now,
            active=session.precision_snap_active,
            anchor=session.snap_anchor,
            started_at=session.snap_started_at,
        )
        session.precision_snap_active = snap_update.active
        session.snap_anchor = snap_update.anchor
        session.snap_started_at = snap_update.started_at

        session.gesture_mode = resolve_runtime_mode(
            commands_enabled=session.commands_enabled,
            spock_blocking=spock.blocking,
            paused=session.paused_by_fist,
            volume=volume.active,
            two_hand=two_hand.active,
            radial=radial.active,
            scrolling=scroll.active,
            swipe=swipe.tracking,
            pointer_move=pointer.move_active,
            pointer_pinch=pointer.pinch_held,
        )
        session.fps_frames += 1
        elapsed = now - session.fps_window_start
        if elapsed >= 0.5:
            session.actual_fps = session.fps_frames / elapsed
            session.camera_target_fps = choose_camera_target_fps(
                session.actual_fps, TARGET_FPS, FALLBACK_FPS,
            )
            session.fps_frames = 0
            session.fps_window_start = now

        mp_elapsed = now - session.mp_fps_window_start
        if mp_elapsed >= 0.5:
            completed_seq = mp_state["seq"]
            session.actual_mp_fps = (completed_seq - session.mp_fps_last_seq) / mp_elapsed
            session.mp_fps_last_seq = completed_seq
            session.mp_fps_window_start = now

        if session.trace:
            session.trace.finish_frame(session)
        if not session.render_enabled:
            perf.observe_ns("loop", loop_started)
            continue
        render_started = perf.now_ns()
        draw_runtime_overlays(
            frame,
            latest_result=session.latest_result,
            fist_states=session.fist_states,
            control_index=session.control_index,
            pinch_active=pointer.pinch_held,
            scroll_active=scroll.active,
            volume_active=volume.active,
            radial_active=radial.active,
            radial_center=radial.center,
            radial_selected=radial.selected,
            two_hand_active=two_hand.active,
            two_hand_points=two_hand.points,
        )

        if session.hud_layer.should_refresh(frame, now):
            hud = session.hud_layer.begin(frame)
            perf_camera = perf.metric("camera")
            perf_preprocess = perf.metric("preprocess")
            perf_flow = perf.metric("flow")
            perf_mp_process = perf.metric("mp_process")
            perf_render = perf.metric("render")
            perf_loop = perf.metric("loop")
            draw_runtime_hud(
                hud,
                gesture_mode=session.gesture_mode,
                gesture_event=session.gesture_event,
                gesture_event_until=session.gesture_event_until,
                now=now,
                flow_active=flow.active,
                commands_enabled=session.commands_enabled,
                spock_blocking=spock.blocking,
                spock_latched=spock.latched,
                spock_progress=spock.progress,
                volume_level=volume.level,
                radial_selected=radial.selected,
                actual_fps=session.actual_fps,
                actual_mp_fps=session.actual_mp_fps,
                mp_infer_ms=session.mp_infer_ms_ema,
                mp_worker_ms=session.mp_worker_ms_ema,
                mp_cycle_ms=session.mp_cycle_ms_ema,
                mp_queue_ms=session.mp_queue_ms_ema,
                mp_overwrites=session.mp_overwrites,
                mp_input_seq=session.mp_input_seq,
                camera_codec=camera.codec,
                reported_w=camera.reported_w,
                reported_h=camera.reported_h,
                reported_fps=camera.reported_fps,
                camera_target_fps=session.camera_target_fps,
                debug_fist_score=session.debug_fist_score,
                debug_volume_score=session.debug_volume_score,
                debug_grip_gap=session.debug_grip_gap,
                debug_fist_folded=session.debug_fist_folded,
                debug_fist_tightness=session.debug_fist_tightness,
                debug_strong_fist=session.debug_strong_fist,
                spock_debug_score=spock.debug_score,
                spock_debug_stable=spock.debug_stable_score,
                swipe_debug_score=swipe.debug_score,
                swipe_debug_stable=swipe.debug_stable,
                swipe_debug_gap=swipe.debug_gap,
                swipe_debug_extended=swipe.debug_extended,
                mp_error_count=session.mp_error_count,
                mp_last_error=session.mp_last_error,
                perf_camera_ms=perf_camera.ema_ms,
                perf_preprocess_ms=perf_preprocess.ema_ms,
                perf_flow_ms=perf_flow.ema_ms,
                perf_mp_process_ms=perf_mp_process.ema_ms,
                perf_render_ms=perf_render.ema_ms,
                perf_loop_ms=perf_loop.ema_ms,
                loop_metric=perf_loop,
                cursor_metric=cursor.output_latency(),
                profile=session.settings.profile,
            )
            session.hud_layer.finish(now)
        session.hud_layer.apply(frame)
        perf.observe_ns("render", render_started)
        if not camera.show(frame):
            break
        perf.observe_ns("loop", loop_started)

    return None


def run(*, settings=None, record_path=None, record_max_frames=1800):
    settings = settings if settings is not None else RuntimeSettings()
    camera = CameraRuntime.open(camera_index=settings.camera_index)
    session = None
    try:
        session = RuntimeSession.create(camera=camera, settings=settings)
        if record_path is not None:
            from handtracking_trace import TraceRecorder
            session.trace = TraceRecorder(
                path=record_path, settings=settings,
                screen_size=(session.screen_w, session.screen_h),
                started_at=session.start_time, initial_cursor=session.cursor.position(),
                initial_volume=session.volume.level,
                initial_commands_enabled=session.commands_enabled,
                camera_target_fps=session.camera_target_fps,
                max_frames=record_max_frames,
            )
            session.cursor = session.trace.wrap_cursor(session.cursor)
        return _run_impl(session)
    finally:
        if session is not None:
            session.close()
        else:
            camera.close()


if __name__ == "__main__":
    run()
