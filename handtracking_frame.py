"""Process one newly-published MediaPipe result against RuntimeSession state."""

from dataclasses import dataclass

from handtracking_config import *
from handtracking_processing import (
    analyze_hand_frame,
    update_hand_mode_metrics,
    update_ema_metrics,
)
from handtracking_modes import (
    process_pointer,
    process_radial,
    process_swipe,
    process_two_hand,
    process_volume_scroll,
    reanchor_flow,
    sync_cursor_and_fallback,
)
from handtracking_spock import (
    update_spock_state,
    update_spock_without_hands,
)
from handtracking_tracking import handle_missing_hands
from handtracking_gestures import (
    spock_all_fingers_up,
    spock_pose_score,
)
from handtracking_windows import (
    ctrl_wheel,
    execute_radial_action,
    get_system_volume,
    left_click,
    set_system_volume,
)


@dataclass(frozen=True)
class FrameProcessResult:
    processed: bool
    skip_frame: bool = False


def process_mediapipe_packet(
    session,
    packet,
    *,
    gray,
    now,
    camera_target_fps,
    ctrl_wheel_cb=ctrl_wheel,
    execute_radial_action_cb=execute_radial_action,
    left_click_cb=left_click,
    get_volume_cb=get_system_volume,
    set_volume_cb=set_system_volume,
):
    if packet is None or packet[0] == session.latest_result_seq:
        return FrameProcessResult(False)

    (session.latest_result_seq, session.latest_result, result_gray,
     mp_infer_ms, mp_worker_ms, mp_cycle_ms, mp_queue_ms) = packet
    (session.mp_infer_ms_ema, session.mp_worker_ms_ema,
     session.mp_cycle_ms_ema, session.mp_queue_ms_ema) = update_ema_metrics(
        (
            session.mp_infer_ms_ema,
            session.mp_worker_ms_ema,
            session.mp_cycle_ms_ema,
            session.mp_queue_ms_ema,
        ),
        (mp_infer_ms, mp_worker_ms, mp_cycle_ms, mp_queue_ms),
    )

    if not session.latest_result.hand_landmarks:
        _process_missing_hands(session, now=now)
        return FrameProcessResult(True)

    hands = session.latest_result.hand_landmarks
    world_hands = getattr(session.latest_result, "hand_world_landmarks", None)
    class_hands = (
        hands
        if not world_hands or len(world_hands) != len(hands)
        else world_hands
    )

    _process_spock(
        session,
        hands=hands,
        now=now,
        mp_cycle_ms=mp_cycle_ms,
        camera_target_fps=camera_target_fps,
    )

    old_mp_ref = session.mp_control_ref
    analysis = analyze_hand_frame(
        latest_result=session.latest_result,
        hands=hands,
        class_hands=class_hands,
        previous_point=session.mp_control_ref,
        previous_label=session.control_handedness,
        paused_by_fist=session.paused_by_fist,
        fist_vote_history=session.fist_vote_history,
        volume_active=session.volume.active,
    )
    session.paused_by_fist = analysis.paused_by_fist
    old_pause = analysis.old_pause
    fist_pending = analysis.fist_pending
    session.control_index = analysis.control_index

    if (session.volume.active and session.mp_control_ref is not None and
            analysis.control_distance > VOLUME_HAND_SWITCH_MAX):
        if now - session.last_hand_seen > VOLUME_TRACKING_LOSS_GRACE:
            session.volume.reset()
            session.mp_control_ref = None
            session.control_handedness = None
            session.cursor.sync(False)
        return FrameProcessResult(True, skip_frame=True)

    session.last_hand_seen = now
    if analysis.selected_handedness:
        session.control_handedness = analysis.selected_handedness
    control_hand = analysis.control_hand
    control_class_hand = analysis.control_class_hand
    points = analysis.points
    mode_metrics = update_hand_mode_metrics(
        analysis,
        hands=hands,
        volume=session.volume,
        flow=session.flow,
    )
    session.fist_states = mode_metrics.fist_states
    session.debug_fist_score = mode_metrics.debug_fist_score
    session.debug_volume_score = mode_metrics.debug_volume_score
    session.debug_grip_gap = mode_metrics.debug_grip_gap
    session.debug_fist_folded = mode_metrics.debug_fist_folded
    session.debug_fist_tightness = mode_metrics.debug_fist_tightness
    session.debug_strong_fist = mode_metrics.debug_strong_fist
    session.mp_control_ref = points[session.control_index]

    process_two_hand(
        session,
        hands=hands,
        volume_candidate_now=mode_metrics.volume_candidate_now,
        now=now,
        ctrl_wheel_cb=ctrl_wheel_cb,
    )
    process_radial(
        session,
        hands=hands,
        control_hand=control_hand,
        control_class_hand=control_class_hand,
        current_anchor=points[session.control_index],
        volume_candidate_now=mode_metrics.volume_candidate_now,
        now=now,
        execute_action_cb=execute_radial_action_cb,
    )
    process_swipe(
        session,
        hands=hands,
        control_hand=control_hand,
        volume_candidate_now=mode_metrics.volume_candidate_now,
        now=now,
    )
    process_pointer(
        session,
        hands=hands,
        control_hand=control_hand,
        volume_candidate_now=mode_metrics.volume_candidate_now,
        now=now,
        left_click_cb=left_click_cb,
    )

    corrected = reanchor_flow(session, control_hand, result_gray, gray)
    process_volume_scroll(
        session,
        control_hand=control_hand,
        control_class_hand=control_class_hand,
        fist_pending=fist_pending,
        volume_gesture_now=mode_metrics.volume_gesture_now,
        volume_candidate_now=mode_metrics.volume_candidate_now,
        scroll_gesture_now=mode_metrics.scroll_gesture_now,
        now=now,
        get_volume_cb=get_volume_cb,
        set_volume_cb=set_volume_cb,
    )
    sync_cursor_and_fallback(
        session,
        corrected=corrected,
        old_mp_ref=old_mp_ref,
        old_pause=old_pause,
        now=now,
    )
    return FrameProcessResult(True)


def _process_spock(session, *, hands, now, mp_cycle_ms, camera_target_fps):
    spock_scores_now = [spock_pose_score(hand) for hand in hands]
    upright_now = any(spock_all_fingers_up(hand) for hand in hands)
    sample_seconds = min(
        SPOCK_SAMPLE_MAX_SECONDS,
        max(
            (mp_cycle_ms / 1000.0) if mp_cycle_ms > 0.0
            else 1.0 / max(camera_target_fps, 1),
            1.0 / 120.0,
        ),
    )
    update = update_spock_state(
        session.spock,
        raw_score=max(spock_scores_now, default=0.0),
        upright_now=upright_now,
        now=now,
        sample_seconds=sample_seconds,
        commands_enabled=session.commands_enabled,
        input_block_until=session.gesture_input_block_until,
    )
    session.commands_enabled = update.commands_enabled
    session.gesture_input_block_until = update.input_block_until
    if update.event is not None:
        session.gesture_event = update.event
        session.gesture_event_until = update.event_until
    if update.released:
        session.mp_control_ref = None
        session.control_handedness = None
        session.flow.points = None
        session.flow.active = False
        session.cursor.sync(False)
    if update.toggled:
        session.paused_by_fist = False
        session.fist_vote_history.clear()
        session.volume.reset()
        session.scroll.reset()
        session.two_hand.reset()
        session.radial.reset()
        session.swipe.tracking = False
        session.mp_control_ref = None
        session.control_handedness = None
        session.flow.points = None
        session.flow.active = False
        session.flow.clear_motion()
        session.cursor.sync(False)


def _process_missing_hands(session, *, now):
    session.debug_fist_score = 0.0
    session.debug_volume_score = 0.0
    session.debug_grip_gap = 0.0
    session.debug_fist_folded = 0
    session.debug_fist_tightness = 2.0
    session.debug_strong_fist = False
    session.swipe.debug_score = 0.0
    session.swipe.debug_stable = 0.0
    session.swipe.debug_gap = 9.0
    session.swipe.debug_extended = 0
    session.gesture_input_block_until = update_spock_without_hands(
        session.spock,
        now=now,
        input_block_until=session.gesture_input_block_until,
    )
    missing = handle_missing_hands(
        now=now,
        last_hand_seen=session.last_hand_seen,
        pointer=session.pointer,
        volume=session.volume,
        scroll=session.scroll,
        two_hand=session.two_hand,
        radial=session.radial,
        swipe=session.swipe,
        flow=session.flow,
        cursor=session.cursor,
        fist_vote_history=session.fist_vote_history,
        paused_by_fist=session.paused_by_fist,
        gesture_input_block_until=session.gesture_input_block_until,
        mp_control_ref=session.mp_control_ref,
        control_handedness=session.control_handedness,
    )
    session.paused_by_fist = missing.paused_by_fist
    session.gesture_input_block_until = missing.gesture_input_block_until
    session.mp_control_ref = missing.mp_control_ref
    session.control_handedness = missing.control_handedness
    session.fist_states = missing.fist_states
