"""Coordinate gesture-mode transitions for one MediaPipe hand result."""

import math

from handtracking_config import *
from handtracking_core import normalize_flow_delta
from handtracking_flow import (
    cursor_gain_for_speed,
    flow_points_from_hand,
    propagate_points,
)
from handtracking_gestures import (
    control_point,
    normalized_pinch_ratio,
    two_hand_geometry,
)
from handtracking_handlers import (
    update_pointer_state,
    update_radial_state,
    update_two_hand_state,
)
from handtracking_processing import update_swipe_pose
from handtracking_scroll import update_scroll_state
from handtracking_volume import update_volume_state


def process_two_hand(session, *, hands, volume_candidate_now, now, ctrl_wheel_cb):
    pair_geometry = None
    held = False
    if (len(hands) >= 2 and session.commands_enabled and
            not session.spock.blocking and not session.paused_by_fist and
            not session.volume.active and session.volume.candidate_at is None and
            not volume_candidate_now):
        pair_hands = sorted(hands[:2], key=lambda hand: control_point(hand)[0])
        pair_geometry = two_hand_geometry(pair_hands[0], pair_hands[1])
        pinch_limit = (
            TWO_HAND_PINCH_OFF if session.two_hand.active else TWO_HAND_PINCH_ON
        )
        held = (
            pair_geometry[0] >= TWO_HAND_MIN_SEPARATION and
            all(normalized_pinch_ratio(hand, 8) < pinch_limit for hand in pair_hands)
        )
    session.gesture_input_block_until = update_two_hand_state(
        session.two_hand,
        held=held,
        pair_geometry=pair_geometry,
        now=now,
        input_block_until=session.gesture_input_block_until,
        radial=session.radial,
        swipe=session.swipe,
        scroll=session.scroll,
        volume=session.volume,
        cursor=session.cursor,
        flow=session.flow,
        ctrl_wheel_cb=ctrl_wheel_cb,
    )


def process_radial(
    session,
    *,
    hands,
    control_hand,
    control_class_hand,
    current_anchor,
    volume_candidate_now,
    now,
    execute_action_cb,
):
    priority_block = (
        not session.commands_enabled or session.spock.blocking or
        session.paused_by_fist or session.volume.active or
        session.two_hand.active or session.two_hand.candidate_at is not None or
        len(hands) != 1
    )
    result = update_radial_state(
        session.radial,
        now=now,
        priority_block=priority_block,
        control_hand=control_hand,
        control_class_hand=control_class_hand,
        current_anchor=current_anchor,
        volume_candidate_now=volume_candidate_now,
        volume_candidate_at=session.volume.candidate_at,
        input_block_until=session.gesture_input_block_until,
        scroll=session.scroll,
        swipe=session.swipe,
        cursor=session.cursor,
        flow=session.flow,
        execute_action_cb=execute_action_cb,
    )
    session.gesture_input_block_until = result.input_block_until
    if result.event is not None:
        session.gesture_event = result.event
        session.gesture_event_until = result.event_until


def process_swipe(session, *, hands, control_hand, volume_candidate_now, now):
    allowed = (
        session.commands_enabled and not session.spock.blocking and len(hands) == 1 and
        not session.paused_by_fist and not session.volume.active and
        not volume_candidate_now and session.volume.candidate_at is None and
        not session.two_hand.active and session.two_hand.candidate_at is None and
        not session.radial.active and not session.scroll.active and
        now >= session.gesture_input_block_until
    )
    update_swipe_pose(
        session.swipe,
        allowed=allowed,
        control_hand=control_hand,
        now=now,
    )


def process_pointer(
    session,
    *,
    hands,
    control_hand,
    volume_candidate_now,
    now,
    left_click_cb,
):
    result = update_pointer_state(
        session.pointer,
        control_hand=control_hand,
        now=now,
        commands_enabled=session.commands_enabled,
        spock_blocking=session.spock.blocking,
        hand_count=len(hands),
        paused=session.paused_by_fist,
        volume_active=session.volume.active,
        two_hand_active=session.two_hand.active,
        two_hand_candidate=session.two_hand.candidate_at is not None,
        radial_active=session.radial.active,
        scroll_active=session.scroll.active,
        swipe_tracking=session.swipe.tracking,
        input_blocked=now < session.gesture_input_block_until,
        volume_candidate=(
            volume_candidate_now or session.volume.candidate_at is not None
        ),
        cursor=session.cursor,
        flow=session.flow,
        swipe=session.swipe,
        precision_snap_active=session.precision_snap_active,
        snap_anchor=session.snap_anchor,
        snap_started_at=session.snap_started_at,
        left_click_cb=left_click_cb,
    )
    if result.event is not None:
        session.gesture_event = result.event
        session.gesture_event_until = result.event_until
    session.precision_snap_active = result.precision_snap_active
    session.snap_anchor = result.snap_anchor
    session.snap_started_at = result.snap_started_at


def reanchor_flow(session, control_hand, result_gray, gray):
    points = flow_points_from_hand(control_hand)
    corrected = propagate_points(result_gray, gray, points)
    if corrected is not None:
        session.flow.points = corrected
        session.flow.prev_gray = gray
        session.flow.active = True
    return corrected


def process_volume_scroll(
    session,
    *,
    control_hand,
    control_class_hand,
    fist_pending,
    volume_gesture_now,
    volume_candidate_now,
    scroll_gesture_now,
    now,
    get_volume_cb,
    set_volume_cb,
):
    if session.paused_by_fist or not session.commands_enabled or session.spock.blocking:
        session.volume.reset()
        session.scroll.reset()
        session.two_hand.reset()
        session.radial.reset()
        session.swipe.tracking = False
        return

    dedicated_mode_block = (
        session.pointer.pinch_held or session.two_hand.active or
        session.two_hand.candidate_at is not None or session.radial.active or
        session.swipe.tracking
    )
    update_volume_state(
        session.volume,
        now=now,
        dedicated_mode_block=dedicated_mode_block,
        volume_gesture_now=volume_gesture_now,
        volume_candidate_now=volume_candidate_now,
        control_hand=control_hand,
        control_class_hand=control_class_hand,
        fist_pending=fist_pending,
        debug_volume_score=session.debug_volume_score,
        scroll=session.scroll,
        cursor=session.cursor,
        flow=session.flow,
        get_volume_cb=get_volume_cb,
        set_volume_cb=set_volume_cb,
    )
    update_scroll_state(
        session.scroll,
        now=now,
        gesture_now=scroll_gesture_now,
        blocked=(
            session.pointer.pinch_held or session.volume.active or
            session.two_hand.active or session.radial.active or
            session.swipe.tracking or session.two_hand.candidate_at is not None
        ),
        cursor=session.cursor,
        flow=session.flow,
    )


def sync_cursor_and_fallback(session, *, corrected, old_mp_ref, old_pause, now):
    if session.paused_by_fist or not session.commands_enabled or session.spock.blocking:
        session.cursor.sync(False)
        session.flow.clear_motion()
        return

    exclusive_cursor_block = (
        session.scroll.active or session.volume.active or session.two_hand.active or
        session.radial.active or session.swipe.tracking or
        session.two_hand.candidate_at is not None
    )
    if (session.pointer.move_active and not exclusive_cursor_block and
            now >= session.gesture_input_block_until and
            (old_pause or not session.cursor.active)):
        session.cursor.sync(True)
        session.flow.clear_motion()
    elif not session.pointer.move_active and session.cursor.active:
        session.cursor.sync(False)

    if not (
        session.pointer.pinch_held and session.pointer.move_active and
        not exclusive_cursor_block and corrected is None and
        old_mp_ref is not None and session.mp_control_ref is not None and
        not old_pause and now >= session.gesture_input_block_until
    ):
        return

    fdx = session.mp_control_ref[0] - old_mp_ref[0]
    fdy = session.mp_control_ref[1] - old_mp_ref[1]
    fdx, fdy = normalize_flow_delta(fdx, fdy, session.flow.motion_scale)
    fmag = math.hypot(fdx, fdy)
    session.pointer.flow_travel += fmag * max(DETECTION_W, DETECTION_H)
    if not 0.0005 < fmag < 0.055:
        return

    fallback_dt = (
        max(session.mp_cycle_ms_ema / 1000.0, 1.0 / 60.0)
        if session.mp_cycle_ms_ema > 0.0 else 1.0 / 30.0
    )
    fallback_speed = fmag * max(DETECTION_W, DETECTION_H) / fallback_dt
    dynamic_gain = cursor_gain_for_speed(fallback_speed)
    dx = fdx * session.screen_w * MOVE_GAIN * MOVEMENT_MULTIPLIER * dynamic_gain
    dy = fdy * session.screen_h * MOVE_GAIN * MOVEMENT_MULTIPLIER * dynamic_gain
    screen_step = math.hypot(float(dx), float(dy))
    if session.precision_snap_active:
        if screen_step <= SNAP_BREAK_DELTA_PX:
            dx *= SNAP_HOLD_GAIN
            dy *= SNAP_HOLD_GAIN
        else:
            session.precision_snap_active = False
            session.snap_anchor = None
            session.snap_started_at = None
    session.cursor.add_delta(
        float(dx),
        float(dy),
        screen_size=(session.screen_w, session.screen_h),
    )
