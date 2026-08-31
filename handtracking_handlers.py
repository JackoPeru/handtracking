"""Focused gesture transition handlers used by the runtime orchestrator."""

from dataclasses import dataclass
import math

from handtracking_config import *
from handtracking_core import pointer_mode_allowed
from handtracking_gestures import (
    is_pointer_pinch_pose,
    is_radial_open_pose,
    normalized_pinch_ratio,
    pointer_other_fingers_valid,
    radial_direction,
)


@dataclass(frozen=True)
class HandlerResult:
    event: str | None
    event_until: float | None
    input_block_until: float


@dataclass(frozen=True)
class PointerResult:
    event: str | None
    event_until: float | None
    precision_snap_active: bool
    snap_anchor: tuple[float, float] | None
    snap_started_at: float | None


def update_pointer_state(
    pointer,
    *,
    control_hand,
    now,
    commands_enabled,
    spock_blocking,
    hand_count,
    paused,
    volume_active,
    two_hand_active,
    two_hand_candidate,
    radial_active,
    scroll_active,
    swipe_tracking,
    input_blocked,
    volume_candidate,
    cursor,
    flow,
    swipe,
    precision_snap_active,
    snap_anchor,
    snap_started_at,
    left_click_cb,
    ratio_fn=normalized_pinch_ratio,
    fingers_valid_fn=pointer_other_fingers_valid,
    pose_fn=is_pointer_pinch_pose,
    allowed_fn=pointer_mode_allowed,
):
    event = None
    event_until = None
    pointer_allowed = allowed_fn(
        commands_enabled=commands_enabled,
        spock_blocking=spock_blocking,
        hand_count=hand_count,
        paused=paused,
        volume_active=volume_active,
        two_hand_active=two_hand_active,
        two_hand_candidate=two_hand_candidate,
        radial_active=radial_active,
        scroll_active=scroll_active,
        swipe_tracking=swipe_tracking,
        input_blocked=input_blocked,
        volume_candidate=volume_candidate,
    )
    pointer_ratio = ratio_fn(control_hand, 8)
    pointer_fingers_valid = fingers_valid_fn(control_hand)
    pointer_pose_on = pose_fn(control_hand, POINTER_PINCH_ON)

    if pointer.pinch_held:
        if not pointer_allowed or not pointer_fingers_valid:
            pointer.reset(preserve_last_click=True)
            cursor.sync(False)
        elif pointer_ratio > POINTER_PINCH_OFF:
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
                    if pointer.cursor_origin is not None:
                        cursor.set_position(
                            pointer.cursor_origin[0], pointer.cursor_origin[1]
                        )
                        cursor.sync(False)
                    is_double_pinch = (
                        pointer.last_click_at is not None and
                        now - pointer.last_click_at <= DOUBLE_PINCH_WINDOW
                    )
                    left_click_cb()
                    if is_double_pinch:
                        event = "DOPPIO PINCH: DOPPIO CLICK"
                        pointer.last_click_at = None
                    else:
                        event = "PINCH RAPIDO: CLICK"
                        pointer.last_click_at = now
                    event_until = now + GESTURE_EVENT_SHOW_SECONDS
                pointer.reset(preserve_last_click=True)
                precision_snap_active = False
                snap_anchor = None
                snap_started_at = None
                cursor.sync(False)
                flow.clear_motion()
        elif pointer_ratio > POINTER_RELEASE_BRAKE_RATIO:
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

    return PointerResult(
        event=event,
        event_until=event_until,
        precision_snap_active=precision_snap_active,
        snap_anchor=snap_anchor,
        snap_started_at=snap_started_at,
    )


def update_two_hand_state(
    two_hand,
    *,
    held,
    pair_geometry,
    now,
    input_block_until,
    radial,
    swipe,
    scroll,
    volume,
    cursor,
    flow,
    ctrl_wheel_cb,
):
    if held:
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
                        two_hand.last_distance = stable_distance
                        two_hand.zoom_residual = 0.0
                    elif abs(distance_delta) >= TWO_HAND_DISTANCE_DEADZONE:
                        two_hand.zoom_residual += distance_delta * TWO_HAND_ZOOM_GAIN
                        two_hand.last_distance = stable_distance
                        zoom_steps = int(two_hand.zoom_residual / TWO_HAND_WHEEL_STEP)
                        if zoom_steps != 0:
                            ctrl_wheel_cb(zoom_steps * int(TWO_HAND_WHEEL_STEP))
                            two_hand.zoom_residual -= zoom_steps * TWO_HAND_WHEEL_STEP
            two_hand.points = (point_a, point_b)
    elif two_hand.active:
        if two_hand.release_at is None:
            two_hand.release_at = now
        elif now - two_hand.release_at >= TWO_HAND_RELEASE_GRACE:
            two_hand.reset()
            input_block_until = now + 0.16
            cursor.sync(True)
            flow.clear_motion()
    else:
        two_hand.candidate_at = None
        two_hand.release_at = None

    return input_block_until


def update_radial_state(
    radial,
    *,
    now,
    priority_block,
    control_hand,
    control_class_hand,
    current_anchor,
    volume_candidate_now,
    volume_candidate_at,
    input_block_until,
    scroll,
    swipe,
    cursor,
    flow,
    direction_fn=radial_direction,
    pinch_ratio_fn=normalized_pinch_ratio,
    open_pose_fn=is_radial_open_pose,
    execute_action_cb,
):
    event = None
    event_until = None

    if priority_block:
        radial.candidate_at = None
        radial.anchor = None
        if radial.active:
            radial.reset()
    elif radial.active:
        raw_selection = direction_fn(control_hand, radial.center)
        if raw_selection != radial.selection_candidate:
            radial.selection_candidate = raw_selection
            radial.selection_since = now
            radial.selected = None
            radial.pinch_candidate_at = None
        elif (radial.selection_since is not None and
              now - radial.selection_since >= RADIAL_SELECTION_HOLD):
            radial.selected = raw_selection

        radial_pinch = pinch_ratio_fn(control_hand, 8)
        if radial_pinch < RADIAL_PINCH_ON:
            if radial.pinch_candidate_at is None:
                radial.pinch_candidate_at = now
            if (not radial.pinch_latched and radial.selected is not None and
                    now - radial.pinch_candidate_at >= RADIAL_PINCH_CONFIRM):
                radial.pinch_latched = True
                action_label = execute_action_cb(radial.selected)
                event = f"RADIAL: {action_label}"
                event_until = now + GESTURE_EVENT_SHOW_SECONDS
                input_block_until = now + 0.20
                radial.reset()
                cursor.sync(True)
                flow.clear_motion()
        elif radial_pinch > RADIAL_PINCH_OFF:
            radial.pinch_latched = False
            radial.pinch_candidate_at = None

        if radial.active:
            open_now = open_pose_fn(control_class_hand)
            if open_now or radial_pinch < RADIAL_PINCH_OFF:
                radial.release_at = None
            else:
                if radial.release_at is None:
                    radial.release_at = now
                elif now - radial.release_at >= RADIAL_RELEASE_GRACE:
                    radial.reset()
                    input_block_until = now + 0.12
                    cursor.sync(True)
                    flow.clear_motion()
    else:
        open_now = open_pose_fn(control_class_hand)
        radial_can_arm = (
            open_now and not scroll.active and not swipe.tracking and
            not volume_candidate_now and volume_candidate_at is None and
            now >= input_block_until
        )
        if radial_can_arm:
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

    return HandlerResult(event, event_until, input_block_until)
