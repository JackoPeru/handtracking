"""Optical-flow measurement and camera-rate gesture dispatch."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from handtracking_config import *
from handtracking_core import normalize_flow_delta
from handtracking_gestures import clamp


@dataclass(frozen=True)
class LKMotion:
    next_points: np.ndarray
    dx: float
    dy: float
    magnitude: float


@dataclass(frozen=True)
class FlowDispatchResult:
    gesture_event: str | None
    gesture_event_until: float | None
    gesture_input_block_until: float
    precision_snap_active: bool
    snap_anchor: tuple[float, float] | None
    snap_started_at: float | None


def smoothstep01(value):
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def cursor_gain_for_speed(speed_px_s):
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


def flow_points_from_hand(hand):
    pts = [
        [hand[idx].x * DETECTION_W, hand[idx].y * DETECTION_H]
        for idx in FLOW_LANDMARK_IDS
    ]
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


def summarize_lk_tracks(old_points, next_points, status, err,
                        back_points, back_status, motion_scale):
    if next_points is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if back_points is not None and back_status is not None:
        good &= back_status.reshape(-1).astype(bool)
        fb_error = np.linalg.norm(
            back_points.reshape(-1, 2) - old_points.reshape(-1, 2), axis=1
        )
        good &= fb_error <= FLOW_FB_MAX
    if good.sum() < 3:
        return None

    old_good = old_points.reshape(-1, 2)[good]
    new_good = next_points.reshape(-1, 2)[good]
    deltas = new_good - old_good
    mdx, mdy = np.median(deltas, axis=0)
    spread = np.median(np.linalg.norm(deltas - [mdx, mdy], axis=1))
    med_err = 0.0 if err is None else float(np.median(err.reshape(-1)[good]))
    raw_mag = math.hypot(float(mdx), float(mdy))
    if not (
        spread <= FLOW_MAX_SPREAD and
        med_err <= FLOW_MAX_ERROR and
        raw_mag <= FLOW_MAX_CAMERA_STEP
    ):
        return None

    dx, dy = normalize_flow_delta(float(mdx), float(mdy), motion_scale)
    return LKMotion(
        next_points=next_points,
        dx=dx,
        dy=dy,
        magnitude=math.hypot(dx, dy),
    )


def measure_optical_flow(prev_gray, gray, points, motion_scale):
    if prev_gray is None or points is None:
        return None
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, None,
        winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if next_pts is None or status is None:
        return None
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, next_pts, None,
        winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    return summarize_lk_tracks(
        points, next_pts, status, err, back_pts, back_status, motion_scale
    )


def dispatch_flow_motion(*, motion_dx, motion_dy, motion_mag, now,
                         mp_result_stale, paused_by_fist, commands_enabled,
                         spock_blocking, gesture_input_block_until,
                         pointer, volume, two_hand, radial, scroll, swipe, flow,
                         cursor, screen_w, screen_h, precision_snap_active,
                         snap_anchor, snap_started_at, execute_swipe_cb,
                         mouse_wheel_cb):
    gesture_event = None
    gesture_event_until = None

    if (mp_result_stale or paused_by_fist or not commands_enabled or
            spock_blocking or now < gesture_input_block_until):
        return FlowDispatchResult(
            gesture_event, gesture_event_until, gesture_input_block_until,
            precision_snap_active, snap_anchor, snap_started_at,
        )

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
        horizontal_flow = (
            abs(motion_dx) >= SWIPE_FLOW_INTENT_PX and
            abs(motion_dx) >= abs(motion_dy) * SWIPE_FLOW_AXIS_DOMINANCE
        )
        if not swipe.tracking and swipe_pose_recent and horizontal_flow:
            swipe.tracking = True
            swipe.flow_started_at = now
            swipe.flow_accum_x = 0.0
            swipe.flow_accum_y = 0.0
            cursor.sync(False)
            flow.clear_motion()

        if swipe.tracking:
            swipe_motion_consumed = True
            swipe.flow_accum_x += motion_dx
            swipe.flow_accum_y += motion_dy
            swipe_elapsed = now - (swipe.flow_started_at or now)
            total_horizontal = (
                abs(swipe.flow_accum_x) >=
                abs(swipe.flow_accum_y) * SWIPE_FLOW_AXIS_DOMINANCE
            )
            if (swipe_elapsed <= SWIPE_FLOW_MAX_SECONDS and total_horizontal and
                    abs(swipe.flow_accum_x) >= SWIPE_FLOW_TRIGGER_PX):
                direction = "RIGHT" if swipe.flow_accum_x > 0 else "LEFT"
                action_label = execute_swipe_cb(direction)
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
        swipe_motion_consumed = True
        swipe.tracking = False
        swipe.flow_started_at = None
        swipe.flow_accum_x = 0.0
        swipe.flow_accum_y = 0.0
        cursor.sync(True)
        flow.clear_motion()

    if swipe_motion_consumed:
        pass
    elif volume.active or two_hand.active or radial.active or two_hand.candidate_at is not None:
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
                mouse_wheel_cb(wheel_delta)
                scroll.residual -= wheel_delta
    elif pointer.pinch_held and not pointer.release_braking:
        pointer.motion_accum += np.array([motion_dx, motion_dy], dtype=np.float64)
        pointer.flow_travel = max(
            pointer.flow_travel,
            float(np.linalg.norm(pointer.motion_accum)),
        )
        if (not pointer.move_active and
                pointer.flow_travel >= POINTER_MOVE_TRIGGER_PX):
            pointer.move_active = True
            cursor.sync(True)
            flow.clear_motion()

        if pointer.move_active:
            flow.virtual += np.array([motion_dx, motion_dy], dtype=np.float64)
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
                    cursor.add_delta(
                        float(dx), float(dy), screen_size=(screen_w, screen_h)
                    )
    else:
        if cursor.active:
            cursor.sync(False)
        flow.clear_motion()

    return FlowDispatchResult(
        gesture_event,
        gesture_event_until,
        gesture_input_block_until,
        precision_snap_active,
        snap_anchor,
        snap_started_at,
    )
