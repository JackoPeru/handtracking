"""Volume gesture state machine."""

from handtracking_config import *
from handtracking_gestures import (
    clamp,
    is_open_hand,
    is_volume_release_pose,
    palm_roll_angle,
    wrapped_angle_delta,
)


def update_volume_state(
    volume,
    *,
    now,
    dedicated_mode_block,
    volume_gesture_now,
    volume_candidate_now,
    control_hand,
    control_class_hand,
    fist_pending,
    debug_volume_score,
    scroll,
    cursor,
    flow,
    get_volume_cb,
    set_volume_cb,
    angle_fn=palm_roll_angle,
    release_pose_fn=is_volume_release_pose,
    open_hand_fn=is_open_hand,
    angle_delta_fn=wrapped_angle_delta,
):
    if not volume.active:
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
                volume.last_angle = angle_fn(control_hand)
                volume.delta_history.clear()
                volume.level = get_volume_cb()
                scroll.reset()
                cursor.sync(False)
                flow.clear_motion()
        elif volume.candidate_at is not None:
            if (volume.candidate_last_seen is None or
                    now - volume.candidate_last_seen > VOLUME_ENTRY_MISS_GRACE):
                volume.candidate_at = None
                volume.candidate_last_seen = None
        return

    release_pose = release_pose_fn(control_class_hand)
    fully_open = open_hand_fn(control_class_hand)
    if fist_pending:
        volume.pose_lost_at = None
        volume.release_at = None
        volume.last_angle = None
        volume.delta_history.clear()
    elif release_pose or fully_open:
        volume.pose_lost_at = None
        if volume.release_at is None:
            volume.release_at = now
        volume.last_angle = None
        volume.delta_history.clear()
        if now - volume.release_at >= VOLUME_RELEASE_GRACE:
            volume.reset()
            cursor.sync(True)
            flow.clear_motion()
    elif debug_volume_score < VOLUME_HOLD_MIN_SCORE:
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
        angle = angle_fn(control_hand)
        if volume.last_angle is None:
            volume.last_angle = angle
            volume.delta_history.clear()
        else:
            raw_delta = angle_delta_fn(angle, volume.last_angle)
            volume.last_angle = angle
            raw_delta = clamp(
                raw_delta,
                -VOLUME_MAX_DELTA_RAD,
                VOLUME_MAX_DELTA_RAD,
            )
            volume.delta_history.append(raw_delta)
            stable_delta = sorted(volume.delta_history)[
                len(volume.delta_history) // 2
            ]
            if abs(stable_delta) >= VOLUME_DEADZONE_RAD:
                volume.level = clamp(
                    volume.level + stable_delta * VOLUME_GAIN * VOLUME_DIRECTION,
                    0.0,
                    1.0,
                )
                set_volume_cb(volume.level)
