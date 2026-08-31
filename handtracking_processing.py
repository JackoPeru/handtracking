"""Semi-pure MediaPipe result processing helpers."""

from dataclasses import dataclass
import math

from handtracking_config import *
from handtracking_core import (
    advance_confirmed_hold,
    choose_control_index,
    fist_evidence_from_hands,
    normalized_points_pixel_distance,
    palm_motion_scale,
    spock_release_gate_active,
)
from handtracking_gestures import (
    clamp,
    control_point,
    fist_fold_metrics,
    grip_class_scores,
    is_fist,
    is_scroll_gesture,
    is_strong_fist,
    swipe_pose_metrics,
)


@dataclass(frozen=True)
class SpockUpdate:
    commands_enabled: bool
    toggled: bool
    released: bool
    event: str | None
    event_until: float | None
    input_block_until: float


@dataclass(frozen=True)
class SnapUpdate:
    active: bool
    anchor: tuple[float, float] | None
    started_at: float | None


@dataclass(frozen=True)
class HandFrameAnalysis:
    control_index: int
    control_distance: float
    selected_handedness: str
    control_hand: object
    control_class_hand: object
    points: list
    fist_scores: list
    volume_scores: list
    gap_scores: list
    raw_fist_states: list
    strong_fist_states: list
    paused_by_fist: bool
    old_pause: bool
    fist_pending: bool


@dataclass(frozen=True)
class HandModeMetrics:
    volume_gesture_now: bool
    volume_candidate_now: bool
    fist_states: list
    scroll_gesture_now: bool
    debug_fist_score: float
    debug_volume_score: float
    debug_grip_gap: float
    debug_fist_folded: int
    debug_fist_tightness: float
    debug_strong_fist: bool


def update_ema_metrics(current, sample):
    infer, worker, cycle, queue = current
    sample_infer, sample_worker, sample_cycle, sample_queue = sample
    if infer <= 0.0:
        return sample_infer, sample_worker, sample_cycle, sample_queue

    infer = infer * 0.85 + sample_infer * 0.15
    worker = worker * 0.85 + sample_worker * 0.15
    if sample_cycle > 0.0:
        if cycle <= 0.0:
            cycle = sample_cycle
        else:
            cycle = cycle * 0.85 + sample_cycle * 0.15
    queue = queue * 0.85 + sample_queue * 0.15
    return infer, worker, cycle, queue


def analyze_hand_frame(
    *,
    latest_result,
    hands,
    class_hands,
    previous_point,
    previous_label,
    paused_by_fist,
    fist_vote_history,
    volume_active,
    grip_fn=grip_class_scores,
    point_fn=control_point,
    choose_fn=choose_control_index,
    fist_fn=is_fist,
    strong_fist_fn=is_strong_fist,
):
    class_metrics = [grip_fn(hand) for hand in class_hands]
    norm_metrics = [grip_fn(hand) for hand in hands]
    fist_scores = [m[0] for m in class_metrics]
    volume_scores = [
        max(class_metrics[i][1], norm_metrics[i][1])
        for i in range(len(hands))
    ]
    gap_scores = [norm_metrics[i][2] for i in range(len(hands))]

    points = [point_fn(hand) for hand in hands]
    handedness_result = getattr(latest_result, "handedness", None) or []
    handedness_labels = []
    for i in range(len(hands)):
        try:
            handedness_labels.append(handedness_result[i][0].category_name or "")
        except (IndexError, AttributeError, TypeError):
            handedness_labels.append("")

    control_index = choose_fn(
        points,
        handedness_labels,
        previous_point=previous_point,
        previous_label=previous_label,
    )
    control_distance = (
        0.0 if previous_point is None else
        math.hypot(
            points[control_index][0] - previous_point[0],
            points[control_index][1] - previous_point[1],
        )
    )
    selected_handedness = handedness_labels[control_index]

    raw_fist_states = [fist_fn(hand) for hand in hands]
    strong_fist_states = [strong_fist_fn(hand) for hand in hands]
    fist_evidence = fist_evidence_from_hands(
        raw_fists=raw_fist_states,
        strong_fists=strong_fist_states,
        volume_scores=volume_scores,
        gap_scores=gap_scores,
        volume_active=volume_active,
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
    elif sum(v < 0.5 for v in fist_vote_history) >= FIST_VOTE_OFF:
        paused_by_fist = False

    return HandFrameAnalysis(
        control_index=control_index,
        control_distance=control_distance,
        selected_handedness=selected_handedness,
        control_hand=hands[control_index],
        control_class_hand=class_hands[control_index],
        points=points,
        fist_scores=fist_scores,
        volume_scores=volume_scores,
        gap_scores=gap_scores,
        raw_fist_states=raw_fist_states,
        strong_fist_states=strong_fist_states,
        paused_by_fist=paused_by_fist,
        old_pause=old_pause,
        fist_pending=fist_pending,
    )


def update_hand_mode_metrics(
    analysis,
    *,
    hands,
    volume,
    flow,
    fold_fn=fist_fold_metrics,
    scroll_fn=is_scroll_gesture,
    pixel_distance_fn=normalized_points_pixel_distance,
    palm_scale_fn=palm_motion_scale,
):
    control_hand = analysis.control_hand
    palm_width_px = pixel_distance_fn(
        (control_hand[5].x, control_hand[5].y),
        (control_hand[17].x, control_hand[17].y),
        DETECTION_W,
        DETECTION_H,
    )
    target_motion_scale = palm_scale_fn(
        palm_width_px,
        reference_width_px=PALM_REFERENCE_WIDTH_PX,
        minimum=PALM_SCALE_MIN,
        maximum=PALM_SCALE_MAX,
    )
    flow.motion_scale = flow.motion_scale * 0.82 + target_motion_scale * 0.18

    debug_fist_score = max(analysis.fist_scores, default=0.0)
    debug_volume_score = analysis.volume_scores[analysis.control_index]
    debug_grip_gap = analysis.gap_scores[analysis.control_index]
    debug_fist_folded, debug_fist_tightness = fold_fn(control_hand)
    debug_strong_fist = analysis.strong_fist_states[analysis.control_index]

    if analysis.fist_pending:
        volume.vote_history.clear()
        volume.last_angle = None
        volume.delta_history.clear()
    else:
        volume.vote_history.append(debug_volume_score)

    volume_gesture_now = (
        sum(s >= VOLUME_SCORE_ON for s in volume.vote_history) >= VOLUME_VOTE_ON
        and not analysis.paused_by_fist
        and not analysis.fist_pending
    )
    volume_candidate_now = (
        debug_volume_score >= VOLUME_SCORE_CANDIDATE
        and not analysis.paused_by_fist
        and not analysis.fist_pending
    )
    fist_states = [
        (analysis.paused_by_fist and analysis.raw_fist_states[i])
        for i in range(len(hands))
    ]
    scroll_states = [
        scroll_fn(hand)
        and not analysis.paused_by_fist
        and not volume.active
        and not volume_candidate_now
        for hand in hands
    ]
    scroll_gesture_now = (
        scroll_states[analysis.control_index] and not volume_gesture_now
    )

    return HandModeMetrics(
        volume_gesture_now=volume_gesture_now,
        volume_candidate_now=volume_candidate_now,
        fist_states=fist_states,
        scroll_gesture_now=scroll_gesture_now,
        debug_fist_score=debug_fist_score,
        debug_volume_score=debug_volume_score,
        debug_grip_gap=debug_grip_gap,
        debug_fist_folded=debug_fist_folded,
        debug_fist_tightness=debug_fist_tightness,
        debug_strong_fist=debug_strong_fist,
    )


def update_swipe_pose(swipe, *, allowed, control_hand, now, score_fn=swipe_pose_metrics):
    if allowed:
        (swipe.debug_score, swipe.debug_gap,
         swipe.debug_extended) = score_fn(control_hand)
        swipe.pose_history.append(swipe.debug_score)
        best_scores = sorted(swipe.pose_history, reverse=True)[:SWIPE_POSE_KEEP_BEST]
        swipe.debug_stable = sum(best_scores) / max(len(best_scores), 1)
        if (swipe.debug_score >= SWIPE_POSE_SCORE_ON or
                swipe.debug_stable >= SWIPE_POSE_SCORE_ON):
            swipe.pose_last_seen = now
        elif swipe.tracking and swipe.debug_score >= SWIPE_POSE_SCORE_HOLD:
            swipe.pose_last_seen = now
    else:
        swipe.debug_score = 0.0
        swipe.debug_stable = 0.0
        swipe.debug_gap = 9.0
        swipe.debug_extended = 0
        swipe.pose_history.clear()
        swipe.pose_last_seen = None


def update_precision_snap(*, allowed, cursor_position, now, active, anchor, started_at):
    if not allowed:
        return SnapUpdate(False, None, None)

    cursor_x, cursor_y = cursor_position
    if anchor is None or started_at is None:
        return SnapUpdate(False, (float(cursor_x), float(cursor_y)), now)

    snap_drift = math.hypot(cursor_x - anchor[0], cursor_y - anchor[1])
    allowed_radius = SNAP_HOLD_RADIUS_PX if active else SNAP_RADIUS_PX
    if snap_drift <= allowed_radius:
        stable_for = now - started_at
        if stable_for >= SNAP_ARM_SECONDS:
            active = True
        return SnapUpdate(active, anchor, started_at)

    return SnapUpdate(False, (float(cursor_x), float(cursor_y)), now)


def update_spock_state(spock, *, raw_score, upright_now, now, sample_seconds,
                       commands_enabled, input_block_until):
    spock.debug_score = raw_score
    if upright_now:
        spock.upright_invalid_frames = 0
        spock.score_history.append(spock.debug_score)
    else:
        spock.upright_invalid_frames += 1
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
    toggled = False
    released = False
    event = None
    event_until = None

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
                input_block_until = max(
                    input_block_until,
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
                sample_seconds,
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
                toggled = True
                event = (
                    "CONTROLLI ATTIVI" if commands_enabled else "CONTROLLI BLOCCATI"
                )
                event_until = now + 1.20
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
                input_block_until = max(
                    input_block_until,
                    now + SPOCK_POST_RELEASE_BLOCK,
                )
                released = True
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

    return SpockUpdate(
        commands_enabled=commands_enabled,
        toggled=toggled,
        released=released,
        event=event,
        event_until=event_until,
        input_block_until=input_block_until,
    )
