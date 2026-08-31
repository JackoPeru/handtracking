"""Semi-pure MediaPipe result processing helpers."""

from dataclasses import dataclass
import math

from handtracking_config import *
from handtracking_core import advance_confirmed_hold, spock_release_gate_active
from handtracking_gestures import clamp, swipe_pose_metrics


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
