"""Tracking-loss fail-safes shared by the runtime loop."""

from dataclasses import dataclass

from handtracking_config import (
    POINTER_TRACKING_LOSS_GRACE,
    TRACKING_LOSS_GRACE,
    VOLUME_TRACKING_LOSS_GRACE,
)


@dataclass(frozen=True)
class StaleResetResult:
    paused_by_fist: bool = False
    latest_result: object | None = None
    fist_states: list | None = None
    mp_control_ref: tuple[float, float] | None = None
    control_handedness: str | None = None
    debug_fist_score: float = 0.0
    debug_volume_score: float = 0.0
    debug_grip_gap: float = 0.0
    debug_fist_folded: int = 0
    debug_fist_tightness: float = 2.0
    debug_strong_fist: bool = False
    snap_anchor: tuple[float, float] | None = None
    snap_started_at: float | None = None
    precision_snap_active: bool = False

    def __post_init__(self):
        if self.fist_states is None:
            object.__setattr__(self, "fist_states", [])


@dataclass(frozen=True)
class MissingHandsResult:
    paused_by_fist: bool
    gesture_input_block_until: float
    mp_control_ref: tuple[float, float] | None
    control_handedness: str | None
    fist_states: list
    full_reset: bool


def apply_stale_fail_safe(
    *,
    spock,
    fist_vote_history,
    volume,
    scroll,
    two_hand,
    radial,
    swipe,
    pointer,
    flow,
    cursor,
):
    spock.release_required = spock.release_required or spock.latched
    spock.reset(preserve_release_required=True)
    fist_vote_history.clear()
    volume.reset()
    scroll.reset()
    two_hand.reset()
    radial.reset()
    swipe.cancel_tracking()
    pointer.reset(preserve_last_click=True)
    flow.points = None
    flow.active = False
    flow.clear_motion()
    spock.debug_score = 0.0
    spock.debug_stable_score = 0.0
    swipe.debug_score = 0.0
    swipe.debug_stable = 0.0
    swipe.debug_gap = 9.0
    swipe.debug_extended = 0
    cursor.sync(False)
    return StaleResetResult()


def handle_missing_hands(
    *,
    now,
    last_hand_seen,
    pointer,
    volume,
    scroll,
    two_hand,
    radial,
    swipe,
    flow,
    cursor,
    fist_vote_history,
    paused_by_fist,
    gesture_input_block_until,
    mp_control_ref,
    control_handedness,
):
    if (pointer.move_active and
            now - last_hand_seen > POINTER_TRACKING_LOSS_GRACE):
        cursor.sync(False)
        flow.points = None
        flow.active = False
        flow.clear_motion()

    loss_grace = (
        VOLUME_TRACKING_LOSS_GRACE if volume.active else TRACKING_LOSS_GRACE
    )
    if now - last_hand_seen <= loss_grace:
        return MissingHandsResult(
            paused_by_fist=paused_by_fist,
            gesture_input_block_until=gesture_input_block_until,
            mp_control_ref=mp_control_ref,
            control_handedness=control_handedness,
            fist_states=[],
            full_reset=False,
        )

    volume.reset()
    fist_vote_history.clear()
    scroll.reset()
    two_hand.reset()
    radial.reset()
    swipe.cancel_tracking()
    pointer.reset(preserve_last_click=True)
    flow.points = None
    flow.active = False
    flow.clear_motion()
    cursor.sync(False)
    return MissingHandsResult(
        paused_by_fist=False,
        gesture_input_block_until=0.0,
        mp_control_ref=None,
        control_handedness=None,
        fist_states=[],
        full_reset=True,
    )


def expire_lost_flow(flow, *, now):
    if not flow.active and now - flow.last_success > TRACKING_LOSS_GRACE:
        flow.points = None
