"""Mutable runtime state grouped by gesture responsibility."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from handtracking_config import (
    SPOCK_SCORE_WINDOW,
    SWIPE_POSE_HISTORY,
    TWO_HAND_DISTANCE_HISTORY,
    VOLUME_DELTA_MEDIAN_FRAMES,
    VOLUME_VOTE_WINDOW,
)


@dataclass
class PointerState:
    pinch_held: bool = False
    move_active: bool = False
    pinch_started_at: float | None = None
    release_at: float | None = None
    release_braking: bool = False
    motion_accum: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    flow_travel: float = 0.0
    cursor_origin: tuple[float, float] | None = None
    last_click_at: float | None = None

    def reset(self, *, preserve_last_click=False):
        last_click = self.last_click_at if preserve_last_click else None
        self.pinch_held = False
        self.move_active = False
        self.pinch_started_at = None
        self.release_at = None
        self.release_braking = False
        self.motion_accum[:] = 0.0
        self.flow_travel = 0.0
        self.cursor_origin = None
        self.last_click_at = last_click


@dataclass
class ScrollState:
    active: bool = False
    candidate_at: float | None = None
    release_at: float | None = None
    residual: float = 0.0

    def reset(self):
        self.active = False
        self.candidate_at = None
        self.release_at = None
        self.residual = 0.0


@dataclass
class VolumeState:
    active: bool = False
    candidate_at: float | None = None
    candidate_last_seen: float | None = None
    release_at: float | None = None
    pose_lost_at: float | None = None
    last_angle: float | None = None
    level: float = 0.5
    delta_history: deque = field(
        default_factory=lambda: deque(maxlen=VOLUME_DELTA_MEDIAN_FRAMES)
    )
    vote_history: deque = field(
        default_factory=lambda: deque(maxlen=VOLUME_VOTE_WINDOW)
    )

    def reset(self, *, preserve_level=True):
        level = self.level if preserve_level else 0.5
        self.active = False
        self.candidate_at = None
        self.candidate_last_seen = None
        self.release_at = None
        self.pose_lost_at = None
        self.last_angle = None
        self.level = level
        self.delta_history.clear()
        self.vote_history.clear()


@dataclass
class SwipeState:
    tracking: bool = False
    cooldown_until: float = 0.0
    pose_history: deque = field(
        default_factory=lambda: deque(maxlen=SWIPE_POSE_HISTORY)
    )
    pose_last_seen: float | None = None
    flow_started_at: float | None = None
    flow_accum_x: float = 0.0
    flow_accum_y: float = 0.0
    debug_score: float = 0.0
    debug_stable: float = 0.0
    debug_gap: float = 9.0
    debug_extended: int = 0

    def reset(self, *, preserve_cooldown=False):
        cooldown = self.cooldown_until if preserve_cooldown else 0.0
        self.tracking = False
        self.cooldown_until = cooldown
        self.pose_history.clear()
        self.pose_last_seen = None
        self.flow_started_at = None
        self.flow_accum_x = 0.0
        self.flow_accum_y = 0.0
        self.debug_score = 0.0
        self.debug_stable = 0.0
        self.debug_gap = 9.0
        self.debug_extended = 0


@dataclass
class RadialState:
    candidate_at: float | None = None
    anchor: tuple[float, float] | None = None
    active: bool = False
    center: tuple[float, float] | None = None
    selected: str | None = None
    selection_candidate: str | None = None
    selection_since: float | None = None
    release_at: float | None = None
    pinch_latched: bool = False
    pinch_candidate_at: float | None = None

    def reset(self):
        self.candidate_at = None
        self.anchor = None
        self.active = False
        self.center = None
        self.selected = None
        self.selection_candidate = None
        self.selection_since = None
        self.release_at = None
        self.pinch_latched = False
        self.pinch_candidate_at = None


@dataclass
class TwoHandState:
    candidate_at: float | None = None
    active: bool = False
    release_at: float | None = None
    last_distance: float | None = None
    distance_history: deque = field(
        default_factory=lambda: deque(maxlen=TWO_HAND_DISTANCE_HISTORY)
    )
    zoom_residual: float = 0.0
    points: tuple[tuple[float, float], tuple[float, float]] | None = None

    def reset(self):
        self.candidate_at = None
        self.active = False
        self.release_at = None
        self.last_distance = None
        self.distance_history.clear()
        self.zoom_residual = 0.0
        self.points = None


@dataclass
class SpockState:
    candidate_at: float | None = None
    last_seen: float | None = None
    release_at: float | None = None
    latched: bool = False
    release_required: bool = False
    blocking: bool = False
    progress: float = 0.0
    confirmed_seconds: float = 0.0
    debug_score: float = 0.0
    debug_stable_score: float = 0.0
    score_history: deque = field(
        default_factory=lambda: deque(maxlen=SPOCK_SCORE_WINDOW)
    )
    upright_invalid_frames: int = 0

    def reset(self, *, preserve_release_required=False):
        release_required = self.release_required if preserve_release_required else False
        self.candidate_at = None
        self.last_seen = None
        self.release_at = None
        self.latched = False
        self.release_required = release_required
        self.blocking = release_required
        self.progress = 0.0
        self.confirmed_seconds = 0.0
        self.debug_score = 0.0
        self.debug_stable_score = 0.0
        self.score_history.clear()
        self.upright_invalid_frames = 0


@dataclass
class FlowState:
    prev_gray: object | None = None
    points: object | None = None
    virtual: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    filtered: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    prev_filtered: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    time: float | None = None
    active: bool = False
    motion_scale: float = 1.0
    last_success: float = 0.0

    def reset(self, *, preserve_prev_gray=False, preserve_motion_scale=False):
        prev_gray = self.prev_gray if preserve_prev_gray else None
        motion_scale = self.motion_scale if preserve_motion_scale else 1.0
        self.prev_gray = prev_gray
        self.points = None
        self.virtual[:] = 0.0
        self.filtered[:] = 0.0
        self.prev_filtered[:] = 0.0
        self.time = None
        self.active = False
        self.motion_scale = motion_scale
        self.last_success = 0.0
