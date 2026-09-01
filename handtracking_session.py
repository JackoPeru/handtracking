"""Runtime resource ownership and persistent session state."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time

import cv2
import mediapipe as mp

from handtracking_config import FIST_VOTE_WINDOW
from handtracking_config import HUD_REFRESH_HZ
from handtracking_display import OverlayLayer
from handtracking_mediapipe import MediaPipeWorker
from handtracking_perf import MediaPipeSubmitScheduler, PerfProfiler
from handtracking_state import (
    FlowState,
    PointerState,
    RadialState,
    ScrollState,
    SpockState,
    SwipeState,
    TwoHandState,
    VolumeState,
)
from handtracking_windows import CursorController, get_system_volume


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

DEFAULT_OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


def build_mediapipe_image(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _close_resources(worker, cursor, camera):
    if worker is not None:
        try:
            worker.stop()
        except Exception:
            pass
        try:
            worker.join(timeout=2.0)
        except Exception:
            pass
    if cursor is not None:
        try:
            cursor.close()
        except Exception:
            pass
    if camera is not None:
        try:
            camera.close()
        except Exception:
            pass


@dataclass(slots=True)
class RuntimeSession:
    camera: object
    worker: object
    cursor: object
    screen_w: int
    screen_h: int
    start_time: float

    flow: FlowState = field(default_factory=FlowState)
    pointer: PointerState = field(default_factory=PointerState)
    scroll: ScrollState = field(default_factory=ScrollState)
    volume: VolumeState = field(default_factory=VolumeState)
    swipe: SwipeState = field(default_factory=SwipeState)
    radial: RadialState = field(default_factory=RadialState)
    two_hand: TwoHandState = field(default_factory=TwoHandState)
    spock: SpockState = field(default_factory=SpockState)

    latest_result: object | None = None
    latest_result_seq: int = -1
    control_index: int = 0
    control_handedness: str | None = None
    mp_control_ref: tuple[float, float] | None = None
    fist_states: list = field(default_factory=list)
    paused_by_fist: bool = False
    fist_vote_history: deque = field(
        default_factory=lambda: deque(maxlen=FIST_VOTE_WINDOW)
    )

    debug_fist_score: float = 0.0
    debug_volume_score: float = 0.0
    debug_grip_gap: float = 0.0
    debug_fist_folded: int = 0
    debug_fist_tightness: float = 2.0
    debug_strong_fist: bool = False

    gesture_mode: str = "MOUSE"
    gesture_event: str = ""
    gesture_event_until: float = 0.0
    gesture_input_block_until: float = 0.0
    commands_enabled: bool = False

    snap_anchor: tuple[float, float] | None = None
    snap_started_at: float | None = None
    precision_snap_active: bool = False
    last_hand_seen: float = 0.0

    fps_window_start: float = 0.0
    fps_frames: int = 0
    actual_fps: float = 0.0
    mp_fps_window_start: float = 0.0
    mp_fps_last_seq: int = 0
    actual_mp_fps: float = 0.0
    mp_infer_ms_ema: float = 0.0
    mp_worker_ms_ema: float = 0.0
    mp_cycle_ms_ema: float = 0.0
    mp_queue_ms_ema: float = 0.0

    mp_input_seq: int = 0
    mp_overwrites: int = 0
    mp_error_count: int = 0
    mp_last_error: str = ""
    camera_target_fps: int = 0
    perf: PerfProfiler = field(default_factory=PerfProfiler)
    mp_scheduler: MediaPipeSubmitScheduler = field(default_factory=MediaPipeSubmitScheduler)
    hud_layer: OverlayLayer = field(
        default_factory=lambda: OverlayLayer(HUD_REFRESH_HZ, height=370)
    )
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        camera,
        worker_cls=MediaPipeWorker,
        cursor_cls=CursorController,
        landmarker_factory=HandLandmarker,
        options=DEFAULT_OPTIONS,
        image_builder=build_mediapipe_image,
        get_volume=get_system_volume,
        time_fn=time.perf_counter,
    ):
        worker = None
        cursor = None
        try:
            worker = worker_cls(
                factory=landmarker_factory,
                options=options,
                image_builder=image_builder,
            )
            worker.start()
            cursor = cursor_cls()
            screen_w, screen_h = cursor.screen_size()
            cursor.sync(False)
            cursor.start()
            now = time_fn()
            return cls(
                camera=camera,
                worker=worker,
                cursor=cursor,
                screen_w=screen_w,
                screen_h=screen_h,
                start_time=now,
                volume=VolumeState(level=get_volume()),
                last_hand_seen=now,
                fps_window_start=now,
                mp_fps_window_start=now,
                camera_target_fps=int(getattr(camera, "target_fps", 0)),
            )
        except Exception:
            _close_resources(worker, cursor, camera)
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        _close_resources(self.worker, self.cursor, self.camera)
