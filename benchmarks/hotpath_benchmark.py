"""Synthetic, reproducible micro-benchmarks for runtime hot paths."""

from types import SimpleNamespace
import time

import cv2
import numpy as np

from handtracking_display import OverlayLayer
from handtracking_flow import measure_optical_flow
from handtracking_gestures import (
    HandFeatures,
    fist_fold_metrics,
    grip_class_scores,
    is_fist,
    is_scroll_gesture,
    is_strong_fist,
    normalized_pinch_ratio,
    pointer_other_fingers_valid,
    spock_all_fingers_up,
    spock_pose_score,
)
from handtracking_hud import draw_runtime_hud
from handtracking_render import draw_runtime_overlays


def _make_hand():
    return [
        SimpleNamespace(
            x=(i % 5) * 0.04 + 0.30,
            y=(i // 5) * 0.04 + 0.25,
            z=(i % 3) * 0.01,
        )
        for i in range(21)
    ]


def _geometry_pass(hand):
    grip_class_scores(hand)
    is_fist(hand)
    is_strong_fist(hand)
    fist_fold_metrics(hand)
    is_scroll_gesture(hand)
    spock_pose_score(hand)
    spock_all_fingers_up(hand)
    normalized_pinch_ratio(hand, 8)
    pointer_other_fingers_valid(hand)


def _hud_kwargs(now=0.0):
    return dict(
        gesture_mode="MOUSE",
        gesture_event="",
        gesture_event_until=0.0,
        now=now,
        flow_active=False,
        commands_enabled=True,
        spock_blocking=False,
        spock_latched=False,
        spock_progress=0.0,
        volume_level=0.5,
        radial_selected=None,
        actual_fps=60.0,
        actual_mp_fps=25.0,
        mp_infer_ms=30.0,
        mp_worker_ms=31.0,
        mp_cycle_ms=40.0,
        mp_queue_ms=2.0,
        mp_overwrites=10,
        mp_input_seq=100,
        camera_codec="MJPG",
        reported_w=1280,
        reported_h=720,
        reported_fps=60.0,
        camera_target_fps=60,
        debug_fist_score=0.1,
        debug_volume_score=0.2,
        debug_grip_gap=0.5,
        debug_fist_folded=2,
        debug_fist_tightness=1.1,
        debug_strong_fist=False,
        spock_debug_score=0.2,
        spock_debug_stable=0.3,
        swipe_debug_score=0.2,
        swipe_debug_stable=0.2,
        swipe_debug_gap=0.4,
        swipe_debug_extended=4,
        mp_error_count=0,
        mp_last_error="",
    )


def _average_seconds(fn, iterations):
    started = time.perf_counter()
    for _ in range(max(int(iterations), 1)):
        fn()
    return (time.perf_counter() - started) / max(int(iterations), 1)


def run_benchmarks(*, iterations=20_000, lk_iterations=300, render_iterations=600):
    hand = _make_hand()
    raw_seconds = _average_seconds(lambda: _geometry_pass(hand), iterations)
    cached_seconds = _average_seconds(
        lambda: _geometry_pass(HandFeatures(hand)), iterations
    )

    old_gray = np.zeros((360, 640), dtype=np.uint8)
    cv2.circle(old_gray, (320, 180), 80, 255, -1)
    new_gray = np.roll(old_gray, 1, axis=1)
    points = np.array(
        [[[280, 180]], [[320, 140]], [[320, 180]], [[320, 220]], [[360, 180]]],
        dtype=np.float32,
    )
    lk_seconds = _average_seconds(
        lambda: measure_optical_flow(old_gray, new_gray, points, 1.0),
        lk_iterations,
    )

    full_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def preprocess():
        detect = cv2.resize(full_frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)

    preprocess_seconds = _average_seconds(preprocess, render_iterations)

    hud_kwargs = _hud_kwargs()

    def direct_hud():
        frame = full_frame.copy()
        draw_runtime_hud(frame, **hud_kwargs)

    direct_hud_seconds = _average_seconds(direct_hud, render_iterations)

    layer = OverlayLayer(refresh_hz=12.0, height=370)
    now = 0.0

    def cached_hud():
        nonlocal now
        frame = full_frame.copy()
        if layer.should_refresh(frame, now):
            canvas = layer.begin(frame)
            kwargs = dict(hud_kwargs)
            kwargs["now"] = now
            draw_runtime_hud(canvas, **kwargs)
            layer.finish(now)
        layer.apply(frame)
        now += 1.0 / 60.0

    cached_hud_seconds = _average_seconds(cached_hud, render_iterations)

    latest_result = SimpleNamespace(hand_landmarks=[hand])

    def direct_overlay():
        frame = full_frame.copy()
        draw_runtime_overlays(
            frame,
            latest_result=latest_result,
            fist_states=[False],
            control_index=0,
            pinch_active=False,
            scroll_active=False,
            volume_active=False,
            radial_active=False,
            radial_center=None,
            radial_selected=None,
            two_hand_active=False,
            two_hand_points=None,
        )

    direct_overlay_seconds = _average_seconds(direct_overlay, render_iterations)

    overlay_layer = OverlayLayer(refresh_hz=30.0)
    overlay_now = 0.0

    def cached_overlay():
        nonlocal overlay_now
        frame = full_frame.copy()
        if overlay_layer.should_refresh(frame, overlay_now):
            canvas = overlay_layer.begin(frame)
            draw_runtime_overlays(
                canvas,
                latest_result=latest_result,
                fist_states=[False],
                control_index=0,
                pinch_active=False,
                scroll_active=False,
                volume_active=False,
                radial_active=False,
                radial_center=None,
                radial_selected=None,
                two_hand_active=False,
                two_hand_points=None,
            )
            overlay_layer.finish(overlay_now)
        overlay_layer.apply(frame)
        overlay_now += 1.0 / 60.0

    cached_overlay_seconds = _average_seconds(cached_overlay, render_iterations)

    return {
        "geometry_raw_us": raw_seconds * 1_000_000.0,
        "geometry_cached_us": cached_seconds * 1_000_000.0,
        "lk_ms": lk_seconds * 1_000.0,
        "preprocess_ms": preprocess_seconds * 1_000.0,
        "hud_direct_ms": direct_hud_seconds * 1_000.0,
        "hud_cached_ms": cached_hud_seconds * 1_000.0,
        "overlay_direct_ms": direct_overlay_seconds * 1_000.0,
        "overlay_cached_ms": cached_overlay_seconds * 1_000.0,
    }


def main():
    results = run_benchmarks()
    for name, value in results.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
