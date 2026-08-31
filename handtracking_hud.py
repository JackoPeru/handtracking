"""Runtime status text and diagnostic HUD rendering."""

import cv2

from handtracking_gestures import clamp


def build_status_text(gesture_mode, *, spock_progress, volume_level, radial_selected):
    if gesture_mode == "SPOCK":
        return f"SPOCK: {int(round(spock_progress * 100))}% | tieni 1 s per TOGGLE"
    if gesture_mode == "LOCKED":
        return "COMANDI BLOCCATI | fai SPOCK per 1 s"
    if gesture_mode == "FIST":
        return "PUGNO = PAUSA TRACKING"
    if gesture_mode == "VOLUME":
        return f"VOLUME: {int(round(volume_level * 100))}%"
    if gesture_mode == "TWO_HAND":
        return "2 MANI: ZOOM"
    if gesture_mode == "RADIAL":
        choice = radial_selected if radial_selected is not None else "CENTRO"
        return f"MENU RADIALE: {choice} | PINCH = OK"
    if gesture_mode == "SCROLL":
        return "SCROLL: INDICE + MEDIO"
    if gesture_mode == "SWIPE":
        return "SWIPE: 4 DITA UNITE | SX=BACK DX=FORWARD"
    if gesture_mode == "POINTER":
        return "PUNTATORE: PINCH INDICE+POLLICE | MUOVI PER SPOSTARE"
    if gesture_mode == "PINCH":
        return "PINCH: RILASCIA RAPIDO = CLICK | MUOVI = PUNTATORE"
    return "CURSORE FERMO | PINCH INDICE+POLLICE PER PUNTARE"


def draw_runtime_hud(
    frame,
    *,
    gesture_mode,
    gesture_event,
    gesture_event_until,
    now,
    flow_active,
    commands_enabled,
    spock_blocking,
    spock_latched,
    spock_progress,
    volume_level,
    radial_selected,
    actual_fps,
    actual_mp_fps,
    mp_infer_ms,
    mp_worker_ms,
    mp_cycle_ms,
    mp_queue_ms,
    mp_overwrites,
    mp_input_seq,
    camera_codec,
    reported_w,
    reported_h,
    reported_fps,
    camera_target_fps,
    debug_fist_score,
    debug_volume_score,
    debug_grip_gap,
    debug_fist_folded,
    debug_fist_tightness,
    debug_strong_fist,
    spock_debug_score,
    spock_debug_stable,
    swipe_debug_score,
    swipe_debug_stable,
    swipe_debug_gap,
    swipe_debug_extended,
    mp_error_count,
    mp_last_error,
):
    status = build_status_text(
        gesture_mode,
        spock_progress=spock_progress,
        volume_level=volume_level,
        radial_selected=radial_selected,
    )
    flow_label = "FLOW ON" if flow_active else "FLOW WAIT"
    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Camera {actual_fps:.1f} | MP {actual_mp_fps:.1f} FPS | call {mp_infer_ms:.0f} ms | worker {mp_worker_ms:.0f} ms | cycle {mp_cycle_ms:.0f} ms | {flow_label}",
        (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
        0.66, (255, 255, 255), 2, cv2.LINE_AA,
    )
    drop_pct = 100.0 * mp_overwrites / max(mp_input_seq, 1)
    cv2.putText(
        frame,
        f"MP queue {mp_queue_ms:.1f} ms | drop {drop_pct:.0f}% | {camera_codec} {reported_w}x{reported_h}@{reported_fps:.0f} | target {camera_target_fps} | ESC",
        (30, 125), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FIST {debug_fist_score:.2f} | VOL {debug_volume_score:.2f} | GAP {debug_grip_gap:.2f}",
        (30, 158), cv2.FONT_HERSHEY_SIMPLEX,
        0.64, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FOLD {debug_fist_folded}/4 | TIGHT {debug_fist_tightness:.2f} | STRONG {int(debug_strong_fist)}",
        (30, 190), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ENGINE: {gesture_mode} | SPOCK raw {spock_debug_score:.2f} stable {spock_debug_stable:.2f}",
        (30, 222), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (200, 255, 200), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"SWIPE raw {swipe_debug_score:.2f} stable {swipe_debug_stable:.2f} | JOIN {swipe_debug_gap:.2f} | EXT {swipe_debug_extended}/4",
        (30, 254), cv2.FONT_HERSHEY_SIMPLEX,
        0.56, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if gesture_event and now < gesture_event_until:
        cv2.putText(
            frame, gesture_event,
            (30, 286), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (0, 255, 255), 2, cv2.LINE_AA,
        )
    if mp_error_count:
        cv2.putText(
            frame,
            f"MP ERR {mp_error_count}: {mp_last_error}",
            (30, 318), cv2.FONT_HERSHEY_SIMPLEX,
            0.50, (0, 80, 255), 2, cv2.LINE_AA,
        )

    frame_w = frame.shape[1]
    if spock_blocking:
        led_color = (0, 180, 255)
    elif commands_enabled:
        led_color = (0, 255, 0)
    else:
        led_color = (0, 0, 255)
    led_x, led_y = frame_w - 58, 45
    cv2.circle(frame, (led_x, led_y), 17, led_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (led_x, led_y), 20, (255, 255, 255), 2, cv2.LINE_AA)
    led_label = "CMD ON" if commands_enabled else "CMD OFF"
    cv2.putText(frame, led_label, (frame_w - 155, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, led_color, 2, cv2.LINE_AA)

    if spock_blocking and not spock_latched:
        bar_w, bar_h = 260, 16
        bar_x, bar_y = frame_w - bar_w - 30, 108
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 2)
        fill_w = int(bar_w * clamp(spock_progress, 0.0, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), led_color, -1)
