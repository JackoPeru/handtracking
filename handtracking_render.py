"""OpenCV rendering helpers. These functions do not mutate gesture state."""

import cv2

from handtracking_gestures import mouse_point


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_hand(frame, hand, pinch_active=False, paused=False,
              scrolling=False, volume_control=False):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand]
    if volume_control:
        line_color = point_color = (255, 0, 255)
    elif paused:
        line_color = point_color = (0, 165, 255)
    elif scrolling:
        line_color = point_color = (255, 180, 0)
    elif pinch_active:
        line_color = point_color = (0, 0, 255)
    else:
        line_color, point_color = (255, 255, 255), (0, 255, 0)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 3, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 8 if i in (4, 8, 12) else 5, point_color, -1, cv2.LINE_AA)
    if pinch_active:
        mx, my = mouse_point(hand)
        cv2.circle(frame, (int(mx * w), int(my * h)), 13, (0, 255, 255), 3, cv2.LINE_AA)


def draw_radial_menu(frame, center, selected):
    h, w = frame.shape[:2]
    cx, cy = int(center[0] * w), int(center[1] * h)
    radius = int(min(w, h) * 0.115)
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(radius * 0.35), (180, 180, 180), 1, cv2.LINE_AA)
    items = {
        "UP": ((cx, cy - radius), "TASK"),
        "RIGHT": ((cx + radius, cy), "APP"),
        "DOWN": ((cx, cy + radius), "DESKTOP"),
        "LEFT": ((cx - radius, cy), "BACK"),
    }
    for direction, (pt, label) in items.items():
        active = direction == selected
        color = (0, 255, 255) if active else (255, 255, 255)
        thickness = 3 if active else 1
        cv2.circle(frame, pt, 24 if active else 19, color, thickness, cv2.LINE_AA)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(
            frame,
            label,
            (pt[0] - size[0] // 2, pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_two_hand_transform(frame, point_a, point_b):
    h, w = frame.shape[:2]
    a = (int(point_a[0] * w), int(point_a[1] * h))
    b = (int(point_b[0] * w), int(point_b[1] * h))
    cv2.line(frame, a, b, (255, 255, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, a, 14, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, b, 14, (255, 255, 0), 3, cv2.LINE_AA)
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    cv2.putText(
        frame,
        "ZOOM",
        (mx - 42, my - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_runtime_overlays(
    frame,
    *,
    latest_result,
    fist_states,
    control_index,
    pinch_active,
    scroll_active,
    volume_active,
    radial_active,
    radial_center,
    radial_selected,
    two_hand_active,
    two_hand_points,
):
    if latest_result is not None and latest_result.hand_landmarks:
        for i, hand in enumerate(latest_result.hand_landmarks):
            paused = fist_states[i] if i < len(fist_states) else False
            draw_hand(
                frame,
                hand,
                pinch_active=(i == control_index and pinch_active),
                paused=paused,
                scrolling=(i == control_index and scroll_active),
                volume_control=(i == control_index and volume_active),
            )

    if radial_active and radial_center is not None:
        draw_radial_menu(frame, radial_center, radial_selected)
    if two_hand_active and two_hand_points is not None:
        draw_two_hand_transform(frame, two_hand_points[0], two_hand_points[1])
