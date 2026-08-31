import math


def pointer_mode_allowed(*, commands_enabled, spock_blocking, hand_count,
                         paused, volume_active, two_hand_active,
                         two_hand_candidate, radial_active, scroll_active,
                         swipe_tracking, input_blocked, volume_candidate=False):
    return (
        commands_enabled and not spock_blocking and hand_count == 1 and
        not paused and not volume_active and not volume_candidate and not two_hand_active and
        not two_hand_candidate and not radial_active and not scroll_active and
        not swipe_tracking and not input_blocked
    )


def advance_confirmed_hold(accumulated, detected, sample_seconds, hold_seconds):
    """Accumulate only positively observed time; missing samples add no credit."""
    if not detected:
        return accumulated
    sample_seconds = max(0.0, sample_seconds)
    return min(max(hold_seconds, 0.0), accumulated + sample_seconds)


def spock_release_gate_active(*, required, detected,
                              release_elapsed, release_seconds):
    """Keep the Spock gate armed until a fresh non-Spock pose is held long enough."""
    if not required:
        return False
    if detected:
        return True
    return max(float(release_elapsed), 0.0) < max(float(release_seconds), 0.0)


def choose_control_index(points, labels, previous_point=None, previous_label=None,
                         label_mismatch_penalty=0.20):
    if not points:
        raise ValueError("points must not be empty")
    if previous_point is None:
        return 0

    normalized_previous = (previous_label or "").casefold()
    best_index = 0
    best_score = float("inf")
    for index, point in enumerate(points):
        distance = math.hypot(
            point[0] - previous_point[0],
            point[1] - previous_point[1],
        )
        label = labels[index] if index < len(labels) else ""
        normalized_label = (label or "").casefold()
        mismatch = (
            bool(normalized_previous) and bool(normalized_label) and
            normalized_label != normalized_previous
        )
        score = distance + (label_mismatch_penalty if mismatch else 0.0)
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def palm_motion_scale(palm_width_px, reference_width_px=90.0,
                      minimum=0.65, maximum=1.50):
    if reference_width_px <= 0:
        raise ValueError("reference_width_px must be positive")
    return max(minimum, min(maximum, palm_width_px / reference_width_px))


def normalize_flow_delta(dx, dy, palm_scale):
    scale = max(float(palm_scale), 1e-6)
    return float(dx) / scale, float(dy) / scale


def normalized_points_pixel_distance(point_a, point_b, frame_width, frame_height):
    """Distance between normalized image points measured in real frame pixels."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    dx = (float(point_a[0]) - float(point_b[0])) * frame_width
    dy = (float(point_a[1]) - float(point_b[1])) * frame_height
    return math.hypot(dx, dy)


def tracking_result_is_stale(last_success_at, now, timeout_seconds):
    if last_success_at is None:
        return True
    return float(now) - float(last_success_at) > max(float(timeout_seconds), 0.0)


def choose_camera_target_fps(reported_fps, target_fps=60, fallback_fps=30):
    """Treat zero/unknown backend reports as unknown instead of forcing fallback."""
    reported_fps = float(reported_fps or 0.0)
    if reported_fps <= 1.0:
        return int(target_fps)
    midpoint = (float(target_fps) + float(fallback_fps)) * 0.5
    return int(target_fps if reported_fps >= midpoint else fallback_fps)


def fist_evidence_from_hands(*, raw_fists, strong_fists, volume_scores, gap_scores,
                             volume_active, volume_score_on, suppress_gap):
    count = min(len(raw_fists), len(strong_fists), len(volume_scores), len(gap_scores))
    for index in range(count):
        if volume_active:
            evidence = bool(strong_fists[index])
        else:
            evidence = bool(strong_fists[index]) or (
                bool(raw_fists[index]) and not (
                    float(volume_scores[index]) >= float(volume_score_on) and
                    float(gap_scores[index]) >= float(suppress_gap)
                )
            )
        if evidence:
            return True
    return False
