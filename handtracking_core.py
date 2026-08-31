import math


def pointer_mode_allowed(*, commands_enabled, spock_blocking, hand_count,
                         paused, volume_active, two_hand_active,
                         two_hand_candidate, radial_active, scroll_active,
                         swipe_tracking, input_blocked):
    return (
        commands_enabled and not spock_blocking and hand_count == 1 and
        not paused and not volume_active and not two_hand_active and
        not two_hand_candidate and not radial_active and not scroll_active and
        not swipe_tracking and not input_blocked
    )


def advance_confirmed_hold(accumulated, detected, sample_seconds, hold_seconds):
    """Accumulate only positively observed time; missing samples add no credit."""
    if not detected:
        return accumulated
    sample_seconds = max(0.0, sample_seconds)
    return min(max(hold_seconds, 0.0), accumulated + sample_seconds)


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
