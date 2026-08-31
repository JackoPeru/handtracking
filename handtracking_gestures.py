"""Pure landmark geometry and gesture classifiers."""

import math

from handtracking_config import *


def clamp(value, low, high):
    return max(low, min(high, value))


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def dist3(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def joint_angle3(a, b, c):
    bax, bay, baz = a.x - b.x, a.y - b.y, a.z - b.z
    bcx, bcy, bcz = c.x - b.x, c.y - b.y, c.z - b.z
    na = math.sqrt(bax * bax + bay * bay + baz * baz)
    nc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cosine = clamp((bax * bcx + bay * bcy + baz * bcz) / (na * nc), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def control_point(hand):
    mid_x = (hand[9].x + hand[13].x) * 0.5
    mid_y = (hand[9].y + hand[13].y) * 0.5
    return mid_x * 0.82 + hand[0].x * 0.18, mid_y * 0.82 + hand[0].y * 0.18


def pinch_contact_point(hand):
    return (hand[4].x + hand[8].x) * 0.5, (hand[4].y + hand[8].y) * 0.5


def mouse_point(hand):
    return pinch_contact_point(hand)


def grip_gap_ratio(hand):
    palm_width = max(dist3(hand[5], hand[17]), 0.001)
    cx = (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5.0
    cy = (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5.0
    cz = (hand[0].z + hand[5].z + hand[9].z + hand[13].z + hand[17].z) / 5.0
    gaps = []
    for tip_id in (8, 12, 16, 20):
        dx = hand[tip_id].x - cx
        dy = hand[tip_id].y - cy
        dz = hand[tip_id].z - cz
        gaps.append(math.sqrt(dx * dx + dy * dy + dz * dz) / palm_width)
    return sum(gaps) / len(gaps)


def finger_curl_score(hand, mcp, pip, dip, tip):
    pip_angle = joint_angle3(hand[mcp], hand[pip], hand[dip])
    dip_angle = joint_angle3(hand[pip], hand[dip], hand[tip])
    pip_score = clamp((172.0 - pip_angle) / 58.0, 0.0, 1.0)
    dip_score = clamp((176.0 - dip_angle) / 58.0, 0.0, 1.0)
    return max(pip_score, dip_score)


def grip_class_scores(hand):
    gap = grip_gap_ratio(hand)
    curls = [
        finger_curl_score(hand, 5, 6, 7, 8),
        finger_curl_score(hand, 9, 10, 11, 12),
        finger_curl_score(hand, 13, 14, 15, 16),
        finger_curl_score(hand, 17, 18, 19, 20),
    ]
    thumb = max(
        clamp((170.0 - joint_angle3(hand[1], hand[2], hand[3])) / 55.0, 0.0, 1.0),
        clamp((176.0 - joint_angle3(hand[2], hand[3], hand[4])) / 55.0, 0.0, 1.0),
    )
    finger_mean = sum(curls) / 4.0
    fist_gap = clamp(
        (FIST_SCORE_GAP_HIGH - gap) / (FIST_SCORE_GAP_HIGH - FIST_SCORE_GAP_LOW),
        0.0, 1.0,
    )
    fist_score = 0.58 * fist_gap + 0.42 * finger_mean
    all_curls = sorted(curls + [thumb])
    curl_consistency = all_curls[1]
    gap_up = clamp(
        (gap - VOLUME_SCORE_GAP_LOW) / (VOLUME_SCORE_GAP_OPT - VOLUME_SCORE_GAP_LOW),
        0.0, 1.0,
    )
    gap_down = clamp(
        (VOLUME_SCORE_GAP_HIGH - gap) / (VOLUME_SCORE_GAP_HIGH - VOLUME_SCORE_GAP_OPT),
        0.0, 1.0,
    )
    volume_gap = gap_up * gap_down
    volume_score = 0.76 * curl_consistency + 0.24 * volume_gap
    return fist_score, volume_score, gap


def fist_fold_metrics(hand):
    wrist = hand[0]
    ratios = []
    folded = 0
    for pip_id, tip_id in ((6, 8), (10, 12), (14, 16), (18, 20)):
        pip_dist = max(dist(hand[pip_id], wrist), 0.001)
        ratio = dist(hand[tip_id], wrist) / pip_dist
        ratios.append(ratio)
        if ratio < 1.08:
            folded += 1
    tightness = sum(sorted(ratios)[:3]) / 3.0
    return folded, tightness


def is_fist(hand):
    if is_point_pose(hand):
        return False
    folded, _ = fist_fold_metrics(hand)
    return folded >= 3


def is_strong_fist(hand):
    if is_point_pose(hand):
        return False
    folded, _ = fist_fold_metrics(hand)
    gap = grip_gap_ratio(hand)
    return (
        (folded >= 4 and gap <= FIST_VOLUME_OVERRIDE_GAP_MAX) or
        (folded >= 3 and gap <= FIST_VOLUME_OVERRIDE_GAP_3F)
    )


def is_scroll_gesture(hand):
    wrist = hand[0]
    hand_scale = max(dist(wrist, hand[9]), 0.001)
    index_extended = dist(hand[8], wrist) > dist(hand[6], wrist) * 1.12
    middle_extended = dist(hand[12], wrist) > dist(hand[10], wrist) * 1.12
    ring_folded = dist(hand[16], wrist) < dist(hand[14], wrist) * 1.08
    pinky_folded = dist(hand[20], wrist) < dist(hand[18], wrist) * 1.08
    fingers_joined = dist(hand[8], hand[12]) / hand_scale < SCROLL_FINGER_JOIN
    return index_extended and middle_extended and ring_folded and pinky_folded and fingers_joined


def is_open_hand(hand):
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        pip_angle = joint_angle3(hand[mcp], hand[pip], hand[dip])
        dip_angle = joint_angle3(hand[pip], hand[dip], hand[tip])
        if pip_angle < VOLUME_OPEN_MIN_DEG or dip_angle < VOLUME_OPEN_MIN_DEG:
            return False
    return True


def is_volume_release_pose(hand):
    extended = 0
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        pip_angle = joint_angle3(hand[mcp], hand[pip], hand[dip])
        dip_angle = joint_angle3(hand[pip], hand[dip], hand[tip])
        if pip_angle >= VOLUME_RELEASE_MIN_DEG and dip_angle >= VOLUME_RELEASE_MIN_DEG:
            extended += 1
    return extended >= VOLUME_RELEASE_MIN_FINGERS


def palm_roll_angle(hand):
    return math.atan2(hand[5].y - hand[17].y, hand[5].x - hand[17].x)


def wrapped_angle_delta(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def normalized_pinch_ratio(hand, finger_tip=8):
    scale = max(dist(hand[0], hand[9]), 0.001)
    return dist(hand[4], hand[finger_tip]) / scale


def pointer_other_fingers_valid(hand):
    curls = [
        finger_curl_score(hand, 9, 10, 11, 12),
        finger_curl_score(hand, 13, 14, 15, 16),
        finger_curl_score(hand, 17, 18, 19, 20),
    ]
    return (
        max(curls) <= POINTER_OTHER_FINGER_CURL_MAX and
        sum(c <= POINTER_OTHER_FINGER_CURL_SOFT for c in curls) >= 2 and
        sum(curls) / 3.0 <= POINTER_OTHER_FINGER_CURL_MEAN_MAX
    )


def is_pointer_pinch_pose(hand, pinch_limit):
    return (
        normalized_pinch_ratio(hand, 8) < pinch_limit and
        pointer_other_fingers_valid(hand)
    )


def is_point_pose(hand):
    wrist = hand[0]
    index_extended = (
        dist(hand[8], wrist) > dist(hand[6], wrist) * 1.14 and
        joint_angle3(hand[5], hand[6], hand[7]) >= 158.0 and
        joint_angle3(hand[6], hand[7], hand[8]) >= 154.0
    )
    folded = 0
    for pip_id, tip_id in ((10, 12), (14, 16), (18, 20)):
        if dist(hand[tip_id], wrist) < dist(hand[pip_id], wrist) * 1.08:
            folded += 1
    return index_extended and folded == 3


def swipe_pose_metrics(hand):
    wrist = hand[0]
    palm_width = max(dist(hand[5], hand[17]), 0.001)
    extension_scores = []
    straight_scores = []
    extended_count = 0
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8), (9, 10, 11, 12),
        (13, 14, 15, 16), (17, 18, 19, 20),
    ):
        pip_dist = max(dist(hand[pip], wrist), 0.001)
        extension_ratio = dist(hand[tip], wrist) / pip_dist
        if extension_ratio >= SWIPE_EXTENSION_COUNT_MIN:
            extended_count += 1
        extension_scores.append(clamp(
            (extension_ratio - SWIPE_EXTENSION_SCORE_LOW) /
            (SWIPE_EXTENSION_SCORE_HIGH - SWIPE_EXTENSION_SCORE_LOW), 0.0, 1.0,
        ))
        pip_angle = joint_angle3(hand[mcp], hand[pip], hand[dip])
        dip_angle = joint_angle3(hand[pip], hand[dip], hand[tip])
        straight_angle = min(pip_angle, dip_angle)
        straight_scores.append(clamp(
            (straight_angle - SWIPE_STRAIGHT_SCORE_LOW) /
            (SWIPE_STRAIGHT_SCORE_HIGH - SWIPE_STRAIGHT_SCORE_LOW), 0.0, 1.0,
        ))
    joined_gaps = (
        (0.62 * dist(hand[8], hand[12]) + 0.38 * dist(hand[7], hand[11])) / palm_width,
        (0.62 * dist(hand[12], hand[16]) + 0.38 * dist(hand[11], hand[15])) / palm_width,
        (0.62 * dist(hand[16], hand[20]) + 0.38 * dist(hand[15], hand[19])) / palm_width,
    )
    max_gap = max(joined_gaps)
    join_score = clamp(
        (SWIPE_JOIN_SCORE_BAD - max_gap) /
        (SWIPE_JOIN_SCORE_BAD - SWIPE_JOIN_SCORE_GOOD), 0.0, 1.0,
    )
    extension_score = sum(extension_scores) / 4.0
    straight_score = sum(straight_scores) / 4.0
    score = 0.50 * join_score + 0.32 * extension_score + 0.18 * straight_score
    if extended_count < 3:
        score *= 0.35
    return clamp(score, 0.0, 1.0), max_gap, extended_count


def two_hand_geometry(hand_a, hand_b):
    ax, ay = control_point(hand_a)
    bx, by = control_point(hand_b)
    dx, dy = bx - ax, by - ay
    return math.hypot(dx, dy), (ax, ay), (bx, by)


def is_radial_open_pose(hand):
    return (
        is_open_hand(hand) and
        normalized_pinch_ratio(hand, 8) >= RADIAL_THUMB_OPEN_MIN
    )


def spock_all_fingers_up(hand):
    palm_height = max(dist(hand[0], hand[9]), 0.001)
    wrist = hand[0]
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        rise_mcp = (hand[mcp].y - hand[tip].y) / palm_height
        rise_pip = (hand[pip].y - hand[tip].y) / palm_height
        long_enough = dist(hand[tip], wrist) > dist(hand[pip], wrist) * SPOCK_TIP_EXTENSION_RATIO
        if (rise_mcp < SPOCK_UP_GUARD_MCP or
                rise_pip < SPOCK_UP_GUARD_PIP or
                not long_enough):
            return False
    return True


def spock_pose_score(hand):
    palm_width = max(dist(hand[5], hand[17]), 0.001)
    palm_height = max(dist(hand[0], hand[9]), 0.001)
    up_scores = []
    extended = 0
    wrist = hand[0]
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8), (9, 10, 11, 12),
        (13, 14, 15, 16), (17, 18, 19, 20),
    ):
        rise_from_mcp = (hand[mcp].y - hand[tip].y) / palm_height
        rise_from_pip = (hand[pip].y - hand[tip].y) / palm_height
        up_score = min(
            clamp((rise_from_mcp - SPOCK_UP_MCP_MIN) /
                  (SPOCK_UP_MCP_GOOD - SPOCK_UP_MCP_MIN), 0.0, 1.0),
            clamp((rise_from_pip - SPOCK_UP_PIP_MIN) /
                  (SPOCK_UP_PIP_GOOD - SPOCK_UP_PIP_MIN), 0.0, 1.0),
        )
        up_scores.append(up_score)
        if (dist(hand[tip], wrist) > dist(hand[pip], wrist) * SPOCK_TIP_EXTENSION_RATIO and
                up_score >= SPOCK_UP_FINGER_ACCEPT):
            extended += 1
    if min(up_scores) < SPOCK_UP_HARD_REJECT or extended < 4:
        return 0.0
    pair_left = (
        0.65 * dist(hand[8], hand[12]) + 0.35 * dist(hand[7], hand[11])
    ) / palm_width
    center_gap = (
        0.65 * dist(hand[12], hand[16]) + 0.35 * dist(hand[11], hand[15])
    ) / palm_width
    pair_right = (
        0.65 * dist(hand[16], hand[20]) + 0.35 * dist(hand[15], hand[19])
    ) / palm_width
    pair_gap = max(pair_left, pair_right)
    pair_score = clamp((SPOCK_PAIR_GAP_REJECT - pair_gap) /
                       (SPOCK_PAIR_GAP_REJECT - SPOCK_PAIR_GAP_GOOD), 0.0, 1.0)
    split_abs = clamp((center_gap - SPOCK_CENTER_GAP_MIN) /
                      (SPOCK_CENTER_GAP_GOOD - SPOCK_CENTER_GAP_MIN), 0.0, 1.0)
    split_ratio = center_gap / max(pair_gap, 0.06)
    split_rel = clamp((split_ratio - SPOCK_CENTER_RATIO_MIN) /
                      (SPOCK_CENTER_RATIO_GOOD - SPOCK_CENTER_RATIO_MIN), 0.0, 1.0)
    upright_score = sum(up_scores) / 4.0
    score = 0.31 * pair_score + 0.25 * split_rel + 0.22 * split_abs + 0.22 * upright_score
    if (pair_gap <= SPOCK_PAIR_GAP_CLEAR and
            center_gap >= SPOCK_CENTER_GAP_CLEAR and
            split_ratio >= SPOCK_CENTER_RATIO_CLEAR and
            min(up_scores) >= SPOCK_UP_CLEAR):
        score = max(score, SPOCK_CLEAR_SCORE)
    return clamp(score, 0.0, 1.0)


def radial_direction(hand, center):
    px, py = control_point(hand)
    dx = px - center[0]
    dy = py - center[1]
    if math.hypot(dx, dy) < RADIAL_SELECT_RADIUS:
        return None
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"
