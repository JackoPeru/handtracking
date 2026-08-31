"""MediaPipe scroll pose arm/release state machine."""

from handtracking_config import SCROLL_CONFIRM_SECONDS, SCROLL_RELEASE_GRACE


def update_scroll_state(scroll, *, now, gesture_now, blocked, cursor, flow):
    if blocked:
        scroll.reset()
        return

    if not scroll.active:
        if gesture_now:
            if scroll.candidate_at is None:
                scroll.candidate_at = now
            elif now - scroll.candidate_at >= SCROLL_CONFIRM_SECONDS:
                scroll.active = True
                scroll.release_at = None
                scroll.residual = 0.0
                cursor.sync(False)
                flow.clear_motion()
        else:
            scroll.candidate_at = None
        return

    if gesture_now:
        scroll.release_at = None
    else:
        if scroll.release_at is None:
            scroll.release_at = now
        elif now - scroll.release_at >= SCROLL_RELEASE_GRACE:
            scroll.reset()
            cursor.sync(True)
            flow.clear_motion()
