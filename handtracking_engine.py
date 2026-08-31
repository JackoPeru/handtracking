"""Pure gesture arbitration for the runtime."""


def resolve_runtime_mode(*, commands_enabled, spock_blocking, paused,
                         volume, two_hand, radial, scrolling, swipe,
                         pointer_move, pointer_pinch):
    if spock_blocking:
        return "SPOCK"
    if not commands_enabled:
        return "LOCKED"
    if paused:
        return "FIST"
    if volume:
        return "VOLUME"
    if two_hand:
        return "TWO_HAND"
    if radial:
        return "RADIAL"
    if scrolling:
        return "SCROLL"
    if swipe:
        return "SWIPE"
    if pointer_move:
        return "POINTER"
    if pointer_pinch:
        return "PINCH"
    return "MOUSE"
