"""Cached OpenCV overlay layers for decoupled render/update rates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class OverlayLayer:
    refresh_hz: float
    cv2_module: object = cv2
    height: int | None = None
    next_refresh_at: float = 0.0
    overlay: object | None = None
    mask: object | None = None

    @property
    def interval(self):
        return 1.0 / max(float(self.refresh_hz), 1.0)

    def should_refresh(self, frame, now):
        region = self._region(frame)
        return (
            self.overlay is None or
            self.overlay.shape != region.shape or
            now >= self.next_refresh_at
        )

    def begin(self, frame):
        region = self._region(frame)
        if self.overlay is None or self.overlay.shape != region.shape:
            self.overlay = np.zeros_like(region)
        else:
            self.overlay.fill(0)
        return self.overlay

    def finish(self, now):
        self.mask = self.cv2_module.cvtColor(
            self.overlay, self.cv2_module.COLOR_BGR2GRAY
        )
        self.next_refresh_at = float(now) + self.interval

    def apply(self, frame):
        if self.overlay is not None and self.mask is not None:
            self.cv2_module.copyTo(self.overlay, self.mask, self._region(frame))

    def _region(self, frame):
        if self.height is None:
            return frame
        return frame[:min(int(self.height), frame.shape[0])]
