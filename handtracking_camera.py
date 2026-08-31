"""Camera ownership, configuration and per-frame preprocessing."""

from dataclasses import dataclass

import cv2

from handtracking_config import (
    CAMERA_H,
    CAMERA_W,
    DETECTION_H,
    DETECTION_W,
    FALLBACK_FPS,
    TARGET_FPS,
)
from handtracking_core import choose_camera_target_fps


@dataclass(frozen=True)
class PreparedFrame:
    frame: object
    detect_frame: object
    gray: object


@dataclass
class CameraRuntime:
    capture: object
    reported_fps: float
    reported_w: int
    reported_h: int
    codec: str
    target_fps: int
    cv2_module: object = cv2
    _closed: bool = False

    @classmethod
    def open(cls, *, cv2_module=cv2):
        cap = cv2_module.VideoCapture(0, cv2_module.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            cap = cv2_module.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Impossibile aprire la webcam")

        cap.set(
            cv2_module.CAP_PROP_FOURCC,
            cv2_module.VideoWriter_fourcc(*"MJPG"),
        )
        cap.set(cv2_module.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        cap.set(cv2_module.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        cap.set(cv2_module.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)

        reported_fps = float(cap.get(cv2_module.CAP_PROP_FPS))
        reported_w = int(round(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH)))
        reported_h = int(round(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT)))
        reported_fourcc = int(cap.get(cv2_module.CAP_PROP_FOURCC))
        codec = "".join(
            chr((reported_fourcc >> (8 * i)) & 0xFF) for i in range(4)
        ).replace("\x00", "") or "?"
        target_fps = choose_camera_target_fps(
            reported_fps,
            TARGET_FPS,
            FALLBACK_FPS,
        )

        cv2_module.namedWindow("Hands", cv2_module.WINDOW_NORMAL)
        cv2_module.setWindowProperty(
            "Hands",
            cv2_module.WND_PROP_FULLSCREEN,
            cv2_module.WINDOW_FULLSCREEN,
        )
        return cls(
            capture=cap,
            reported_fps=reported_fps,
            reported_w=reported_w,
            reported_h=reported_h,
            codec=codec,
            target_fps=target_fps,
            cv2_module=cv2_module,
        )

    def read_prepared(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        frame = self.cv2_module.flip(frame, 1)
        detect_frame = self.cv2_module.resize(
            frame,
            (DETECTION_W, DETECTION_H),
            interpolation=self.cv2_module.INTER_AREA,
        )
        gray = self.cv2_module.cvtColor(
            detect_frame,
            self.cv2_module.COLOR_BGR2GRAY,
        )
        return PreparedFrame(frame, detect_frame, gray)

    def show(self, frame):
        self.cv2_module.imshow("Hands", frame)
        return (self.cv2_module.waitKey(1) & 0xFF) != 27

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.capture.release()
        finally:
            self.cv2_module.destroyAllWindows()
