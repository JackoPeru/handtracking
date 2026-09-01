"""Threaded MediaPipe inference worker.

The worker owns the landmarker context so it can never be closed by the main
thread while an inference call is still running.
"""

import threading
import time


class MediaPipeWorker(threading.Thread):
    def __init__(self, *, factory, options, image_builder):
        super().__init__(daemon=True, name="mediapipe-worker")
        self._factory = factory
        self._options = options
        self._image_builder = image_builder
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_event = threading.Event()
        self._pending = None
        self._latest = None
        self._seq = 0
        self._input_seq = 0
        self._overwrites = 0
        self._error_count = 0
        self._last_error = ""
        self._last_success_at = None
        self._last_result_input_at = None

    def submit(self, frame, gray, timestamp_ms, enqueued_at):
        with self._lock:
            self._input_seq += 1
            if self._pending is not None:
                self._overwrites += 1
            self._pending = (frame, gray, timestamp_ms, enqueued_at)
            self._pending_event.set()

    def snapshot(self):
        with self._lock:
            return self._latest

    def stats(self):
        with self._lock:
            return {
                "seq": self._seq,
                "input_seq": self._input_seq,
                "overwrites": self._overwrites,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "last_success_at": self._last_success_at,
                "last_result_input_at": self._last_result_input_at,
            }

    def snapshot_state(self):
        """Return result and worker metadata from the same locked snapshot."""
        with self._lock:
            return {
                "latest": self._latest,
                "seq": self._seq,
                "input_seq": self._input_seq,
                "overwrites": self._overwrites,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "last_success_at": self._last_success_at,
                "last_result_input_at": self._last_result_input_at,
                "alive": self.is_alive(),
            }

    @property
    def error_count(self):
        return self.stats()["error_count"]

    @property
    def last_error(self):
        return self.stats()["last_error"]

    def stop(self):
        self._stop_event.set()
        self._pending_event.set()

    def run(self):
        last_ts = -1
        last_completed_at = None
        try:
            landmarker_context = self._factory.create_from_options(self._options)
        except Exception as exc:
            with self._lock:
                self._error_count += 1
                self._last_error = f"{type(exc).__name__}: {exc}"[:140]
            return

        with landmarker_context as landmarker:
            while not self._stop_event.is_set():
                self._pending_event.wait()
                if self._stop_event.is_set():
                    break
                with self._lock:
                    packet = self._pending
                    self._pending = None
                    self._pending_event.clear()
                if packet is None:
                    continue

                frame, gray, timestamp_ms, enqueued_at = packet
                worker_started = time.perf_counter()
                queue_ms = (worker_started - enqueued_at) * 1000.0
                timestamp_ms = max(timestamp_ms, last_ts + 1)
                last_ts = timestamp_ms

                try:
                    image = self._image_builder(frame)
                    infer_started = time.perf_counter()
                    result = landmarker.detect_for_video(image, timestamp_ms)
                except Exception as exc:
                    with self._lock:
                        self._error_count += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"[:140]
                    self._stop_event.wait(0.005)
                    continue

                completed_at = time.perf_counter()
                infer_ms = (completed_at - infer_started) * 1000.0
                worker_ms = (completed_at - worker_started) * 1000.0
                cycle_ms = (
                    0.0 if last_completed_at is None
                    else (completed_at - last_completed_at) * 1000.0
                )
                last_completed_at = completed_at
                with self._lock:
                    self._seq += 1
                    self._last_success_at = completed_at
                    self._last_result_input_at = enqueued_at
                    self._latest = (
                        self._seq, result, gray, infer_ms,
                        worker_ms, cycle_ms, queue_ms,
                    )
