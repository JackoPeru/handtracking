"""Low-overhead runtime performance counters."""

from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class PerfMetric:
    samples: int = 0
    last_ms: float = 0.0
    ema_ms: float = 0.0


ZERO_METRIC = PerfMetric()


class PerfProfiler:
    __slots__ = ("alpha", "_metrics")

    def __init__(self, *, alpha=0.12):
        self.alpha = float(alpha)
        self._metrics = {}

    @staticmethod
    def now_ns():
        return time.perf_counter_ns()

    def observe_ns(self, name, started_ns, ended_ns=None):
        if ended_ns is None:
            ended_ns = time.perf_counter_ns()
        self.observe_ms(name, (ended_ns - started_ns) / 1_000_000.0)

    def observe_ms(self, name, value_ms):
        value_ms = float(value_ms)
        previous = self._metrics.get(name)
        if previous is None:
            metric = PerfMetric(1, value_ms, value_ms)
        else:
            metric = PerfMetric(
                previous.samples + 1,
                value_ms,
                previous.ema_ms * (1.0 - self.alpha) + value_ms * self.alpha,
            )
        self._metrics[name] = metric

    def metric(self, name):
        return self._metrics.get(name, ZERO_METRIC)


class MediaPipeSubmitScheduler:
    __slots__ = ("min_fps", "cycle_fraction", "last_check_at", "credit")

    def __init__(self, *, min_fps=20.0, cycle_fraction=0.5):
        self.min_fps = float(min_fps)
        self.cycle_fraction = float(cycle_fraction)
        self.last_check_at = None
        self.credit = 0.0

    def should_submit(self, now, *, cycle_ms, target_fps):
        now = float(now)
        camera_interval = 1.0 / max(float(target_fps), 1.0)
        inferred_interval = max(float(cycle_ms), 0.0) / 1000.0 * self.cycle_fraction
        if self.last_check_at is None:
            self.last_check_at = now
            return True

        elapsed = max(now - self.last_check_at, 0.0)
        self.last_check_at = now

        if inferred_interval <= camera_interval:
            self.credit = 0.0
            return True

        interval = max(camera_interval, inferred_interval)
        interval = min(interval, 1.0 / max(self.min_fps, 1.0))
        self.credit += elapsed / interval
        if self.credit >= 1.0:
            self.credit -= 1.0
            return True
        return False
