"""Low-overhead runtime performance counters."""

from collections import deque
from dataclasses import dataclass
import math
import time


PERF_HISTORY_SIZE = 256


@dataclass(frozen=True, slots=True)
class PerfMetric:
    samples: int = 0
    last_ms: float = 0.0
    ema_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


ZERO_METRIC = PerfMetric()


def percentile_metric(values):
    """Return nearest-rank p50/p95/p99 from finite values."""
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            finite.append(value)
    ordered = sorted(finite)
    if not ordered:
        return (0.0, 0.0, 0.0)

    def nearest_rank(fraction):
        rank = max(1, math.ceil(fraction * len(ordered)))
        return ordered[rank - 1]

    return tuple(nearest_rank(fraction) for fraction in (0.50, 0.95, 0.99))


class PerfProfiler:
    __slots__ = ("alpha", "_metrics", "_histories")

    def __init__(self, *, alpha=0.12):
        self.alpha = float(alpha)
        self._metrics = {}
        self._histories = {}

    @staticmethod
    def now_ns():
        return time.perf_counter_ns()

    def observe_ns(self, name, started_ns, ended_ns=None):
        if ended_ns is None:
            ended_ns = time.perf_counter_ns()
        self.observe_ms(name, (ended_ns - started_ns) / 1_000_000.0)

    def observe_ms(self, name, value_ms):
        try:
            value_ms = float(value_ms)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value_ms):
            return False
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
        history = self._histories.get(name)
        if history is None:
            history = deque(maxlen=PERF_HISTORY_SIZE)
            self._histories[name] = history
        history.append(value_ms)
        return True

    def metric(self, name):
        metric = self._metrics.get(name)
        if metric is None:
            return ZERO_METRIC
        history = tuple(self._histories.get(name, ()))
        if not history:
            return metric
        p50_ms, p95_ms, p99_ms = percentile_metric(history)
        return PerfMetric(
            metric.samples, metric.last_ms, metric.ema_ms,
            p50_ms, p95_ms, p99_ms,
        )


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
