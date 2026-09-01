import unittest


class PerfProfilerTests(unittest.TestCase):
    def test_profiler_tracks_ema_and_sample_count(self):
        from handtracking_perf import PerfProfiler

        profiler = PerfProfiler(alpha=0.25)
        profiler.observe_ms("flow", 4.0)
        profiler.observe_ms("flow", 8.0)

        metric = profiler.metric("flow")
        self.assertEqual(metric.samples, 2)
        self.assertAlmostEqual(metric.last_ms, 8.0)
        self.assertAlmostEqual(metric.ema_ms, 5.0)

    def test_profiler_unknown_metric_is_zero_without_allocating_sample(self):
        from handtracking_perf import PerfProfiler

        profiler = PerfProfiler()
        metric = profiler.metric("missing")

        self.assertEqual(metric.samples, 0)
        self.assertEqual(metric.last_ms, 0.0)
        self.assertEqual(metric.ema_ms, 0.0)

    def test_hot_result_objects_are_slotted(self):
        from handtracking_flow import FlowDispatchResult, LKMotion
        from handtracking_frame import FrameProcessResult
        from handtracking_processing import HandFrameAnalysis, HandModeMetrics, SnapUpdate

        for cls in (
            FlowDispatchResult, LKMotion, FrameProcessResult,
            HandFrameAnalysis, HandModeMetrics, SnapUpdate,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(hasattr(cls, "__slots__"))
                self.assertNotIn("__dict__", cls.__slots__)

    def test_mediapipe_scheduler_adapts_to_inference_cycle_without_starving(self):
        from handtracking_perf import MediaPipeSubmitScheduler

        scheduler = MediaPipeSubmitScheduler(min_fps=20.0, cycle_fraction=0.5)
        self.assertTrue(scheduler.should_submit(1.000, cycle_ms=0.0, target_fps=60))
        self.assertFalse(scheduler.should_submit(1.010, cycle_ms=80.0, target_fps=60))
        self.assertTrue(scheduler.should_submit(1.041, cycle_ms=80.0, target_fps=60))

        # Even a very slow inference cycle cannot reduce submissions below min_fps.
        self.assertFalse(scheduler.should_submit(1.070, cycle_ms=500.0, target_fps=60))
        self.assertTrue(scheduler.should_submit(1.092, cycle_ms=500.0, target_fps=60))

    def test_mediapipe_scheduler_preserves_average_rate_on_60hz_camera(self):
        from handtracking_perf import MediaPipeSubmitScheduler

        fast = MediaPipeSubmitScheduler(min_fps=20.0, cycle_fraction=0.5)
        fast_count = sum(
            fast.should_submit(i / 60.0, cycle_ms=0.0, target_fps=60)
            for i in range(60)
        )
        self.assertEqual(fast_count, 60)

        adaptive = MediaPipeSubmitScheduler(min_fps=20.0, cycle_fraction=0.5)
        adaptive_count = sum(
            adaptive.should_submit(i / 60.0, cycle_ms=40.0, target_fps=60)
            for i in range(60)
        )
        self.assertGreaterEqual(adaptive_count, 47)
        self.assertLessEqual(adaptive_count, 52)


if __name__ == "__main__":
    unittest.main()
