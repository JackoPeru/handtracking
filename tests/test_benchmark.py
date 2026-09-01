import unittest


class HotpathBenchmarkTests(unittest.TestCase):
    def test_benchmark_returns_expected_metrics_with_small_iteration_count(self):
        from benchmarks.hotpath_benchmark import run_benchmarks

        results = run_benchmarks(iterations=2, lk_iterations=2, render_iterations=2)
        for key in (
            "geometry_raw_us",
            "geometry_cached_us",
            "lk_ms",
            "preprocess_ms",
            "hud_direct_ms",
            "hud_cached_ms",
            "overlay_direct_ms",
            "overlay_cached_ms",
        ):
            with self.subTest(key=key):
                self.assertIn(key, results)
                self.assertGreaterEqual(results[key], 0.0)


if __name__ == "__main__":
    unittest.main()
