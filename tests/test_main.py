import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

import main


class MainTests(unittest.TestCase):
    def test_profile_cli_reaches_runtime_without_changing_defaults(self):
        with mock.patch.object(main, "run") as run:
            try:
                main.main(["--profile", "precisione"])
            except TypeError as exc:
                self.fail("CLI not implemented: " + str(exc))
        self.assertEqual(run.call_args.kwargs["settings"].profile, "precisione")

    def test_explicit_missing_config_fails_before_camera(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(main, "run") as run:
            try:
                with self.assertRaises(SystemExit) as result:
                    main.main(["--config", str(Path(directory) / "missing.json")])
            except TypeError as exc:
                self.fail("CLI not implemented: " + str(exc))
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()

    def test_record_and_replay_are_mutually_exclusive(self):
        with mock.patch.object(main, "run") as run:
            try:
                with self.assertRaises(SystemExit) as result:
                    main.main(["--record", "capture.jsonl", "--replay", "capture.jsonl"])
            except TypeError as exc:
                self.fail("CLI not implemented: " + str(exc))
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()

    def test_default_local_config_is_loaded_with_real_validation(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(main, "run") as run, \
                mock.patch.object(main, "__file__", str(Path(directory) / "main.py")):
            config = Path(directory) / "handtracking.json"
            config.write_text('{"camera_index": 2, "sensitivity": 1.5}', encoding="utf-8")
            main.main([])
        settings = run.call_args.kwargs["settings"]
        self.assertEqual(settings.camera_index, 2)
        self.assertEqual(settings.sensitivity, 1.5)

    def test_invalid_local_config_and_record_bound_fail_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(main, "run") as run, \
                mock.patch.object(main, "__file__", str(Path(directory) / "main.py")), \
                contextlib.redirect_stderr(io.StringIO()):
            config = Path(directory) / "handtracking.json"
            config.write_text('{"sensitivity": NaN}', encoding="utf-8")
            for args in ([], ["--record-max-frames", "0"],
                         ["--record-max-frames", "18001"]):
                with self.subTest(args=args), self.assertRaises(SystemExit) as result:
                    main.main(args)
                self.assertEqual(result.exception.code, 2)
        run.assert_not_called()

    def test_record_options_reach_runtime(self):
        with mock.patch.object(main, "run") as run:
            main.main(["--record", "traces/new.jsonl", "--record-max-frames", "120"])
        self.assertEqual(run.call_args.kwargs["record_path"], Path("traces/new.jsonl"))
        self.assertEqual(run.call_args.kwargs["record_max_frames"], 120)

    def test_failed_replay_has_nonzero_exit_and_never_starts_runtime(self):
        with mock.patch("handtracking_trace.replay_trace", return_value={
                "frames": 2, "source": "synthetic", "verified": False,
                "mismatches": ["command mismatch"],
        }), mock.patch.object(main, "run") as run, \
                contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main.main(["--replay", "trace.jsonl"])
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
