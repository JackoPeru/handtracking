import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_are_immutable_and_derive_legacy_values(self):
        from handtracking_settings import RuntimeSettings

        settings = RuntimeSettings()

        self.assertEqual(settings.camera_index, 0)
        self.assertEqual(settings.profile, "standard")
        self.assertEqual(settings.sensitivity, 1.0)
        self.assertAlmostEqual(settings.pinch_on, 0.46)
        self.assertAlmostEqual(settings.pinch_off, 0.62)
        self.assertAlmostEqual(settings.move_gain, 1.65)
        self.assertAlmostEqual(settings.pinch_release_brake, 0.57)
        with self.assertRaises(FrozenInstanceError):
            settings.sensitivity = 2.0

    def test_profile_scales_move_gain(self):
        from handtracking_settings import RuntimeSettings

        self.assertAlmostEqual(
            RuntimeSettings(profile="precisione").move_gain,
            1.65 * 0.65,
        )
        self.assertAlmostEqual(
            RuntimeSettings(profile="rapidita", sensitivity=2.0).move_gain,
            1.65 * 2.0 * 1.35,
        )

    def test_pinch_release_brake_keeps_legacy_fraction(self):
        from handtracking_settings import RuntimeSettings

        settings = RuntimeSettings(pinch_on=0.20, pinch_off=0.80)

        self.assertAlmostEqual(settings.pinch_release_brake, 0.20 + 0.60 * 0.6875)

    def test_load_settings_reads_real_json_and_profile_override(self):
        from handtracking_settings import load_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_index": 4,
                        "profile": "standard",
                        "sensitivity": 1.5,
                        "pinch_on": 0.30,
                        "pinch_off": 0.90,
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(path, profile="precisione")

        self.assertEqual(settings.camera_index, 4)
        self.assertEqual(settings.profile, "precisione")
        self.assertEqual(settings.sensitivity, 1.5)
        self.assertEqual(settings.pinch_on, 0.30)
        self.assertEqual(settings.pinch_off, 0.90)

    def test_load_settings_none_returns_defaults(self):
        from handtracking_settings import RuntimeSettings, load_settings

        self.assertEqual(load_settings(), RuntimeSettings())
        self.assertEqual(load_settings(profile="rapidita").profile, "rapidita")

    def test_load_settings_reports_missing_explicit_file(self):
        from handtracking_settings import load_settings

        with self.assertRaisesRegex(FileNotFoundError, "File impostazioni non trovato"):
            load_settings(Path("does-not-exist-handtracking-settings.json"))

    def test_load_settings_rejects_unknown_keys_and_profiles(self):
        from handtracking_settings import load_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text('{"unknown": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Chiavi.*non riconosciute"):
                load_settings(path)

            path.write_text('{"profile": "turbo"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Profilo non riconosciuto"):
                load_settings(path)

        with self.assertRaisesRegex(ValueError, "Profilo non riconosciuto"):
            load_settings(profile="turbo")

    def test_load_settings_rejects_wrong_types_and_booleans(self):
        from handtracking_settings import load_settings

        cases = (
            '{"camera_index": true}',
            '{"camera_index": 1.5}',
            '{"sensitivity": false}',
            '{"profile": 1}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for content in cases:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content):
                    with self.assertRaisesRegex(ValueError, "Tipo non valido"):
                        load_settings(path)

    def test_load_settings_rejects_nonfinite_and_out_of_range_values(self):
        from handtracking_settings import load_settings

        cases = (
            '{"sensitivity": 0.1}',
            '{"sensitivity": 3.1}',
            '{"camera_index": -1}',
            '{"camera_index": 17}',
            '{"pinch_on": 0.01}',
            '{"pinch_off": 1.6}',
            '{"sensitivity": NaN}',
            '{"sensitivity": Infinity}',
            '{"sensitivity": ' + '9' * 400 + '}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for content in cases:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content):
                    with self.assertRaisesRegex(ValueError, "Valore"):
                        load_settings(path)

    def test_runtime_settings_rejects_huge_integer_as_validation_error(self):
        from handtracking_settings import RuntimeSettings

        with self.assertRaisesRegex(ValueError, "Valore"):
            RuntimeSettings(sensitivity=int("9" * 400))

    def test_load_settings_rejects_reversed_pinch_thresholds(self):
        from handtracking_settings import load_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                '{"pinch_on": 0.80, "pinch_off": 0.20}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pinch_on.*pinch_off"):
                load_settings(path)

    def test_load_settings_rejects_duplicate_json_keys(self):
        from handtracking_settings import load_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                '{"sensitivity": 1.0, "sensitivity": 1.5}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicata"):
                load_settings(path)

    def test_load_settings_rejects_oversize_file(self):
        from handtracking_settings import MAX_SETTINGS_BYTES, load_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_bytes(b" " * (MAX_SETTINGS_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "troppo grande"):
                load_settings(path)

    def test_deep_json_is_a_validation_error_before_runtime(self):
        from handtracking_settings import load_settings
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("[" * 30000 + "]" * 30000, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON"):
                load_settings(path)


if __name__ == "__main__":
    unittest.main()
