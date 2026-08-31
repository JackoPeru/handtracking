import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_launcher_targets_main_application(self):
        launcher = (ROOT / "Avvia Hand Tracking.bat").read_text(encoding="utf-8")
        self.assertIn("test.py", launcher)

    def test_main_entrypoint_is_guarded(self):
        entrypoint = (ROOT / "test.py").read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', entrypoint)
        self.assertIn("main()", entrypoint)

    def test_runtime_is_separate_from_entrypoint(self):
        self.assertTrue((ROOT / "handtracking_runtime.py").exists())

    def test_public_repo_has_dependency_manifest(self):
        requirements = ROOT / "requirements.txt"
        self.assertTrue(requirements.exists())
        text = requirements.read_text(encoding="utf-8")
        for package in ("mediapipe", "opencv-contrib-python", "numpy", "pycaw", "comtypes"):
            self.assertIn(package, text)

    def test_launcher_can_bootstrap_virtualenv(self):
        launcher = (ROOT / "Avvia Hand Tracking.bat").read_text(encoding="utf-8")
        self.assertIn("requirements.txt", launcher)
        self.assertIn("-m venv", launcher)
        self.assertIn("import cv2, mediapipe, numpy, pycaw, comtypes", launcher)


if __name__ == "__main__":
    unittest.main()
