import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_launcher_targets_main_application(self):
        launcher = (ROOT / "Avvia Hand Tracking.bat").read_text(encoding="utf-8")
        self.assertIn("main.py", launcher)

    def test_main_entrypoint_is_guarded(self):
        entrypoint = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', entrypoint)
        self.assertIn("main()", entrypoint)

    def test_legacy_test_entrypoint_delegates_to_main(self):
        entrypoint = (ROOT / "test.py").read_text(encoding="utf-8")
        self.assertIn("from main import main", entrypoint)
        self.assertNotIn("runpy", entrypoint)

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

    def test_launcher_requires_python_312_for_new_environment(self):
        launcher = (ROOT / "Avvia Hand Tracking.bat").read_text(encoding="utf-8")
        self.assertIn("py -3.12", launcher)
        self.assertIn("sys.version_info >= (3, 12)", launcher)

    def test_github_actions_runs_windows_python_312_checks(self):
        workflow = ROOT / ".github" / "workflows" / "tests.yml"
        self.assertTrue(workflow.exists())
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("windows-latest", text)
        self.assertIn("python-version: '3.12'", text)
        self.assertIn("unittest discover -s tests", text)
        self.assertIn("py_compile", text)
        self.assertIn("pip check", text)


if __name__ == "__main__":
    unittest.main()
