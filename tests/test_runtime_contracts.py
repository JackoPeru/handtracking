import unittest
from pathlib import Path
import re


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

    def test_runtime_has_no_direct_launcher(self):
        runtime = (ROOT / "handtracking_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if __name__ == "__main__":', runtime)

    def test_public_repo_has_dependency_manifest(self):
        requirements = ROOT / "requirements.txt"
        self.assertTrue(requirements.exists())
        text = requirements.read_text(encoding="utf-8")
        for package in ("mediapipe", "opencv-contrib-python", "numpy", "pycaw", "comtypes"):
            self.assertIn(package, text)

    def test_hashed_lock_matches_direct_requirements(self):
        requirements = {}
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
            self.assertIsNotNone(match, line)
            name, version = match.groups()
            requirements[re.sub(r"[-_.]+", "-", name).lower()] = version

        lock = {}
        for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(
                r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})",
                line,
            )
            self.assertIsNotNone(match, line)
            name, version, _ = match.groups()
            lock[re.sub(r"[-_.]+", "-", name).lower()] = version

        for name, version in requirements.items():
            self.assertEqual(lock.get(name), version, name)

    def test_launcher_can_bootstrap_virtualenv(self):
        launcher = (ROOT / "Avvia Hand Tracking.bat").read_text(encoding="utf-8")
        self.assertIn("requirements.lock", launcher)
        self.assertIn("-m venv", launcher)
        self.assertIn("--require-hashes", launcher)
        self.assertIn("--only-binary=:all:", launcher)

    def test_launcher_requires_python_312_for_new_environment(self):
        launcher = (ROOT / "Avvia Hand Tracking.bat").read_text(encoding="utf-8")
        self.assertIn("py -3.12", launcher)
        self.assertIn("sys.version_info[:2] == (3, 12)", launcher)
        self.assertIn("struct.calcsize('P') * 8 == 64", launcher)
        self.assertIn("platform.python_implementation() == 'CPython'", launcher)

    def test_github_actions_runs_windows_python_312_checks(self):
        workflow = ROOT / ".github" / "workflows" / "tests.yml"
        self.assertTrue(workflow.exists())
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("windows-latest", text)
        self.assertIn("python-version: '3.12'", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("contents: read", text)
        self.assertRegex(text, r"actions/checkout@[0-9a-f]{40}")
        self.assertRegex(text, r"actions/setup-python@[0-9a-f]{40}")
        self.assertIn("--require-hashes", text)
        self.assertIn("--only-binary=:all:", text)
        self.assertIn("unittest discover -s tests", text)
        self.assertIn("py_compile", text)
        self.assertIn("pip check", text)

    def test_ci_install_step_uses_yaml_safe_block_scalar(self):
        workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
            encoding="utf-8"
        )
        start = workflow.index("      - name: Install dependencies")
        end = workflow.find("\n      - name:", start + 1)
        install_step = workflow[start:] if end < 0 else workflow[start:end]

        self.assertRegex(install_step, r"(?m)^        run:\s*\|\s*$")
        self.assertRegex(
            install_step,
            r"(?m)^\s+python -m pip install --require-hashes "
            r"--only-binary=:all: -r requirements\.lock\s*$",
        )
        self.assertNotRegex(
            install_step,
            r"(?m)^\s+run:\s+(?!\|)[^\r\n]*:\s",
        )


if __name__ == "__main__":
    unittest.main()
