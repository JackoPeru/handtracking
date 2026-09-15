import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "Avvia Hand Tracking.bat"


@unittest.skipUnless(os.name == "nt", "Windows launcher contract")
class LauncherBehaviorTests(unittest.TestCase):
    def _fixture(self, *, create_venv=True, import_stubs=False):
        temp_dir = tempfile.TemporaryDirectory(prefix="handtracking launcher ")
        try:
            root = Path(temp_dir.name)
            if create_venv:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(root / ".venv"), "--without-pip"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            (root / "launcher.bat").write_text(
                LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (root / "requirements.lock").write_text("fixture\n", encoding="utf-8")
            (root / "pip.py").write_text(
                "import json, os\n"
                "from pathlib import Path\n"
                "events = Path('launcher-events.jsonl')\n"
                "with events.open('a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps({'name': 'pip', 'args': __import__('sys').argv[1:]}) + '\\n')\n"
                "raise SystemExit(int(os.environ.get('FAKE_PIP_EXIT', '0')))\n",
                encoding="utf-8",
            )
            (root / "main.py").write_text(
                "from pathlib import Path\n"
                "with Path('launcher-events.jsonl').open('a', encoding='utf-8') as stream:\n"
                "    stream.write('{\"name\": \"main\"}\\n')\n",
                encoding="utf-8",
            )
            if not create_venv:
                (root / "bin").mkdir()
                (root / "bin" / "py.cmd").write_text(
                    "@echo off\n"
                    "if /i \"%~1\"==\"-3.12\" shift\n"
                    "if /i \"%~1\"==\"-c\" exit /b 0\n"
                    "if /i \"%~1\"==\"-m\" if /i \"%~2\"==\"venv\" goto create_venv\n"
                    "exit /b 1\n"
                    ":create_venv\n"
                    "\"%FAKE_PYTHON%\" -m venv \"%~3\" --without-pip\n"
                    "exit /b %errorlevel%\n",
                    encoding="utf-8",
                )
            if import_stubs:
                for module in ("cv2", "mediapipe", "numpy", "pycaw", "comtypes"):
                    (root / f"{module}.py").write_text("", encoding="utf-8")
            return temp_dir, root
        except Exception:
            temp_dir.cleanup()
            raise

    def _run(self, root, *, env=None, fake_python=None):
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        if fake_python:
            process_env["FAKE_PYTHON"] = str(fake_python)
            process_env["PATH"] = str(root / "bin") + os.pathsep + process_env["PATH"]
        return subprocess.run(
            ["cmd.exe", "/d", "/c", "call", "launcher.bat"],
            cwd=root,
            env=process_env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _events(self, root):
        return [
            json.loads(line)
            for line in (root / "launcher-events.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_existing_environment_reconciles_hashed_lock_before_main(self):
        temp_dir, root = self._fixture(import_stubs=True)
        self.addCleanup(temp_dir.cleanup)

        result = self._run(root)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self._events(root),
            [
                {
                    "name": "pip",
                    "args": [
                        "install",
                        "--require-hashes",
                        "--only-binary=:all:",
                        "-r",
                        "requirements.lock",
                    ],
                },
                {"name": "main"},
            ],
        )

    def test_failed_dependency_reconciliation_stops_before_main(self):
        temp_dir, root = self._fixture()
        self.addCleanup(temp_dir.cleanup)

        result = self._run(root, env={"FAKE_PIP_EXIT": "23"})

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self._events(root),
            [
                {
                    "name": "pip",
                    "args": [
                        "install",
                        "--require-hashes",
                        "--only-binary=:all:",
                        "-r",
                        "requirements.lock",
                    ],
                }
            ],
        )

    def test_missing_environment_creates_312_venv_without_unpinned_upgrade(self):
        temp_dir, root = self._fixture(create_venv=False)
        self.addCleanup(temp_dir.cleanup)
        fake_python = ROOT / ".venv" / "Scripts" / "python.exe"
        if not fake_python.exists():
            fake_python = Path(sys.executable)

        result = self._run(root, fake_python=fake_python)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((root / ".venv" / "Scripts" / "python.exe").exists())
        self.assertEqual(
            self._events(root),
            [
                {
                    "name": "pip",
                    "args": [
                        "install",
                        "--require-hashes",
                        "--only-binary=:all:",
                        "-r",
                        "requirements.lock",
                    ],
                },
                {"name": "main"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
