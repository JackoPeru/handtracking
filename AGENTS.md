# Development Notes

- Keep gesture behavior deterministic and testable without a webcam where possible.
- Add regression tests before changing gesture-state behavior.
- Do not reintroduce executable snapshot files under names matched by `test_*.py`.
- Keep MediaPipe ownership inside `handtracking_mediapipe.py`; the main runtime must never close a landmarker used by another thread.
- Any loss of fresh MediaPipe results must disable cursor/gesture output before optical flow can continue on its own.
- Verify with `python -m unittest discover -s tests -v` and `python -m py_compile test.py handtracking_core.py handtracking_mediapipe.py handtracking_runtime.py`.
