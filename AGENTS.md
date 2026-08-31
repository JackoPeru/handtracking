# Development Notes

- Keep gesture behavior deterministic and testable without a webcam where possible.
- Add regression tests before changing gesture-state behavior.
- Do not reintroduce executable snapshot files under names matched by `test_*.py`.
- Verify with `python -m unittest discover -s tests -v` and `python -m py_compile test.py handtracking_core.py handtracking_runtime.py`.
