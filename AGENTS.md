# Development Notes

- Keep gesture behavior deterministic and testable without a webcam where possible.
- Add regression tests before changing gesture-state behavior.
- Do not reintroduce executable snapshot files under names matched by `test_*.py`.
- Keep MediaPipe ownership inside `handtracking_mediapipe.py`; the main runtime must never close a landmarker used by another thread.
- Any loss of fresh MediaPipe results must disable cursor/gesture output before optical flow can continue on its own.
- Keep gesture mode priority in `handtracking_engine.py`; do not duplicate it in the runtime or classifiers.
- Keep resettable gesture state in `handtracking_state.py`; prefer tested reset helpers over repeated assignment blocks.
- Keep LK measurement/camera-rate motion dispatch in `handtracking_flow.py`, focused gesture transitions in `handtracking_handlers.py`, semi-pure MediaPipe processing in `handtracking_processing.py`, and diagnostics in `handtracking_hud.py`.
- `handtracking_runtime.py` must remain an orchestrator; do not move extracted optical-flow, HUD, pointer, radial, two-hand, Spock or hand-analysis logic back into `_run_impl()`.
- Verify with `python -m unittest discover -s tests -v` and compile every root Python module before merging.
