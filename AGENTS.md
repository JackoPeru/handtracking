# Development Notes

- Keep gesture behavior deterministic and testable without a webcam where possible.
- Add regression tests before changing gesture-state behavior.
- Do not reintroduce executable snapshot files under names matched by `test_*.py`.
- Keep MediaPipe ownership inside `handtracking_mediapipe.py`; the main runtime must never close a landmarker used by another thread.
- Any loss of fresh MediaPipe results must disable cursor/gesture output before optical flow can continue on its own.
- Keep gesture mode priority in `handtracking_engine.py`; do not duplicate it in the runtime or classifiers.
- Keep resettable gesture state in `handtracking_state.py`; prefer tested reset helpers over repeated assignment blocks.
- Keep LK measurement/camera-rate motion dispatch in `handtracking_flow.py`, focused gesture transitions in `handtracking_handlers.py`, semi-pure MediaPipe processing in `handtracking_processing.py`, and diagnostics in `handtracking_hud.py`.
- Keep webcam ownership/preprocessing in `handtracking_camera.py`, runtime resource ownership in `handtracking_session.py`, tracking-loss fail-safes in `handtracking_tracking.py`, MediaPipe packet orchestration in `handtracking_frame.py`, gesture-mode coordination in `handtracking_modes.py`, and volume/scroll/Spock state machines in their dedicated modules.
- `handtracking_runtime.py` must remain an orchestrator; do not move extracted optical-flow, HUD, pointer, radial, two-hand, Spock or hand-analysis logic back into `_run_impl()`.
- Keep `RuntimeSession` as the single scalar source of truth. Do not mirror session state into a second set of loop-local scalar variables.
- Gate LK before calling OpenCV and keep camera detection preprocessing conditional on submit/flow/new-packet demand.
- Keep MediaPipe worker wakeup event-driven; do not reintroduce millisecond polling loops.
- Preserve the lazy `HandFeatures` cache when adding classifiers. Reuse cached joint angles instead of recomputing identical `acos` geometry in the same MediaPipe result.
- HUD caching is allowed because it benchmarks faster. Do not cache the full-frame skeleton overlay unless a local benchmark proves it is faster than direct drawing.
- Run `python -m benchmarks.hotpath_benchmark` for hot-path performance changes and report before/after numbers; do not accept a performance refactor based only on code appearance.
- Verify with `python -m unittest discover -s tests -v` and compile every root Python module before merging.
