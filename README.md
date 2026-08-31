# Hand Tracking

Applicazione Windows per controllare mouse e gesture tramite webcam, MediaPipe e optical flow OpenCV.

Avvio: `Avvia Hand Tracking.bat`.

Struttura principale:

- `test.py`: entry-point protetto da `__main__`.
- `handtracking_runtime.py`: camera, MediaPipe, optical flow e gesture runtime.
- `handtracking_core.py`: logica pura e testabile di priorita', timing e tracking della mano.
- `tests/`: regressioni automatiche che non richiedono la webcam.
- `snapshots/`: vecchie versioni manuali conservate solo come riferimento.

Test logici senza webcam: `python -m unittest discover -s tests -v`.
