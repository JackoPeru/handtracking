# Hand Tracking

Applicazione Windows per controllare mouse e gesture tramite webcam, MediaPipe e optical flow OpenCV.

Avvio: `Avvia Hand Tracking.bat`.

Al primo avvio il launcher crea automaticamente `.venv`, aggiorna `pip` e installa le dipendenze da `requirements.txt`. Il modello `hand_landmarker.task` e' incluso nel repository e viene risolto rispetto alla cartella del progetto, quindi l'app puo' essere avviata anche da una working directory diversa.

Struttura principale:

- `main.py`: entry-point principale protetto da `__main__`.
- `test.py`: shim di compatibilita' che delega a `main.py`.
- `handtracking_runtime.py`: orchestrazione camera, MediaPipe, optical flow e stato runtime.
- `handtracking_config.py`: costanti e soglie senza side effect.
- `handtracking_gestures.py`: geometria e classificatori gesture puri.
- `handtracking_engine.py`: priorita' e risoluzione della modalita' gesture.
- `handtracking_state.py`: state object e reset centralizzati.
- `handtracking_windows.py`: input Windows, volume e cursore asincrono.
- `handtracking_render.py`: rendering OpenCV e overlay.
- `handtracking_core.py`: logica pura e testabile di priorita', timing e tracking della mano.
- `handtracking_mediapipe.py`: worker di inferenza che possiede il ciclo di vita del `HandLandmarker`.
- `tests/`: regressioni automatiche che non richiedono la webcam.
- `snapshots/`: vecchie versioni manuali conservate solo come riferimento.

Test logici senza webcam: `python -m unittest discover -s tests -v`.

La modalita' a due mani implementa lo zoom. La vecchia indicazione di rotazione e' stata rimossa perche' non esiste una scorciatoia di rotazione universale affidabile tra le applicazioni Windows.
