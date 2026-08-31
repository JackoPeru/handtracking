# Hand Tracking

Applicazione Windows per controllare mouse e gesture tramite webcam, MediaPipe e optical flow OpenCV.

Avvio: `Avvia Hand Tracking.bat`.

Al primo avvio il launcher crea automaticamente `.venv`, aggiorna `pip` e installa le dipendenze da `requirements.txt`. Il modello `hand_landmarker.task` e' incluso nel repository e viene risolto rispetto alla cartella del progetto, quindi l'app puo' essere avviata anche da una working directory diversa.

Struttura principale:

- `test.py`: entry-point protetto da `__main__`.
- `handtracking_runtime.py`: camera, MediaPipe, optical flow e gesture runtime.
- `handtracking_core.py`: logica pura e testabile di priorita', timing e tracking della mano.
- `handtracking_mediapipe.py`: worker di inferenza che possiede il ciclo di vita del `HandLandmarker`.
- `tests/`: regressioni automatiche che non richiedono la webcam.
- `snapshots/`: vecchie versioni manuali conservate solo come riferimento.

Test logici senza webcam: `python -m unittest discover -s tests -v`.

La modalita' a due mani implementa lo zoom. La vecchia indicazione di rotazione e' stata rimossa perche' non esiste una scorciatoia di rotazione universale affidabile tra le applicazioni Windows.
