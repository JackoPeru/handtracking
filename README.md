# Hand Tracking

Applicazione Windows per controllare mouse e gesture tramite webcam, MediaPipe e optical flow OpenCV.

Avvio: `Avvia Hand Tracking.bat`.

Al primo avvio il launcher crea automaticamente `.venv`, aggiorna `pip` e installa le dipendenze da `requirements.txt`. Il modello `hand_landmarker.task` e' incluso nel repository e viene risolto rispetto alla cartella del progetto, quindi l'app puo' essere avviata anche da una working directory diversa.

Struttura principale:

- `main.py`: entry-point principale protetto da `__main__`.
- `test.py`: shim di compatibilita' che delega a `main.py`.
- `handtracking_runtime.py`: loop principale e wiring tra sessione, flow, rendering e HUD.
- `handtracking_camera.py`: apertura/configurazione webcam, preprocessing frame e finestra OpenCV.
- `handtracking_session.py`: ownership di camera, worker, cursore, state object e metriche persistenti.
- `handtracking_tracking.py`: fail-safe MediaPipe stale e perdita tracking/mano.
- `handtracking_frame.py`: orchestrazione di ogni nuovo risultato MediaPipe.
- `handtracking_modes.py`: coordinamento delle transizioni gesture e fallback cursore per un risultato MediaPipe.
- `handtracking_volume.py`: macchina a stati volume candidate/lock/release.
- `handtracking_scroll.py`: arm/release dello scroll MediaPipe.
- `handtracking_spock.py`: macchina a stati Spock e release senza mano.
- `handtracking_config.py`: costanti e soglie senza side effect.
- `handtracking_gestures.py`: geometria e classificatori gesture puri.
- `handtracking_engine.py`: priorita' e risoluzione della modalita' gesture.
- `handtracking_flow.py`: optical flow LK, filtro del movimento e dispatch camera-rate di swipe/scroll/puntatore.
- `handtracking_handlers.py`: transizioni focalizzate di pointer, two-hand e menu radiale.
- `handtracking_processing.py`: processing semipuro dei risultati MediaPipe, metriche mano, swipe, EMA e precision snap.
- `handtracking_state.py`: state object e reset centralizzati.
- `handtracking_windows.py`: input Windows, volume e cursore asincrono.
- `handtracking_render.py`: rendering OpenCV e overlay.
- `handtracking_hud.py`: stato testuale, diagnostica, LED e barra Spock.
- `handtracking_core.py`: logica pura e testabile di priorita', timing e tracking della mano.
- `handtracking_mediapipe.py`: worker di inferenza che possiede il ciclo di vita del `HandLandmarker`.
- `tests/`: regressioni automatiche che non richiedono la webcam.
- `snapshots/`: eventuali vecchie versioni locali sono ignorate da Git e non fanno parte del repository pubblico.

Test logici senza webcam: `python -m unittest discover -s tests -v`.

La modalita' a due mani implementa lo zoom. La vecchia indicazione di rotazione e' stata rimossa perche' non esiste una scorciatoia di rotazione universale affidabile tra le applicazioni Windows.
