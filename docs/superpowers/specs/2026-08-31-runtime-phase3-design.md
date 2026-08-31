# Runtime Phase 3 Design

## Goal

Ridurre ulteriormente i file grandi senza cambiare gesture, soglie o comportamento osservabile. `handtracking_runtime.py` deve diventare un orchestratore di circa 350-500 righe; `handtracking_processing.py` deve perdere la macchina Spock e restare focalizzato sull'analisi MediaPipe semipura.

## Boundaries

- `handtracking_camera.py`: apertura/configurazione webcam, metadata camera, preparazione frame e finestra OpenCV.
- `handtracking_session.py`: ownership e inizializzazione della sessione runtime, worker MediaPipe, cursore, state object e metriche persistenti.
- `handtracking_tracking.py`: fail-safe per MediaPipe stale, perdita mano e reset coordinati tra gesture/flow/cursore.
- `handtracking_volume.py`: macchina a stati volume candidate/lock/release, inclusi freeze e callback di lettura/scrittura volume.
- `handtracking_scroll.py`: arm/release dello scroll MediaPipe. Il movimento wheel camera-rate resta in `handtracking_flow.py`.
- `handtracking_spock.py`: macchina a stati Spock oggi contenuta in `handtracking_processing.py`.
- `handtracking_processing.py`: analisi mano, handedness, clutch fist, swipe, EMA e precision snap.
- `handtracking_runtime.py`: loop, ordine delle fasi, wiring tra moduli e rendering finale.

## Behavioral constraints

- Nessuna soglia in `handtracking_config.py` cambia.
- La priorita' gesture resta in `handtracking_engine.py`.
- La perdita di MediaPipe fresco deve disabilitare output prima che LK possa continuare da solo.
- La webcam deve continuare a richiedere MJPG, dimensioni/FPS correnti e buffer 1.
- Il worker MediaPipe deve continuare a possedere il `HandLandmarker`.
- Pointer, radial, two-hand e flow restano nei moduli esistenti.
- Nessun test deve richiedere webcam reale o produrre input mouse/tastiera reale.

## Target structure

`handtracking_runtime.py` coordina una `RuntimeSession` e un `CameraRuntime`, consuma un frame per iterazione, applica tracking fail-safe, optical flow, nuovo risultato MediaPipe, gesture handlers, snap, rendering e uscita.

Il runtime non deve piu' contenere setup dettagliato camera/worker/state, reset stale/no-hand, macchina volume/scroll o macchina Spock.

## Success criteria

- `handtracking_runtime.py` tra 350 e 500 righe, salvo motivazione tecnica documentata.
- `_run_impl()` sotto 400 righe.
- `handtracking_processing.py` sotto 300 righe.
- Suite completa verde con almeno gli 82 test esistenti piu' i nuovi regression test.
- `py_compile`, `pip check`, `git diff --check`, secret scan e audit AST puliti.
- Smoke reale di caricamento modello e worker MediaPipe.
- Nessuna dipendenza nuova.
