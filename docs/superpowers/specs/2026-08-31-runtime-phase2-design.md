# Runtime Phase 2 Design

## Goal

Ridurre `handtracking_runtime.py` e in particolare `_run_impl()` separando i blocchi di processing che sono già concettualmente indipendenti, senza modificare soglie, gesture o output osservabile.

## Boundaries

- `handtracking_flow.py`: optical-flow per-frame, validazione LK e aggiornamento del moto filtrato.
- `handtracking_hud.py`: costruzione testo/status e rendering diagnostico dell'HUD, usando gli adapter di rendering già esistenti.
- `handtracking_runtime.py`: resta responsabile di camera lifecycle, MediaPipe orchestration e sequenza dei sottosistemi.
- `handtracking_engine.py`: continua a possedere esclusivamente la priorità delle modalità gesture.
- `handtracking_state.py`: continua a possedere lo stato resettable; nessuna duplicazione di stato nei moduli nuovi.

## Constraints

- Nessuna nuova gesture o modifica intenzionale di soglie.
- Nessun accesso Win32 diretto fuori da `handtracking_windows.py`.
- Nessun ownership MediaPipe fuori da `handtracking_mediapipe.py`.
- I moduli estratti devono essere testabili senza webcam reale.
- Il fail-safe MediaPipe stale deve continuare a fermare l'output prima che l'optical flow possa agire da solo.
- Python 3.12+ e dipendenze esistenti soltanto.

## Success criteria

- `_run_impl()` sotto 800 righe, oppure motivazione documentata se un blocco non può essere estratto senza aumentare il rischio.
- Suite completa verde, `py_compile`, `pip check`, `git diff --check`.
- Smoke reale del modello MediaPipe e del worker.
- Nessun import runtime morto o nuovo stato duplicato rilevato dall'audit AST.
