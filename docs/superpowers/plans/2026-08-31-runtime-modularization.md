# Hand Tracking Runtime Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correggere i bug di lifecycle/stato emersi dall'audit e trasformare `handtracking_runtime.py` in un orchestratore modulare senza cambiare intenzionalmente il comportamento delle gesture.

**Architecture:** Separare configurazione, classificazione gesture, adapter Windows, rendering e stato in moduli focalizzati. Il runtime conserva camera/optical-flow/orchestrazione; MediaPipe mantiene ownership del landmarker con snapshot atomico.

**Tech Stack:** Python 3.12, MediaPipe 1.0.1, OpenCV contrib 5.0.0.93, NumPy 2.5.2, pycaw 20251023, comtypes 1.4.16, unittest.

**Spec:** `docs/superpowers/specs/2026-08-31-runtime-modularization-design.md`

## Global Constraints

- Windows e Python 3.12.
- Nessun cambio arbitrario a soglie/sensibilita' gesture.
- Pugno clutch globale; two-hand zoom-only.
- TDD per ogni bug fix e comportamento nuovo.
- Suite verde dopo ogni estrazione.
- Modifiche tramite `apply_patch`.

---

### Task 1: Lifecycle e fail-safe regressions

**Files:**
- Modify: `tests/test_runtime_smoke.py`
- Modify: `tests/test_mediapipe_worker.py`
- Modify: `handtracking_mediapipe.py`
- Modify: `handtracking_runtime.py`

**Interfaces:**
- Produces: `MediaPipeWorker.snapshot_state() -> dict` con `latest`, `seq`, `input_seq`, `overwrites`, `error_count`, `last_error`, `last_success_at`, `last_result_input_at`, `alive`.

- [ ] **Step 1:** Aggiungere test runtime che inietta un'eccezione nel loop e verifica sempre `capture.release`, `worker.stop/join`, `cursor_stop=True`, `destroyAllWindows`.
- [ ] **Step 2:** Eseguire il test e verificare RED sul cleanup attuale.
- [ ] **Step 3:** Aggiungere test worker-init-failure e snapshot atomico.
- [ ] **Step 4:** Eseguire i test e verificare RED.
- [ ] **Step 5:** Implementare `snapshot_state()` e `try/finally` nel runtime; rilevare worker morto come errore fatale.
- [ ] **Step 6:** Eseguire test mirati e suite completa.
- [ ] **Step 7:** Commit `fix: harden runtime lifecycle`.

### Task 2: Spock stale recovery e priorita' volume candidate

**Files:**
- Modify: `handtracking_core.py`
- Modify: `tests/test_core.py`
- Modify: `handtracking_runtime.py`

**Interfaces:**
- Produces: helper puro per policy Spock stale/rearm.
- Produces: helper centrale `gesture_priority_blocks(...)` o equivalente che considera `volume_candidate`.

- [ ] **Step 1:** Scrivere test che simula Spock latched -> stale -> ritorno ancora Spock e richiede nessun nuovo toggle finche' non avviene release.
- [ ] **Step 2:** Verificare RED.
- [ ] **Step 3:** Scrivere test che volume candidate blocca radial/swipe/pointer.
- [ ] **Step 4:** Verificare RED.
- [ ] **Step 5:** Implementare le policy pure minime e integrarle nel runtime.
- [ ] **Step 6:** Eseguire suite completa.
- [ ] **Step 7:** Commit `fix: preserve gesture priority across tracking gaps`.

### Task 3: Python 3.12 launcher e CI

**Files:**
- Modify: `Avvia Hand Tracking.bat`
- Modify: `tests/test_runtime_contracts.py`
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Launcher: crea `.venv` solo con Python 3.12+; messaggio esplicito se non disponibile.

- [ ] **Step 1:** Aggiungere contract test per `py -3.12`/check versione.
- [ ] **Step 2:** Verificare RED.
- [ ] **Step 3:** Aggiornare launcher e aggiungere workflow Windows/Python 3.12 con unittest, py_compile, pip check.
- [ ] **Step 4:** Verificare test locali e validita' YAML tramite lettura/controllo sintattico disponibile.
- [ ] **Step 5:** Commit `ci: verify hand tracking on windows`.

### Task 4: Estrarre configurazione e gesture pure

**Files:**
- Create: `handtracking_config.py`
- Create: `handtracking_gestures.py`
- Modify: `handtracking_runtime.py`
- Modify/Create: `tests/test_gestures.py`
- Modify: `tests/test_source_contracts.py`

**Interfaces:**
- `handtracking_config.py`: costanti senza side effects.
- `handtracking_gestures.py`: funzioni pure attualmente definite nelle righe iniziali del runtime (geometria/classificazione pose).

- [ ] **Step 1:** Aggiungere test di import senza side effect e parita' per classificatori rappresentativi.
- [ ] **Step 2:** Verificare RED per moduli mancanti.
- [ ] **Step 3:** Spostare costanti/config e funzioni gesture senza modificarne il corpo salvo import.
- [ ] **Step 4:** Aggiornare runtime agli import.
- [ ] **Step 5:** Eseguire suite + py_compile.
- [ ] **Step 6:** Commit `refactor: extract gesture configuration and classifiers`.

### Task 5: Estrarre adapter Windows e rendering

**Files:**
- Create: `handtracking_windows.py`
- Create: `handtracking_render.py`
- Modify: `handtracking_runtime.py`
- Create/Modify: `tests/test_windows_adapter.py`, `tests/test_render.py`

**Interfaces:**
- Windows adapter espone input primitives, cursor controller/worker, volume adapter e azioni browser/radial.
- Render module espone `draw_hand`, `draw_radial_menu`, `draw_two_hand_transform` e helper overlay senza mutare gesture state.

- [ ] **Step 1:** Aggiungere test import-safe/adapter con user32 e audio mockati.
- [ ] **Step 2:** Verificare RED.
- [ ] **Step 3:** Estrarre adapter Windows spostando i side effect dentro inizializzazione esplicita.
- [ ] **Step 4:** Estrarre rendering OpenCV meccanicamente.
- [ ] **Step 5:** Eseguire suite + smoke runtime.
- [ ] **Step 6:** Commit `refactor: extract windows and rendering adapters`.

### Task 6: Introdurre state objects e reset centralizzati

**Files:**
- Create: `handtracking_state.py`
- Create: `tests/test_state.py`
- Modify: `handtracking_runtime.py`

**Interfaces:**
- Dataclass: `PointerState`, `VolumeState`, `ScrollState`, `SwipeState`, `RadialState`, `TwoHandState`, `SpockState`, `FlowState`, ognuna con `reset()`.

- [ ] **Step 1:** Scrivere unit test dei reset con stato non-default -> `reset()` -> default.
- [ ] **Step 2:** Verificare RED.
- [ ] **Step 3:** Implementare dataclass e migrare un gruppo alla volta iniziando da pointer/scroll, poi volume/swipe/radial/two-hand/Spock/flow.
- [ ] **Step 4:** Dopo ogni gruppo eseguire suite completa.
- [ ] **Step 5:** Contare le assegnazioni duplicate residue e rimuovere reset manuali equivalenti.
- [ ] **Step 6:** Commit `refactor: centralize gesture runtime state`.

### Task 7: Estrarre decisioni engine e rinominare entry-point

**Files:**
- Create: `handtracking_engine.py`
- Create: `tests/test_engine.py`
- Create: `main.py`
- Modify: `test.py`
- Modify: `Avvia Hand Tracking.bat`
- Modify: `handtracking_runtime.py`
- Modify: `README.md`, `AGENTS.md`

**Interfaces:**
- Engine contiene priorita'/mode resolution/transizioni pure; nessun OpenCV/Windows.
- `main.py:main()` invoca `handtracking_runtime.run()`.
- `test.py` compatibility shim delega a `main.main()`.

- [ ] **Step 1:** Scrivere test tabellare priorita' engine.
- [ ] **Step 2:** Verificare RED.
- [ ] **Step 3:** Estrarre decisioni pure e aggiornare runtime.
- [ ] **Step 4:** Aggiungere `main.py`, aggiornare launcher e mantenere shim `test.py`.
- [ ] **Step 5:** Aggiornare docs.
- [ ] **Step 6:** Eseguire suite, py_compile, pip check, diff check e smoke.
- [ ] **Step 7:** Misurare dimensione/complessita' finale del runtime e documentarla.
- [ ] **Step 8:** Commit `refactor: make runtime a modular orchestrator`.

### Task 8: Final verification and integration

**Files:** all changed files.

- [ ] **Step 1:** `python -m unittest discover -s tests -q`.
- [ ] **Step 2:** `python -m py_compile` su tutti i moduli applicativi/test pertinenti.
- [ ] **Step 3:** `python -m pip check`.
- [ ] **Step 4:** `git diff --check` e secret scan sui file tracciati.
- [ ] **Step 5:** Test reale caricamento modello MediaPipe da working directory esterna.
- [ ] **Step 6:** Review AST per funzioni/stato morto e complessita' del runtime.
- [ ] **Step 7:** Merge/fast-forward su `main`, push GitHub e verifica hash locale/remoto.
