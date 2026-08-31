# Runtime Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendere `handtracking_runtime.py` un orchestratore da 350-500 righe separando camera/sessione, tracking lifecycle, volume/scroll e Spock senza modificare il comportamento.

**Architecture:** Introdurre moduli focalizzati con dipendenze esplicite e callback per i side effect. Il runtime conserva solo l'ordine delle fasi e il wiring; gli state object esistenti restano la fonte di verita' mutabile.

**Tech Stack:** Python 3.12, OpenCV, MediaPipe, NumPy, pycaw/comtypes, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-runtime-phase3-design.md`

## Global Constraints

- Non cambiare soglie o gesture.
- Non aggiungere dipendenze.
- Non eseguire input Windows reali nei test.
- Conservare fail-safe stale e ownership MediaPipe esistenti.
- Ogni estrazione usa RED -> GREEN -> suite completa -> commit.

---

### Task 1: Camera runtime

**Files:**
- Create: `handtracking_camera.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_camera.py`

**Interfaces:**
- Produces: `CameraRuntime.open(cv2_module=cv2)`, `read_prepared()`, metadata `reported_fps/reported_w/reported_h/codec/target_fps`, `show(frame) -> bool`.

- [ ] Scrivere test failing per fallback MSMF, proprieta' camera e preparazione frame.
- [ ] Verificare RED con `..\..\.venv\Scripts\python.exe -m unittest tests.test_camera -v`.
- [ ] Implementare `CameraRuntime` copiando le impostazioni correnti e senza cambiare costanti.
- [ ] Sostituire setup/read/show camera nel runtime.
- [ ] Eseguire test mirati e suite completa.
- [ ] Commit `refactor: extract camera runtime`.

### Task 2: Session ownership and initialization

**Files:**
- Create: `handtracking_session.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `RuntimeSession.create(...)`, `close()`, gesture state objects, cursor/worker, debug/timing fields e helper `build_mediapipe_image`.

- [ ] Scrivere test failing per inizializzazione state/cursor/worker e cleanup idempotente.
- [ ] Verificare RED.
- [ ] Implementare factory con dipendenze iniettabili e spostare `RuntimeCleanup`/setup dal runtime.
- [ ] Aggiornare runtime a usare `RuntimeSession`.
- [ ] Suite completa e commit `refactor: extract runtime session`.

### Task 3: Tracking lifecycle

**Files:**
- Create: `handtracking_tracking.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Produces: `apply_stale_fail_safe(session)`, `handle_missing_hands(session, now)`, `expire_lost_flow(session, now)`.

- [ ] Scrivere test failing per stale reset, pointer tracking-loss freeze e full tracking-loss reset.
- [ ] Verificare RED.
- [ ] Estrarre i reset coordinati mantenendo esattamente grace e preserve flags correnti.
- [ ] Suite completa e commit `refactor: extract tracking lifecycle`.

### Task 4: Volume and scroll state machines

**Files:**
- Create: `handtracking_volume.py`
- Create: `handtracking_scroll.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_volume.py`
- Test: `tests/test_scroll.py`

**Interfaces:**
- Produces: `update_volume_state(...)` e `update_scroll_state(...)` con callback esplicite per volume/cursor/flow side effect.

- [ ] Scrivere RED test per volume entry, immediate release freeze, pose-loss release e scroll arm/release.
- [ ] Implementare le due macchine copiando l'ordine e le soglie esistenti.
- [ ] Sostituire i blocchi runtime e verificare suite completa.
- [ ] Commit `refactor: extract volume and scroll handlers`.

### Task 5: Split Spock from processing

**Files:**
- Create: `handtracking_spock.py`
- Modify: `handtracking_processing.py`
- Modify: `handtracking_runtime.py`
- Move/update: `tests/test_processing.py`
- Create: `tests/test_spock.py`

**Interfaces:**
- Produces: `SpockUpdate` e `update_spock_state(...)` identici semanticamente alla versione attuale.

- [ ] Scrivere/importare i test Spock contro il nuovo modulo e verificare RED.
- [ ] Spostare la macchina Spock senza modificarne il corpo logico.
- [ ] Rimuovere import/config inutilizzati da `handtracking_processing.py`.
- [ ] Suite completa e commit `refactor: split spock processing`.

### Task 6: Final orchestrator and contracts

Prima del final audit, estrarre il ramo `new_mp` in `handtracking_frame.py` per raggiungere il budget runtime approvato.

**Files aggiuntivi:**
- Create: `handtracking_frame.py`
- Create: `handtracking_modes.py`
- Create: `tests/test_frame.py`

**Interface:**
- Produces: `process_mediapipe_packet(session, packet, *, gray, now, camera_target_fps, callbacks...) -> FrameProcessResult`.

- [ ] Scrivere RED test per packet duplicato, packet nuovo senza mani e hand-switch volume che richiede skip frame.
- [ ] Implementare il processor usando `RuntimeSession` come stato scalare condiviso.
- [ ] Sostituire il ramo `new_mp` nel runtime con una singola chiamata.
- [ ] Tenere `handtracking_frame.py` sotto 300 righe spostando il coordinamento delle modalita' in `handtracking_modes.py`.
- [ ] Suite completa e commit `refactor: extract mediapipe frame processor`.

**Files:**
- Modify: `handtracking_runtime.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `tests/test_source_contracts.py`

**Interfaces:**
- Enforces: runtime 350-500 righe, `_run_impl()` <400, processing <300, nessun codice camera/stale/volume/scroll/Spock duplicato nel runtime.

- [ ] Aggiungere contract test failing sui nuovi confini e budget.
- [ ] Rimuovere residui/import morti e aggiornare documentazione.
- [ ] Eseguire suite completa, compile di ogni `*.py`, `pip check`, `git diff --check`.
- [ ] Eseguire smoke modello e worker reali, secret scan e audit AST.
- [ ] Integrare con fast-forward su `main`, riverificare e push senza force.
