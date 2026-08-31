# Runtime Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estrarre optical-flow e HUD dal loop principale e ridurre `_run_impl()` mantenendo invariato il comportamento delle gesture.

**Architecture:** La fase 2 mantiene `handtracking_runtime.py` come orchestratore. Le operazioni pure o quasi-pure per frame vengono spostate in moduli dedicati che ricevono stato e input espliciti; nessun nuovo singleton o side effect globale.

**Tech Stack:** Python 3.12, OpenCV, MediaPipe, NumPy, unittest.

**Spec:** `docs/superpowers/specs/2026-08-31-runtime-phase2-design.md`

## Global Constraints

- Nessuna modifica intenzionale di gesture o soglie.
- MediaPipe resta posseduto da `handtracking_mediapipe.py`.
- Win32 resta in `handtracking_windows.py`.
- Nuova logica testabile senza webcam reale.
- Il fail-safe stale conserva priorità assoluta sull'optical flow.

---

### Task 1: Extract optical-flow primitives

**Files:**
- Create: `handtracking_flow.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_flow.py`

**Interfaces:**
- Produces: funzioni per validare un passo LK, normalizzare/filtrare movimento e aggiornare `FlowState` senza input Win32.

- [ ] Scrivere test failing per validazione LK e aggiornamento filtered-motion.
- [ ] Eseguire i test mirati e verificare RED.
- [ ] Estrarre il codice optical-flow senza cambiare costanti.
- [ ] Eseguire test mirati + suite completa.
- [ ] Commit separato.

### Task 2: Extract HUD/status rendering

**Files:**
- Create: `handtracking_hud.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_hud.py`

**Interfaces:**
- Produces: `build_status_text(...)` e `draw_runtime_hud(...)` con dati espliciti e nessuna mutazione dello stato gesture.

- [ ] Scrivere test failing per testi delle modalità e diagnostica essenziale.
- [ ] Verificare RED.
- [ ] Spostare il blocco HUD fuori dal runtime.
- [ ] Eseguire test mirati + suite completa.
- [ ] Commit separato.

### Task 3: Extract MediaPipe-result processing helpers

**Files:**
- Create: `handtracking_processing.py`
- Modify: `handtracking_runtime.py`
- Test: `tests/test_processing.py`

**Interfaces:**
- Produces helper per aggiornare metriche FPS/inference, handedness selection e geometry metadata senza side effect di input.

- [ ] Identificare sottoblocchi realmente pure/semi-pure nel ramo `if new_mp`.
- [ ] Scrivere regression test sulle helper selezionate.
- [ ] Estrarre solo i blocchi che non richiedono un context object gigante.
- [ ] Suite completa e misurazione `_run_impl()`.
- [ ] Commit separato.

### Task 4: Final structural audit

**Files:**
- Modify: `README.md`, `AGENTS.md` se la struttura cambia.
- Test: contract tests esistenti o nuovi.

- [ ] Misurare righe/decisioni AST e import/stato morto.
- [ ] Aggiornare docs e source contracts.
- [ ] Eseguire 64+ test, `py_compile`, `pip check`, `git diff --check`.
- [ ] Eseguire smoke reale modello + worker MediaPipe.
- [ ] Commit finale, integrare su `main`, riverificare e push senza force.
