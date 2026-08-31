# Hand Tracking Runtime Modularization Design

## Goal

Ridurre `handtracking_runtime.py` da monolite di oltre 2.600 righe a un orchestratore leggibile e testabile, correggendo prima i bug di lifecycle/stato emersi dall'audit senza cambiare le gesture intenzionalmente.

## Constraints

- Windows, Python 3.12.
- Conservare MediaPipe 1.0.1, OpenCV contrib 5.0.0.93, NumPy 2.5.2, pycaw 20251023 e comtypes 1.4.16.
- Nessun cambio arbitrario alle soglie gesture o alla sensibilita' del puntatore.
- Il pugno resta clutch globale.
- La modalita' a due mani resta zoom-only.
- Il runtime deve restare import-safe e testabile senza webcam/input reali.
- Ogni bug fix deve avere un test che fallisce prima della correzione.
- Ogni estrazione di codice deve lasciare la suite completamente verde.

## Bugs to fix before refactoring

1. `run()` deve avere cleanup garantito con `try/finally` per webcam, MediaPipe worker, cursor worker e finestre OpenCV.
2. Se il worker MediaPipe muore durante l'inizializzazione, il runtime deve terminare con un errore esplicito invece di restare fullscreen inutilizzabile.
3. Un timeout MediaPipe durante uno Spock gia' latched non deve permettere un secondo toggle senza un vero rilascio della posa.
4. La priorita' volume deve includere anche la fase candidate, impedendo a swipe/pointer/radial di prevaricarla.
5. Snapshot risultato + statistiche MediaPipe devono poter essere letti atomicamente.
6. Il launcher deve richiedere esplicitamente Python 3.12 quando crea una nuova `.venv`.

## Target architecture

### `main.py`

Entry-point reale dell'app. `test.py` resta temporaneamente come compatibility shim che importa/invoca `main.main()` per non rompere launcher o riferimenti esistenti durante la transizione; al termine il launcher punta a `main.py`.

### `handtracking_config.py`

Contiene esclusivamente costanti di configurazione: camera, cursor, flow, pointer, scroll, volume, swipe, radial, two-hand e Spock. Nessun side effect.

### `handtracking_gestures.py`

Contiene geometria e classificatori puri delle pose: distanze, angoli, fist, volume, pointer, swipe, Spock, radial e two-hand geometry. Non invia input Windows e non usa OpenCV UI.

### `handtracking_windows.py`

Contiene adapter Windows: mouse/tastiera, cursor worker, screen metrics, foreground window, browser back/forward, volume endpoint. I side effect vengono inizializzati esplicitamente e non durante l'import del runtime.

### `handtracking_render.py`

Contiene solo rendering OpenCV: scheletro mano, menu radiale, overlay two-hand e status/debug overlay. Non modifica lo stato gesture.

### `handtracking_state.py`

Contiene dataclass/state object con metodi `reset()` per gruppi di stato ripetutamente azzerati: pointer, volume, scroll, swipe, radial, two-hand, Spock e flow. Serve a eliminare reset manuali divergenti.

### `handtracking_engine.py`

Contiene la priorita' centrale delle modalita' e helper di transizione. La priorita' unica e':

`fist > volume (active o candidate) > two-hand > radial > scroll > swipe > pointer`.

Non gestisce camera, thread o rendering.

### `handtracking_mediapipe.py`

Resta proprietario esclusivo del `HandLandmarker`. Aggiunge uno snapshot atomico (`snapshot_state`) che restituisce latest result e stats coerenti sotto un unico lock.

### `handtracking_runtime.py`

Resta orchestratore: camera loop, submit MediaPipe, optical flow, chiamate all'engine, dispatch input e rendering. Target indicativo 300-700 righe; non e' necessario forzare il numero se farlo peggiora le interfacce.

## State/lifecycle rules

- Se MediaPipe e' stale, tutti gli output gestuali si congelano subito.
- Se Spock era latched prima dello stale, rimane in stato "release required" finche' non viene osservata una posa non-Spock fresca per il tempo di rilascio.
- Se il worker MediaPipe non e' vivo e non e' stato richiesto lo shutdown, `run()` solleva `RuntimeError` con `last_error`.
- Il cleanup deve essere idempotente e avvenire anche se camera read, rendering, gesture logic o input Windows generano eccezioni.
- Lo stato di una gesture deve essere azzerato tramite il relativo `reset()` anziche' replicando liste di assegnazioni.

## Testing strategy

- Unit test per utility e state reset.
- Worker test per lifecycle, error reporting e snapshot atomico.
- Runtime smoke test per cleanup su eccezione e worker morto.
- Gesture priority test su volume candidate vs radial/swipe/pointer.
- Spock stale/recovery test che dimostri assenza di doppio toggle.
- Source-contract tests ridotti progressivamente: preferire comportamento reale a ricerca di stringhe.
- Test completi: `python -m unittest discover -s tests -q`, `py_compile`, `pip check`, `git diff --check`.

## Migration strategy

Il refactor procede in estrazioni meccaniche, una responsabilita' alla volta. Dopo ogni modulo estratto la suite deve restare verde e viene creato un commit isolato. Non si modificano contemporaneamente architettura e soglie gesture.
