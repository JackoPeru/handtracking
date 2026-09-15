# Hand Tracking

Applicazione Windows per controllare mouse e gesture tramite webcam, MediaPipe e optical flow OpenCV.

Supporto: Windows x64 con CPython 3.12 x64. Avvio: `Avvia Hand Tracking.bat`.

Al primo avvio il launcher crea automaticamente `.venv` se manca e verifica che l'interprete sia CPython 3.12 x64. A ogni avvio riconcilia l'ambiente con `requirements.lock` usando `pip --require-hashes --only-binary=:all:` prima di eseguire l'applicazione; non esegue upgrade non bloccati di `pip` e non usa `requirements.txt` per l'installazione runtime. Il modello `hand_landmarker.task` e' incluso nel repository e viene risolto rispetto alla cartella del progetto, quindi l'app puo' essere avviata anche da una working directory diversa.

Setup manuale equivalente:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements.lock
.venv\Scripts\python.exe main.py
```

`requirements.txt` contiene solo le dipendenze dirette leggibili. `requirements.lock` e' il manifest operativo: blocca versioni e hash delle wheel verificate per Windows x64/CPython 3.12. Non aggirare il lock con installazioni non hashate o con sorgenti non binarie.

Aggiornamento del lock:

1. Aggiornare solo i pin diretti in `requirements.txt`.
2. In un ambiente temporaneo Windows x64/CPython 3.12, scaricare la chiusura completa in una wheelhouse locale: `py -3.12 -m venv .lock-venv`, poi `.lock-venv\Scripts\python.exe -m pip download --only-binary=:all: --dest .lock-wheelhouse -r requirements.txt`.
3. Calcolare l'hash di ogni wheel con `.lock-venv\Scripts\python.exe -m pip hash <wheel>` (in PowerShell si puo' iterare `Get-ChildItem .lock-wheelhouse\*.whl`). Per ogni archivio leggere `*.dist-info/METADATA` e riportare `Name` e `Version` esatti in `requirements.lock`, con il relativo `sha256` della wheel CPython 3.12 x64; verificare il diff, senza righe non versionate o prive di hash.
4. Verificare il lock offline in una virtualenv pulita: `.lock-venv\Scripts\python.exe -m pip install --no-index --find-links .lock-wheelhouse --require-hashes --only-binary=:all: -r requirements.lock`, poi `.lock-venv\Scripts\python.exe -m pip check` e `.lock-venv\Scripts\python.exe -m unittest discover -s tests -v`.
5. Dopo la verifica, rimuovere gli artefatti temporanei `.lock-venv` e `.lock-wheelhouse` senza includerli nel repository.

Il launcher e la CI devono continuare a usare esclusivamente questo lock; modifiche alle dipendenze richiedono quindi aggiornamento coordinato di `requirements.txt`, `requirements.lock` e dei test di contratto.

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
- `handtracking_display.py`: layer HUD cached a frequenza ridotta senza rallentare tracking/input.
- `handtracking_perf.py`: profiler EMA e scheduler adattivo dei submit MediaPipe.
- `handtracking_core.py`: logica pura e testabile di priorita', timing e tracking della mano.
- `handtracking_mediapipe.py`: worker di inferenza che possiede il ciclo di vita del `HandLandmarker`.
- `benchmarks/hotpath_benchmark.py`: micro-benchmark riproducibile di geometria, LK, preprocessing e rendering.
- `tests/`: regressioni automatiche che non richiedono la webcam.
- `snapshots/`: eventuali vecchie versioni locali sono ignorate da Git e non fanno parte del repository pubblico.

Test logici senza webcam: `python -m unittest discover -s tests -v`.

Benchmark hot path: `python -m benchmarks.hotpath_benchmark`.

Ottimizzazioni runtime principali:

- optical flow LK eseguito solo quando pointer/scroll/swipe possono consumarlo;
- preprocessing 640x360/gray eseguito solo per submit MediaPipe, LK o nuovo packet da riallineare;
- `HandFeatures` memoizza geometria e angoli condivisi per ogni mano/risultato MediaPipe;
- `RuntimeSession` e' la singola source of truth dello stato scalare del loop;
- HUD diagnostico aggiornato a 12 Hz su un layer limitato alla fascia superiore; tracking e input non vengono throttled;
- skeleton/overlay resta diretto: il benchmark locale ha mostrato che una cache full-frame con `copyTo` e' piu' lenta;
- worker MediaPipe event-driven, senza polling a 1 ms, con scheduler submit adattivo e latest-frame-wins;
- state/result object hot slotted e aggiornamenti 2D flow in-place per ridurre allocazioni.

Sul benchmark sintetico usato durante la fase performance, la geometria rappresentativa e' passata da circa 65 us a 51 us per mano (-20% circa); HUD da circa 0,80 ms/frame diretto a 0,39 ms/frame medio cached. I valori dipendono dall'hardware: usare sempre il benchmark locale prima di modificare frequenze o strategie di caching.

La modalita' a due mani implementa lo zoom. La vecchia indicazione di rotazione e' stata rimossa perche' non esiste una scorciatoia di rotazione universale affidabile tra le applicazioni Windows.
