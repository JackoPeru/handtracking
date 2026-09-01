# Runtime Performance and Stability Design

## Goal

Ridurre CPU, jitter, allocazioni e complessita' del path caldo senza cambiare gesture, soglie o semantica osservabile.

## Baseline

- Suite: 105 test.
- Geometria/classifier sintetici: ~63.9 us per mano/pass.
- LK forward+backward 640x360 sintetico: ~0.98 ms per chiamata.

## Interventi

1. Profiling interno leggero con EMA per camera/preprocess, LK, MediaPipe processing, render e loop totale.
2. Gate LK: eseguire optical flow solo quando pointer/scroll/swipe possono consumarlo.
3. `HandFeatures`: calcolare una sola volta per mano distanze, curl, pinch e score condivisi dai classifier principali.
4. `RuntimeSession` come unica source of truth: eliminare la copia locale->session->locale del runtime loop.
5. Throttling del solo HUD diagnostico: tracking/input e skeleton restano alla massima frequenza. La cache full-frame dello skeleton e' stata misurata piu' lenta del draw diretto e viene quindi esclusa.
6. Worker MediaPipe event-driven: eliminare polling 1 ms e aggiungere scheduling adattivo all'inferenza reale per limitare overwrite/queue pressure.
7. Riduzione allocazioni nel flow: aggiornamenti scalari/in-place invece di creare array NumPy temporanei per vettori 2D.
8. Benchmark prima/dopo riproducibile e contract test sulle nuove invarianti.

## Vincoli

- Nessuna modifica alle soglie gesture esistenti.
- Nessun input OS reale nei test.
- Nessuna webcam reale nei test automatici.
- Il worker continua a possedere il `HandLandmarker`.
- Fail-safe stale/dead worker resta prioritario.
- I miglioramenti devono essere misurabili o ridurre chiaramente un rischio di concorrenza/state drift.

## Target

- Suite completa verde.
- Nessun LK quando nessuna modalita' flow puo' usarlo.
- Una sola estrazione feature per mano/risultato MediaPipe.
- Nessun blocco di sincronizzazione massiva locals/session nel runtime.
- Worker idle senza polling periodico a 1 ms.
- Allocazioni NumPy 2D nel dispatch flow eliminate.
- Il riferimento LK mantiene sempre frame e punti coerenti; un fallimento invalida i punti prima del prossimo tentativo.
- `py_compile`, `pip check`, `git diff --check`, smoke modello/worker e secret scan puliti.
