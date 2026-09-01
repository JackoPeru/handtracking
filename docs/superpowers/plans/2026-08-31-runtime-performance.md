# Runtime Performance Phase 4 Plan

1. [x] Aggiungere profiler leggero e test di timing/EMA senza dipendenze esterne.
2. [x] Aggiungere `should_measure_optical_flow()` con test RED/GREEN e integrare il gate nel loop.
3. [x] Introdurre `HandFeatures`/feature extraction con test di equivalenza rispetto ai classifier correnti; migrare processing/Spock/pointer/radial/two-hand ai dati cache dove utile.
4. [x] Eliminare la doppia copia stato locale/session nel runtime, mantenendo `RuntimeSession` come source of truth.
5. [x] Throttling HUD a 12 Hz con layer top-region. Cache skeleton full-frame provata e scartata perche' benchmarkata piu' lenta del draw diretto.
6. [x] Sostituire polling worker con `Event`, aggiungere submit scheduling adattivo e test di cadenza/wakeup/stop.
7. [x] Rimuovere allocazioni NumPy temporanee dal path flow, usare state/result slotted e aggiungere regression test.
8. [ ] Eseguire benchmark finale, suite, compile, pip check, AST/import audit, modello/worker reali, secret scan; merge fast-forward e push.
