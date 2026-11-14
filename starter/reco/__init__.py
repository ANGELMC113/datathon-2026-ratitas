"""Recomendador del equipo ratitas (RawTree Challenge, Datathon FME 2026).

Flujo por petición (mismo código offline y en vivo):
    session.py     petición -> lo que sabemos del comprador (vistas recientes, afinidades)
    candidates.py  ~200 candidatos: co-visitación, contenido, tendencias, populares      (rol B)
    features.py    una fila de features por candidato                                    (rol B)
    model.py       probabilidad de compra (LightGBM; heurística si no hay modelo)        (rol C)
    rerank.py      filtro de stock + Thompson sampling en los huecos 8-10                (rol D)
    recommender.py une las piezas y expone las funciones de CONTRACTS.md
    live.py        estado en vivo refrescado en segundo plano                            (rol A)
"""
