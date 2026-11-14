"""Pipeline de recomendación: candidatos -> features -> score -> rerank.

Cada función tiene dueño (ver CONTRACTS.md):
    candidates.py  -> get_candidates()   B (Angel y Martí)
    features.py    -> build_features()   B (Angel y Martí)
    model.py       -> score()            C (Angel y Martí)
    rerank.py      -> rerank()           D (Inés)
serve.py (A) solo llama a estas funciones; no las implementa.
"""
