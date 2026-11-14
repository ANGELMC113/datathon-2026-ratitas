"""Etapa 3: de la lista ordenada a los 10 que se muestran (rol D).

1. Filtro: solo productos disponibles (activos y con stock), distintos y que el usuario no haya comprado.
2. Huecos 1-7: los 7 mejores según el modelo (explotación).
3. Huecos 8-10: Thompson sampling. Para cada producto del "banquillo" (puestos 8-40 y productos
   nuevos) tomamos una Beta cuyo centro es la predicción del modelo y que se va afinando con las
   compras reales del feedback; sorteamos y elegimos los 3 mejores sorteos. Un producto del que
   sabemos poco tiene una Beta ancha y a veces gana: así aprendemos sin arriesgar los primeros huecos.
4. Si falta algo, se rellena con la lista de emergencia. Siempre salen 10.
"""
from __future__ import annotations

import numpy as np

from .session import Session
from .state import State

POOL_SIZE = 33
MAX_NEW_IN_POOL = 10


def rerank_idx(state: State, ranked: np.ndarray, scores: np.ndarray, sess: Session,
               rng: np.random.Generator | None, n: int = 10, n_explore: int = 3,
               prior: float = 50.0) -> list[int]:
    products = state.products
    total = len(products)
    seen: set[int] = set()
    clean: list[int] = []
    clean_s: list[float] = []
    for i, s in zip(ranked, scores):
        i = int(i)
        if i in seen or i in sess.bought or not (0 <= i < total) or not products.available[i]:
            continue
        seen.add(i)
        clean.append(i)
        clean_s.append(float(s))

    n_explore = n_explore if rng is not None else 0
    head = clean[: n - n_explore]
    explore: list[int] = []
    if n_explore > 0 and len(clean) > len(head):
        pool = list(range(len(head), min(len(clean), len(head) + POOL_SIZE)))
        extra_new = [k for k in range(len(head) + POOL_SIZE, len(clean)) if products.live_new[clean[k]]]
        pool += extra_new[:MAX_NEW_IN_POOL]
        p = np.clip(np.array([clean_s[k] for k in pool], dtype=np.float64), 1e-4, 0.5)
        shown, buys = state.bandit.lookup([products.ids[clean[k]] for k in pool])
        a = 1.0 + prior * p + buys
        b = 1.0 + prior * (1.0 - p) + (shown - buys)
        theta = rng.beta(a, b)
        explore = [clean[pool[k]] for k in np.argsort(-theta)[:n_explore]]

    out = head + explore
    if len(out) < n:                                  # pocos candidatos: completa con el resto
        used = set(out)
        out += [i for i in clean if i not in used][: n - len(out)]
    if len(out) < n:                                  # y si aún faltan, con la lista de emergencia
        used_ids = {products.ids[i] for i in out} | {products.ids[i] for i in sess.bought if i < total}
        for pid in state.fallback_items(n - len(out), exclude=used_ids):
            out.append(products.index[pid])
    return out[:n]
