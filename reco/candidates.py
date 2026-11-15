"""Etapa 1: ~200 candidatos de varias fuentes baratas (rol B).

Cada fuente cubre un caso: co-visitación cuando hay historial, contenido para productos nuevos,
popularidad y tendencias para usuarios sin vistas. Ninguna necesita entrenamiento.
"""
from __future__ import annotations

import numpy as np

from .session import Session
from .state import State

# (fuente, cupo máximo) en orden de prioridad; lo que sobra se rellena con los más vendidos
QUOTAS = (("recent", 10), ("view", 60), ("buy", 40), ("b2b", 20), ("knn", 30),
          ("new", 10), ("cat", 15), ("trending", 10))


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    nz = np.flatnonzero(scores > 0)
    if nz.size > k:
        nz = nz[np.argpartition(-scores[nz], k - 1)[:k]]
    return nz[np.argsort(-scores[nz], kind="stable")]


def generate(state: State, sess: Session, n: int = 200) -> np.ndarray:
    products = state.products
    total = len(products)
    allowed = products.available.copy()
    if sess.bought:
        allowed[[i for i in sess.bought if i < total]] = False
    taken = np.zeros(total, dtype=bool)
    out: list[int] = []

    def add(ids, limit: int) -> None:
        c = 0
        for i in ids:
            if c >= limit or len(out) >= n:
                return
            i = int(i)
            if 0 <= i < total and allowed[i] and not taken[i]:
                taken[i] = True
                out.append(i)
                c += 1

    pop = state.pop_at(sess.ts)
    for source, quota in QUOTAS:
        if len(out) >= n:
            break
        if source == "recent":
            add(sess.recent, quota)
        elif source in sess.scores:
            add(_top(sess.scores[source], 3 * quota), quota)
        elif source == "new":
            new = np.flatnonzero(products.live_new & allowed)
            if new.size:
                if sess.centroid is not None:
                    new = new[np.argsort(-(products.emb[new] @ sess.centroid), kind="stable")]
                else:
                    new = new[np.argsort(-products.created[new], kind="stable")]
                add(new, quota)
        elif source == "cat" and len(sess.cat_share):
            for c in np.argsort(-sess.cat_share)[:2]:
                if sess.cat_share[c] > 0:
                    add(pop.rank_by_cat.get(int(c), ()), quota)
        elif source == "trending":
            add(pop.trending, quota)
    add(pop.rank_global, n)
    if len(out) < n:                                    # catálogo raro: cualquiera disponible
        add(np.flatnonzero(allowed & ~taken), n)
    return np.array(out, dtype=np.int64)
