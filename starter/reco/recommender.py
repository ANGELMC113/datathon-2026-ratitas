"""Une las piezas: petición -> sesión -> candidatos -> features -> score -> rerank -> 10 IDs.

También expone las funciones de CONTRACTS.md (get_candidates, build_features, score, rerank)
para que cada rol pueda probar su parte por separado con la misma lógica que corre en vivo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import candidates as cand_mod
from .features import FEATURE_NAMES, build_matrix
from .model import HeuristicScorer, Scorer
from .rerank import rerank_idx
from .session import Session, make_session, request_ts
from .state import State


@dataclass
class Recommendation:
    items: list[str]
    version: str
    cold: bool
    n_candidates: int
    raw: dict | None = None        # pid -> probabilidad del modelo sin corrección (para aprender el lift)


class Recommender:
    def __init__(self, state: State, scorer: Scorer | None = None, *, n_candidates: int = 200,
                 n_items: int = 10, n_explore: int = 3, explore_prior: float = 50.0,
                 max_recent: int = 10, decay: float = 0.7, seed: int | None = None,
                 max_per_sub: int = 0) -> None:
        self.state = state
        self.scorer = scorer or HeuristicScorer()
        self.n_candidates = n_candidates
        self.n_items = n_items
        self.n_explore = n_explore
        self.explore_prior = explore_prior
        self.max_recent = max_recent
        self.decay = decay
        self.rng = np.random.default_rng(seed)
        self.max_per_sub = max_per_sub
        self.quotas = cand_mod.QUOTAS

    def session(self, request: dict, state: State | None = None) -> Session:
        return make_session(state or self.state, request, self.max_recent, self.decay)

    def rank(self, request: dict, scorer: Scorer | None = None, state: State | None = None,
             quotas: tuple | None = None, rebuy: bool = False,
             adapt: bool = False) -> tuple[Session, np.ndarray, np.ndarray]:
        """Candidatos ordenados por score (sin exploración). Lo usa también la evaluación offline."""
        sess, cand, scores, _ = self._rank(request, scorer, state, quotas, rebuy, adapt)
        return sess, cand, scores

    def _rank(self, request: dict, scorer: Scorer | None, state: State | None, quotas: tuple | None,
              rebuy: bool, adapt: bool) -> tuple[Session, np.ndarray, np.ndarray, np.ndarray]:
        """Como rank, y además la probabilidad del modelo sin corregir (en el mismo orden).
        rebuy=True no excluye lo que el usuario ya compró (variante para medir si se recompra).
        adapt=True multiplica cada score por el lift aprendido del feedback en vivo (LiftStats)."""
        scorer = scorer or self.scorer
        if state is None:
            state = self.state.frozen(request_ts(request))
        sess = self.session(request, state)
        if rebuy and sess.bought:
            sess = replace(sess, bought=set())
        cand = cand_mod.generate(state, sess, self.n_candidates, quotas or self.quotas)
        if len(cand) == 0:
            return sess, cand, np.zeros(0), np.zeros(0)
        raw = np.asarray(scorer.predict(build_matrix(state, sess, cand)), dtype=np.float64)
        scores = raw
        if adapt:
            ids = state.products.ids
            scores = raw * state.lift.lookup([ids[i] for i in cand])
        order = np.argsort(-scores, kind="stable")
        return sess, cand[order], scores[order], raw[order]

    def recommend(self, request: dict, explore: bool = True, scorer: Scorer | None = None,
                  n_explore: int | None = None, max_per_sub: int | None = None,
                  quotas: tuple | None = None, rebuy: bool = False, adapt: bool = False) -> Recommendation:
        scorer = scorer or self.scorer            # se lee una vez: la recarga en caliente no nos pilla a medias
        state = self.state.frozen(request_ts(request))
        sess, ranked, scores, raw = self._rank(request, scorer, state, quotas, rebuy, adapt)
        idx = rerank_idx(state, ranked, scores, sess, self.rng if explore else None,
                         n=self.n_items, n_explore=self.n_explore if n_explore is None else n_explore,
                         prior=self.explore_prior,
                         max_per_sub=self.max_per_sub if max_per_sub is None else max_per_sub)
        ids = state.products.ids
        raw_of = {int(i): float(r) for i, r in zip(ranked, raw)}
        items = [ids[i] for i in idx]
        return Recommendation(items, scorer.version, sess.cold, len(ranked),
                              {ids[i]: raw_of[int(i)] for i in idx if int(i) in raw_of})


def repair(items: list, state: State, n: int = 10, exclude: set[str] = frozenset()) -> list[str]:
    """Conserva, en orden, los IDs que siguen siendo válidos y completa con la lista de emergencia
    (p. ej. si un producto se agotó entre el cálculo y la validación)."""
    products = state.products
    keep: list[str] = []
    for pid in items or []:
        i = products.index.get(pid) if isinstance(pid, str) else None
        if i is not None and products.available[i] and pid not in keep and pid not in exclude:
            keep.append(pid)
    if len(keep) < n:
        keep += state.fallback_items(n - len(keep), exclude=set(keep) | set(exclude))
    return keep[:n]


def is_valid(items: list, state: State, n: int = 10) -> bool:
    """Regla del reto: exactamente n IDs distintos y activos (y con stock)."""
    if not isinstance(items, list) or len(items) != n or len(set(items)) != n:
        return False
    products = state.products
    for pid in items:
        i = products.index.get(pid) if isinstance(pid, str) else None
        if i is None or not products.available[i]:
            return False
    return True


# ----------------------------------------------------------------------------------------------
# Interfaces de CONTRACTS.md. `context` es la petición completa (user_id, ts, is_new_user, context,
# recent_views): así se resuelve el punto abierto sobre la firma de recommend().
# ----------------------------------------------------------------------------------------------
_default: Recommender | None = None


def set_default(recommender: Recommender) -> None:
    global _default
    _default = recommender


def _rec(recommender: Recommender | None) -> Recommender:
    r = recommender or _default
    if r is None:
        raise RuntimeError("llama antes a set_default(Recommender(...))")
    return r


def get_candidates(user_id: str, context: dict, recommender: Recommender | None = None) -> list[str]:
    r = _rec(recommender)
    sess = r.session({**context, "user_id": user_id})
    return [r.state.products.ids[i] for i in cand_mod.generate(r.state, sess, r.n_candidates)]


def build_features(user_id: str, candidates: list[str], context: dict,
                   recommender: Recommender | None = None) -> pd.DataFrame:
    r = _rec(recommender)
    sess = r.session({**context, "user_id": user_id})
    idx = np.array([r.state.products.index[p] for p in candidates], dtype=np.int64)
    df = pd.DataFrame(build_matrix(r.state, sess, idx), columns=list(FEATURE_NAMES))
    df.insert(0, "product_id", candidates)
    return df


def score(features: pd.DataFrame, recommender: Recommender | None = None) -> np.ndarray:
    r = _rec(recommender)
    return np.asarray(r.scorer.predict(features[list(FEATURE_NAMES)].to_numpy(np.float32)))


def rerank(ranked: list[str], user_id: str, context: dict, scores: np.ndarray | None = None,
           recommender: Recommender | None = None) -> list[str]:
    r = _rec(recommender)
    sess = r.session({**context, "user_id": user_id})
    idx = np.array([r.state.products.index[p] for p in ranked], dtype=np.int64)
    s = np.asarray(scores) if scores is not None else np.linspace(0.05, 0.001, len(idx))
    out = rerank_idx(r.state, idx, s, sess, r.rng, r.n_items, r.n_explore, r.explore_prior)
    return [r.state.products.ids[i] for i in out]
