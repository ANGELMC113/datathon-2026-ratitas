"""Etapa 2a: una fila de features por par (petición, producto candidato) (rol B).

Reglas:
- Nunca un ID como feature: solo características que generalizan a usuarios y productos nuevos.
- Todo es relativo (cuotas, ratios, similitudes): la misma escala con 40 o 60 días de histórico.
- Lo desconocido se deja NaN (LightGBM aprende qué hacer con el hueco).
- Esta MISMA función genera las filas de entrenamiento y las de producción.
"""
from __future__ import annotations

import math

import numpy as np

from .artifacts import gather
from .session import Session
from .state import State
from .timeutil import DAY_S

FEATURE_NAMES: tuple[str, ...] = (
    "position", "hour", "device", "page",
    "n_recent", "in_recent", "recent_rank", "recent_count",
    "covis_view", "covis_buy", "covis_b2b", "content_knn", "content_sim", "content_sim_last",
    "cat_share", "sub_share", "brand_share", "same_sub_last",
    "log_price", "price_ratio", "age_days",
    "pop_view_1d", "pop_cart_1d", "pop_buy_1d", "pop_view_7d", "pop_buy_7d", "pop_buy_all",
    "conv_7d", "trend_view",
    "car_shown_log", "car_click_lift", "car_buy_lift",
    "tt_z", "tt_pct",
)
F = {name: k for k, name in enumerate(FEATURE_NAMES)}


def build_matrix(state: State, sess: Session, cand: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    """Matriz float32 (len(cand), len(FEATURE_NAMES)). `positions` solo al entrenar (en vivo, 0:
    puntuamos como si cada producto fuera a ir arriba y ordenamos por esa probabilidad)."""
    products = state.products
    cand = np.asarray(cand, dtype=np.int64)
    m = len(cand)
    X = np.full((m, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    if m == 0:
        return X
    pop = state.pop_at(sess.ts)
    car = state.artifacts.carousel

    X[:, F["position"]] = 0.0 if positions is None else positions
    if not math.isnan(sess.ts):
        X[:, F["hour"]] = (sess.ts % DAY_S) // 3600
    X[:, F["device"]] = sess.device
    X[:, F["page"]] = sess.page
    X[:, F["n_recent"]] = sess.n_recent

    X[:, F["in_recent"]] = [1.0 if int(i) in sess.recent_rank else 0.0 for i in cand]
    X[:, F["recent_rank"]] = [sess.recent_rank.get(int(i), np.nan) for i in cand]
    X[:, F["recent_count"]] = [sess.recent_count.get(int(i), 0) for i in cand]

    for kind, name in (("view", "covis_view"), ("buy", "covis_buy"), ("b2b", "covis_b2b"), ("knn", "content_knn")):
        X[:, F[name]] = gather(sess.scores[kind], cand, 0.0)

    emb = products.emb[cand]
    if sess.centroid is not None:
        X[:, F["content_sim"]] = emb @ sess.centroid
    if sess.last_emb is not None:
        X[:, F["content_sim_last"]] = emb @ sess.last_emb

    if len(sess.cat_share):
        X[:, F["cat_share"]] = gather(sess.cat_share.astype(np.float32), products.cat[cand].astype(np.int64), 0.0)
        X[:, F["sub_share"]] = gather(sess.sub_share.astype(np.float32), products.sub[cand].astype(np.int64), 0.0)
        X[:, F["brand_share"]] = gather(sess.brand_share.astype(np.float32), products.brand[cand].astype(np.int64), 0.0)
        X[:, F["same_sub_last"]] = (products.sub[cand] == sess.last_sub).astype(np.float32)

    price = products.price[cand].astype(np.float32)
    X[:, F["log_price"]] = np.log1p(np.maximum(price, 0))
    if not math.isnan(sess.med_price) and sess.med_price > 0:
        X[:, F["price_ratio"]] = price / sess.med_price
    created = products.created[cand]
    age = (sess.ts - created) / DAY_S
    age[created <= 0] = np.nan
    X[:, F["age_days"]] = age

    for key in ("view_1d", "cart_1d", "buy_1d", "view_7d", "buy_7d", "buy_all"):
        X[:, F[f"pop_{key}"]] = gather(pop.shares[key], cand, 0.0)
    buy7 = gather(pop.counts["buy_7d"], cand, 0.0)
    view7 = gather(pop.counts["view_7d"], cand, 0.0)
    X[:, F["conv_7d"]] = (buy7 + 0.5) / (view7 + 50.0)
    X[:, F["trend_view"]] = (X[:, F["pop_view_1d"]] + 1.0) / (X[:, F["pop_view_7d"]] + 1.0)

    X[:, F["car_shown_log"]] = gather(car.shown_log, cand, 0.0)
    X[:, F["car_click_lift"]] = gather(car.click_lift, cand)
    X[:, F["car_buy_lift"]] = gather(car.buy_lift, cand)

    # afinidad del two-tower (reco/tt.py). Sin modelo cargado o si falla: NaN, y el LightGBM sigue
    tt = getattr(state, "tt", None)
    if tt is not None:
        try:
            X[:, F["tt_z"]], X[:, F["tt_pct"]] = tt.features(products, sess, cand)
        except Exception as exc:                      # nunca tumbar una respuesta por esta feature
            print(f"[tt] feature falló: {exc!r}")
    return X
