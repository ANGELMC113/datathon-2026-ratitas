"""Lo que sabemos de un comprador en esta petición: sus vistas recientes y lo que se deriva de ellas.
Se calcula una vez por petición y lo usan candidatos y features (mismo código offline y en vivo)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .catalog import DEVICE_CODES, PAGE_CODES
from .state import State
from .timeutil import parse_ts, to_float

KNN_ON_THE_FLY = 30


@dataclass
class Session:
    ts: float
    user_id: str
    is_new_user: bool
    device: float
    page: float
    n_recent: int                 # vistas en la petición (incluidas las de productos desconocidos)
    recent: np.ndarray            # índices de productos, el más reciente primero, sin repetir
    recent_w: np.ndarray          # peso por recencia: decay**k
    recent_rank: dict[int, int]
    recent_count: dict[int, int]
    scores: dict[str, np.ndarray]  # puntuación densa por producto: view, buy, b2b, knn
    cat_share: np.ndarray
    sub_share: np.ndarray
    sub_trans: np.ndarray         # por código de subcategoría: tasa de compra tras las vistas
    brand_share: np.ndarray
    last_sub: int
    centroid: np.ndarray | None
    last_emb: np.ndarray | None
    med_price: float
    bought: set[int]
    cold: bool                    # usuario nuevo o vista de un producto nuevo (definición del reto)


def _context(request: dict) -> dict:
    ctx = request.get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except ValueError:
            ctx = None
    if not isinstance(ctx, dict):
        ctx = {k.split(".", 1)[1]: v for k, v in request.items() if k.startswith("context.")}
    return ctx


def recent_views(request: dict) -> list[tuple[str, float]]:
    rv = request.get("recent_views")
    if isinstance(rv, str):
        try:
            rv = json.loads(rv)
        except ValueError:
            rv = None
    if rv is None or isinstance(rv, float):          # NaN de pandas
        return []
    out: list[tuple[str, float]] = []
    for v in list(rv):
        if isinstance(v, dict):
            pid = v.get("product_id")
            if pid:
                out.append((str(pid), parse_ts(v.get("ts"))))
        elif v:
            out.append((str(v), math.nan))
    return out


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1")


def request_ts(request: dict) -> float:
    if request.get("ts_ms") not in (None, ""):
        return to_float(request["ts_ms"]) / 1000.0
    return parse_ts(request.get("ts"))


def make_session(state: State, request: dict, max_recent: int = 10, decay: float = 0.7) -> Session:
    products = state.products
    art = state.artifacts
    n = len(products)
    ctx = _context(request)
    ts = request_ts(request)
    if math.isnan(ts):                              # sin ts válido: ahora
        ts = time.time()
    user_id = str(request.get("user_id") or "")

    views = recent_views(request)
    if views and all(not math.isnan(t) for _, t in views):
        order = sorted(range(len(views)), key=lambda k: (views[k][1], k), reverse=True)
    else:
        order = list(range(len(views)))[::-1]       # sin ts: asumimos orden cronológico
    recent: list[int] = []
    count: dict[int, int] = {}
    cold = _bool(request.get("is_new_user", False))
    for k in order:
        i = products.index.get(views[k][0])
        if i is None:
            continue
        first = i not in count
        count[i] = count.get(i, 0) + 1
        if products.live_new[i]:
            cold = True
        if first and len(recent) < max_recent:
            recent.append(i)
    rec = np.array(recent, dtype=np.int64)
    w = decay ** np.arange(len(rec), dtype=np.float32)

    # co-visitación y vecinos de contenido acumulados sobre las vistas recientes (arrays densos)
    scores: dict[str, np.ndarray] = {}
    for kind in ("view", "buy", "b2b"):
        acc = np.zeros(n, dtype=np.float32)
        nb = art.covis.get(kind)
        if nb is not None:
            for i, wi in zip(rec, w):
                got = nb.get(int(i))
                if got is not None:
                    acc[got[0]] += wi * got[1]
        scores[kind] = acc
    acc = np.zeros(n, dtype=np.float32)
    for i, wi in zip(rec, w):
        got = art.content_knn.get(int(i))
        if got is None and products.emb.shape[1] > 1:   # producto nuevo: vecinos calculados al vuelo
            sims = products.emb @ products.emb[i]
            sims[i] = -np.inf
            k = min(KNN_ON_THE_FLY, n - 1)
            if k > 0:
                top = np.argpartition(-sims, k - 1)[:k]
                got = (top, np.maximum(sims[top], 0).astype(np.float32))
        if got is not None:
            acc[got[0]] += wi * got[1]
    scores["knn"] = acc

    # transiciones "vista -> compra en el carrusel" (si los artefactos las traen)
    acc = np.zeros(n, dtype=np.float32)
    trans = getattr(art, "trans_prod", None)
    if trans is not None:
        for i, wi in zip(rec, w):
            got = trans.get(int(i))
            if got is not None:
                acc[got[0]] += wi * got[1]
    scores["trans"] = acc
    sub_trans = np.zeros(len(products.vocab_sub.codes), dtype=np.float32)
    trans_sub = getattr(art, "trans_sub", None) or {}
    if trans_sub and len(rec):
        names = products.sub_names()
        ws_t = w / w.sum()
        for i, wi in zip(rec, ws_t):
            row = trans_sub.get(names[int(products.sub[i])])
            if row:
                for sb, val in row.items():
                    c = products.vocab_sub.codes.get(sb)
                    if c is not None:
                        sub_trans[c] += wi * val

    if len(rec):
        ws = w / w.sum()
        cat_share = np.bincount(products.cat[rec], weights=ws, minlength=len(products.vocab_cat.codes))
        sub_share = np.bincount(products.sub[rec], weights=ws, minlength=len(products.vocab_sub.codes))
        brand_share = np.bincount(products.brand[rec], weights=ws, minlength=len(products.vocab_brand.codes))
        c = (products.emb[rec] * ws[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(c))
        centroid = (c / norm).astype(np.float32) if norm > 1e-9 else None
        last_emb = products.emb[rec[0]]
        last_sub = int(products.sub[rec[0]])
        med_price = float(np.median(products.price[rec]))
    else:
        cat_share = sub_share = brand_share = np.zeros(0)
        centroid = last_emb = None
        last_sub = -1
        med_price = math.nan

    bought = {products.index[p] for p in state.bought(user_id) if p in products.index}
    return Session(
        ts=ts, user_id=user_id, is_new_user=_bool(request.get("is_new_user", False)),
        device=float(DEVICE_CODES.get(str(ctx.get("device", "")), math.nan)),
        page=float(PAGE_CODES.get(str(ctx.get("page", "")), math.nan)),
        n_recent=len(views), recent=rec, recent_w=w,
        recent_rank={int(i): k for k, i in enumerate(rec)}, recent_count=count,
        scores=scores, cat_share=cat_share, sub_share=sub_share, sub_trans=sub_trans, brand_share=brand_share,
        last_sub=last_sub, centroid=centroid, last_emb=last_emb, med_price=med_price,
        bought=bought, cold=cold,
    )
