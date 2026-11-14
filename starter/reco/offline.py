"""Construcción offline (rol B y C): descarga de RawTree, artefactos por corte temporal, filas de
entrenamiento y evaluación de políticas. Todo con la misma lógica que corre en vivo."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import sql
from .artifacts import Artifacts, CarouselStats, DailyPop, Neighbors
from .catalog import ContentModel, ProductTable, fit_content_model, knn, product_text
from .client import RawTreeClient
from .features import FEATURE_NAMES, build_matrix
from .model import Scorer
from .recommender import Recommender
from .session import make_session
from .state import State
from .timeutil import DAY_MS, parse_ts, to_float


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Descargas
# ----------------------------------------------------------------------------------------------
def history_range(client: RawTreeClient) -> tuple[int, int]:
    r = client.query(sql.HISTORY_RANGE, timeout=120)[0]
    return int(to_float(r["t0"])), int(to_float(r["t1"]))


def load_catalog(client: RawTreeClient) -> pd.DataFrame:
    df = client.query_df(sql.CATALOG, timeout=120)
    df["product_id"] = df["product_id"].astype(str)
    return df.drop_duplicates("product_id", keep="last").sort_values("product_id").reset_index(drop=True)


def fit_content(catalog: pd.DataFrame) -> tuple[ContentModel, np.ndarray]:
    texts = [product_text(str(c), str(s), str(b), str(d)) for c, s, b, d in
             zip(catalog["category"], catalog["subcategory"], catalog["brand"], catalog["description"])]
    content = fit_content_model(texts)
    return content, content.embed(texts)


def download_daily(client: RawTreeClient, product_ids: list[str],
                   tables: tuple[str, ...] = ("shop.history_events",)) -> DailyPop:
    frames: dict[str, pd.DataFrame] = {}
    for event in ("view", "add_to_cart", "purchase"):
        parts = []
        for table in tables:
            try:
                parts.append(client.query_df(sql.daily_counts(table, event), timeout=600))
            except Exception as exc:
                log(f"conteos diarios de {table}/{event} fallaron: {exc}")
        parts = [p for p in parts if not p.empty]
        frames[event] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["day", "product_id", "n"])
        log(f"conteos diarios {event}: {len(frames[event])} filas")
    return DailyPop.from_frames(frames, product_ids)


def build_artifacts(client: RawTreeClient, product_ids: list[str], content: ContentModel | None,
                    emb: np.ndarray, until_ms: int, since_ms: int = 0, top_k: int = 40,
                    knn_k: int = 30) -> Artifacts:
    """Todo lo que se puede precalcular con el histórico anterior a `until_ms`."""
    n = len(product_ids)
    index = {p: i for i, p in enumerate(product_ids)}
    covis: dict[str, Neighbors] = {}
    for kind in sql.COVIS_KINDS:
        t = time.time()
        try:
            df = client.query_df(sql.covis(kind, since_ms, until_ms, top_k), timeout=900)
            covis[kind] = Neighbors.from_frame(df, index, n, top_k)
            log(f"  co-visitación {kind}: {len(df)} pares ({time.time() - t:.0f} s)")
        except Exception as exc:
            log(f"  co-visitación {kind} FALLÓ ({exc}); queda vacía")
            covis[kind] = Neighbors.empty(n)
    try:
        car = CarouselStats.from_frame(client.query_df(sql.carousel_stats(since_ms, until_ms), timeout=600), index, n)
        log("  estadísticas del carrusel listas")
    except Exception as exc:
        log(f"  estadísticas del carrusel FALLARON ({exc}); quedan vacías")
        car = CarouselStats.empty(n)
    sales = client.query(sql.best_sellers(until_ms), timeout=300)
    ranked = sorted(((int(to_float(r["n"], 0)), str(r["product_id"])) for r in sales), reverse=True)
    buy_all = np.zeros(n, dtype=np.float32)
    for c, p in ranked:
        if p in index:
            buy_all[index[p]] = c
    nb_idx, nb_sim = knn(emb, knn_k)
    return Artifacts(product_ids=list(product_ids), until_ms=int(until_ms), covis=covis,
                     content_knn=Neighbors.from_dense(nb_idx, np.maximum(nb_sim, 0)), carousel=car,
                     content=content, emb=emb.astype(np.float32), buy_all=buy_all,
                     best_sellers=[p for _, p in ranked],
                     meta={"since_ms": int(since_ms), "built_at": time.time()})


def download_impressions(client: RawTreeClient, since_ms: int, until_ms: int, keep_permille: int,
                         chunk_ms: int = DAY_MS) -> pd.DataFrame:
    """Por trozos de un día para que ninguna respuesta sea enorme."""
    parts = []
    t = since_ms
    while t < until_ms:
        parts.append(client.query_df(sql.impressions(t, min(t + chunk_ms, until_ms), keep_permille), timeout=600))
        t += chunk_ms
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if not df.empty:
        df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")
        for c in ("clicked", "purchased"):
            df[c] = df[c].fillna("").astype(str)
    log(f"impresiones [{since_ms}, {until_ms}): {len(df)} (con compra: {int((df.get('purchased', pd.Series(dtype=str)) != '').sum())})")
    return df


def feedback_impressions(client: RawTreeClient, team: str, logs_dir: Path) -> pd.DataFrame:
    """Nuestras propias respuestas (logs/served-*.jsonl) unidas al feedback: lo que el stream nos
    ha enseñado, en el mismo formato que las impresiones del histórico."""
    reqs: dict[str, dict] = {}
    for path in sorted(logs_dir.glob("served-*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                reqs[str(r.get("request_id"))] = r
    if not reqs:
        log("sin logs de serving: no hay impresiones propias")
        return pd.DataFrame()
    fb = client.query_df(sql.feedback_since(team, 0, limit=5_000_000), timeout=600)
    rows = []
    for r in fb.itertuples(index=False):
        req = reqs.get(str(r.request_id))
        shown = r.shown if isinstance(r.shown, list) else None
        if req is None or not shown:
            continue
        ts = parse_ts(req.get("ts"))
        rows.append({"request_id": r.request_id, "ts_ms": ts * 1000.0, "user_id": req.get("user_id", ""),
                     "context": req.get("context"), "recent_views": req.get("recent_views"),
                     "shown": [str(p) for p in shown], "clicked": str(r.clicked or ""),
                     "purchased": str(r.purchased or ""), "is_new_user": req.get("is_new_user")})
    df = pd.DataFrame(rows)
    log(f"impresiones propias con feedback: {len(df)} (con compra: {int((df['purchased'] != '').sum()) if len(df) else 0})")
    return df


# ----------------------------------------------------------------------------------------------
# Estado offline y filas de entrenamiento
# ----------------------------------------------------------------------------------------------
def offline_state(products: ProductTable, artifacts: Artifacts, daily: DailyPop) -> State:
    """Estado tal como se veía en el pasado: popularidad del día de cada impresión, todo disponible."""
    return State(products.all_available(), artifacts, daily.snapshot)


def impressions_to_rows(state: State, imps: pd.DataFrame, neg_weight: float, max_recent: int = 10,
                        decay: float = 0.7) -> dict[str, np.ndarray]:
    """Una fila por producto mostrado: features en el instante de la impresión + etiquetas."""
    products = state.products
    Xs, ys, yc, ws, gid, pos_all = [], [], [], [], [], []
    for g, r in enumerate(imps.itertuples(index=False)):
        req = r._asdict()
        shown = list(req["shown"]) if req["shown"] is not None else []
        idx, pos = [], []
        for k, pid in enumerate(shown[:10]):
            i = products.index.get(str(pid))
            if i is not None:
                idx.append(i)
                pos.append(k)
        if not idx:
            continue
        sess = make_session(state, req, max_recent, decay)
        idx_a = np.array(idx, dtype=np.int64)
        pos_a = np.array(pos, dtype=np.float32)
        Xs.append(build_matrix(state, sess, idx_a, pos_a))
        bought, clicked = str(req["purchased"] or ""), str(req["clicked"] or "")
        ys.append(np.array([products.ids[i] == bought for i in idx], dtype=np.float32))
        yc.append(np.array([products.ids[i] == clicked for i in idx], dtype=np.float32))
        ws.append(np.full(len(idx), 1.0 if bought else neg_weight, dtype=np.float32))
        gid.append(np.full(len(idx), g, dtype=np.int64))
        pos_all.append(pos_a)
    if not Xs:
        return {"X": np.zeros((0, len(FEATURE_NAMES)), np.float32), "y": np.zeros(0), "yc": np.zeros(0),
                "w": np.zeros(0), "g": np.zeros(0, np.int64), "pos": np.zeros(0)}
    return {"X": np.vstack(Xs), "y": np.concatenate(ys), "yc": np.concatenate(yc),
            "w": np.concatenate(ws), "g": np.concatenate(gid), "pos": np.concatenate(pos_all)}


# ----------------------------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------------------------
def row_metrics(p: np.ndarray, rows: dict[str, np.ndarray]) -> dict[str, float]:
    """Calidad de la probabilidad (AUC, logloss) y del orden dentro de lo mostrado: en las
    impresiones con compra, ¿en qué puesto pone el modelo el producto comprado, frente al orden
    que usó el sistema antiguo?"""
    from sklearn.metrics import log_loss, roc_auc_score

    y, w, g, pos = rows["y"], rows["w"], rows["g"], rows["pos"]
    out: dict[str, float] = {}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p, sample_weight=w))
        out["logloss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=w))
    df = pd.DataFrame({"g": g, "y": y, "p": p, "pos": pos})
    df = df[df.groupby("g")["y"].transform("max") > 0]
    if len(df):
        df["rank_model"] = df.groupby("g")["p"].rank(ascending=False, method="first")
        bought = df[df.y > 0]
        out["mrr_in_shown_model"] = float((1.0 / bought["rank_model"]).mean())
        out["mrr_in_shown_old_order"] = float((1.0 / (bought["pos"] + 1)).mean())
    return out


def evaluate_policies(state: State, imps: pd.DataFrame, scorers: dict[str, Scorer], n_candidates: int = 200,
                      max_imps: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Evaluación de la lista completa (candidatos + orden) en impresiones con compra: ¿está lo
    comprado en nuestro top 10? Sesgada hacia lo que el sistema antiguo mostraba, pero igual para
    todas las políticas, así que sirve para compararlas."""
    from . import candidates as cand_mod

    target = imps[imps["purchased"] != ""]
    if len(target) > max_imps:
        target = target.sample(max_imps, random_state=seed)
    products = state.products
    base_top = [products.index[p] for p in state.artifacts.best_sellers[:10] if p in products.index]
    recs = {name: Recommender(state, sc, n_candidates=n_candidates) for name, sc in scorers.items()}
    first = next(iter(recs.values())) if recs else Recommender(state, None, n_candidates=n_candidates)

    rows = []
    for r in target.itertuples(index=False):
        req = r._asdict()
        t = products.index.get(str(req["purchased"]))
        if t is None:
            continue
        sess = first.session(req)
        row = {"cold": bool(sess.cold or sess.n_recent == 0), "baseline": t in base_top}
        for k in (50, 100, 200):
            row[f"recall@{k}"] = t in set(cand_mod.generate(state, sess, k).tolist())
        for name, rec in recs.items():
            _, ranked, _ = rec.rank(req)
            top = ranked[:10].tolist()
            row[name] = t in top
            row[f"{name}_rr"] = 1.0 / (top.index(t) + 1) if t in top else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["baseline"] + list(recs) + [c for c in df.columns if c.startswith("recall@")]
    summary = pd.DataFrame({
        "todas": df[cols].mean(),
        "cold": df[df.cold][cols].mean() if df.cold.any() else np.nan,
    })
    summary.loc["n_impresiones"] = [len(df), int(df.cold.sum())]
    for name in recs:
        summary.loc[f"{name} MRR@10"] = [df[f"{name}_rr"].mean(),
                                         df[df.cold][f"{name}_rr"].mean() if df.cold.any() else np.nan]
    return summary
