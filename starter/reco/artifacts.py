"""Datos precalculados offline (co-visitación, vecinos de contenido, estadísticas del carrusel)
y popularidad (por día para entrenar sin fuga de futuro, o por ventana móvil en vivo)."""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import ContentModel, ProductTable
from .timeutil import DAY_S


def gather(arr: np.ndarray, idx: np.ndarray, fill: float = math.nan) -> np.ndarray:
    """arr[idx] tolerando índices fuera de rango (productos añadidos después de calcular arr)."""
    out = np.full(len(idx), fill, dtype=np.float32)
    ok = idx < len(arr)
    out[ok] = arr[idx[ok]]
    return out


# ----------------------------------------------------------------------------------------------
# Vecinos (co-visitación y contenido) en formato CSR: compacto y rápido de consultar
# ----------------------------------------------------------------------------------------------
@dataclass
class Neighbors:
    indptr: np.ndarray    # int64 (n+1)
    indices: np.ndarray   # int32
    weights: np.ndarray   # float32

    @classmethod
    def empty(cls, n: int) -> "Neighbors":
        return cls(np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))

    @classmethod
    def from_frame(cls, df: pd.DataFrame, index: dict[str, int], n: int, top_k: int,
                   normalize: bool = True) -> "Neighbors":
        """df con columnas p, q, n. El peso es n / suma de n del producto origen: así las
        matrices calculadas con 40 o con 60 días de histórico tienen la misma escala."""
        if df is None or df.empty:
            return cls.empty(n)
        d = pd.DataFrame({"p": df["p"].astype(str).map(index), "q": df["q"].astype(str).map(index),
                          "n": pd.to_numeric(df["n"], errors="coerce")}).dropna()
        d = d[d.p != d.q]
        d["p"] = d["p"].astype(np.int64)
        d["q"] = d["q"].astype(np.int64)
        d = d.sort_values(["p", "n"], ascending=[True, False])
        d = d.groupby("p", sort=False).head(top_k)
        d = d[(d.p < n) & (d.q < n)]
        if normalize:
            d["n"] = d["n"] / d.groupby("p")["n"].transform("sum")
        counts = np.bincount(d["p"].to_numpy(), minlength=n)[:n]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(indptr, d["q"].to_numpy(np.int32), d["n"].to_numpy(np.float32))

    @classmethod
    def from_dense(cls, idx: np.ndarray, w: np.ndarray) -> "Neighbors":
        n, k = idx.shape
        return cls(np.arange(0, n * k + 1, k, dtype=np.int64), idx.reshape(-1).astype(np.int32),
                   w.reshape(-1).astype(np.float32))

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray] | None:
        if i < 0 or i + 1 >= len(self.indptr):
            return None
        a, b = self.indptr[i], self.indptr[i + 1]
        if a == b:
            return None
        return self.indices[a:b], self.weights[a:b]


# ----------------------------------------------------------------------------------------------
# Carrusel antiguo: rendimiento de cada producto corregido por posición
# ----------------------------------------------------------------------------------------------
@dataclass
class CarouselStats:
    shown_log: np.ndarray    # log(1 + impresiones)
    click_lift: np.ndarray   # clics reales / clics esperados por sus posiciones (1 = normal)
    buy_lift: np.ndarray     # lo mismo con compras

    @classmethod
    def empty(cls, n: int) -> "CarouselStats":
        nan = np.full(n, np.nan, dtype=np.float32)
        return cls(np.zeros(n, dtype=np.float32), nan.copy(), nan.copy())

    @classmethod
    def from_frame(cls, df: pd.DataFrame, index: dict[str, int], n: int) -> "CarouselStats":
        if df is None or df.empty:
            return cls.empty(n)
        d = df.copy()
        for c in ("pos", "shown", "clicks", "buys"):
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
        slot = d.groupby("pos")[["shown", "clicks", "buys"]].sum()
        click_rate = (slot["clicks"] / slot["shown"].clip(lower=1)).rename("cr")
        buy_rate = (slot["buys"] / slot["shown"].clip(lower=1)).rename("br")
        d = d.join(click_rate, on="pos").join(buy_rate, on="pos")
        d["exp_c"] = d["shown"] * d["cr"]
        d["exp_b"] = d["shown"] * d["br"]
        per = d.groupby(d["product_id"].astype(str))[["shown", "clicks", "buys", "exp_c", "exp_b"]].sum()
        per = per[per.index.isin(index.keys())]
        idx = per.index.map(index).to_numpy(np.int64)
        out = cls.empty(n)
        out.shown_log[idx] = np.log1p(per["shown"].to_numpy(np.float32))
        # suavizado bayesiano: con pocos datos el lift se acerca a 1 (producto "normal")
        out.click_lift[idx] = ((per["clicks"] + 5.0) / (per["exp_c"] + 5.0)).to_numpy(np.float32)
        out.buy_lift[idx] = ((per["buys"] + 2.0) / (per["exp_b"] + 2.0)).to_numpy(np.float32)
        return out


# ----------------------------------------------------------------------------------------------
# Popularidad
# ----------------------------------------------------------------------------------------------
EVENTS = {"view": "view", "add_to_cart": "cart", "purchase": "buy"}
POP_KEYS = ("view_1d", "cart_1d", "buy_1d", "view_7d", "cart_7d", "buy_7d", "buy_all")


@dataclass
class PopSnapshot:
    counts: dict[str, np.ndarray]      # eventos por producto en cada ventana
    shares: dict[str, np.ndarray]      # cuota por 10.000 eventos de la ventana (independiente del volumen)
    rank_global: np.ndarray            # productos más vendidos ahora mismo
    rank_by_cat: dict[int, np.ndarray]
    trending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def build(cls, counts: dict[str, np.ndarray], products: ProductTable,
              trending: np.ndarray | None = None, top_global: int = 300, top_cat: int = 60) -> "PopSnapshot":
        n = len(products)
        full: dict[str, np.ndarray] = {}
        for k in POP_KEYS:
            a = np.zeros(n, dtype=np.float32)
            src = counts.get(k)
            if src is not None:
                m = min(n, len(src))
                a[:m] = src[:m]
            full[k] = a
        shares = {k: v / max(float(v.sum()), 1.0) * 1e4 for k, v in full.items()}
        blend = (shares["buy_1d"] + 0.5 * shares["buy_7d"] + 0.1 * shares["buy_all"]
                 + 0.02 * shares["view_1d"] + 0.01 * shares["cart_1d"])
        order = np.argsort(-blend, kind="stable")
        rank_by_cat: dict[int, np.ndarray] = {}
        cats = products.cat[order]
        for c in np.unique(products.cat):
            rank_by_cat[int(c)] = order[cats == c][:top_cat]
        tr = trending if trending is not None else np.zeros(0, dtype=np.int64)
        return cls(full, shares, order[:top_global], rank_by_cat, tr)

    def with_trending(self, trending: np.ndarray) -> "PopSnapshot":
        return PopSnapshot(self.counts, self.shares, self.rank_global, self.rank_by_cat, trending)


class DailyPop:
    """Eventos por día y producto. snapshot(ts) da la popularidad que se conocía ANTES del día de
    `ts` (ayer, últimos 7 días, todo lo anterior): sin fuga de futuro al entrenar."""

    def __init__(self, day0: int, cum: dict[str, np.ndarray], product_ids: list[str]) -> None:
        self.day0 = day0
        self.cum = cum                 # evento -> (n_days + 1, n_products), suma de los días < d
        self.product_ids = product_ids
        self._cache: dict[int, PopSnapshot] = {}

    @property
    def n_days(self) -> int:
        return next(iter(self.cum.values())).shape[0] - 1

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], product_ids: list[str]) -> "DailyPop":
        index = {p: i for i, p in enumerate(product_ids)}
        days = [pd.to_numeric(f["day"], errors="coerce").dropna() for f in frames.values() if not f.empty]
        if not days:
            raise ValueError("sin datos diarios de popularidad")
        day0 = int(min(d.min() for d in days))
        day1 = int(max(d.max() for d in days))
        n_days, n = day1 - day0 + 1, len(product_ids)
        cum: dict[str, np.ndarray] = {}
        for event, short in EVENTS.items():
            mat = np.zeros((n_days, n), dtype=np.float32)
            f = frames.get(event)
            if f is not None and not f.empty:
                d = pd.to_numeric(f["day"], errors="coerce")
                p = f["product_id"].astype(str).map(index)
                v = pd.to_numeric(f["n"], errors="coerce")
                ok = d.notna() & p.notna() & v.notna()
                np.add.at(mat, ((d[ok].astype(int) - day0).to_numpy(), p[ok].astype(int).to_numpy()),
                          v[ok].to_numpy(np.float32))
            c = np.zeros((n_days + 1, n), dtype=np.float32)
            np.cumsum(mat, axis=0, out=c[1:])
            cum[short] = c
        return cls(day0, cum, product_ids)

    def counts_before_day(self, day: int) -> dict[str, np.ndarray]:
        d = int(np.clip(day - self.day0, 0, self.n_days))
        out: dict[str, np.ndarray] = {}
        for short, c in self.cum.items():
            out[f"{short}_1d"] = c[d] - c[max(d - 1, 0)]
            out[f"{short}_7d"] = c[d] - c[max(d - 7, 0)]
        out["buy_all"] = self.cum["buy"][d]
        return out

    def snapshot(self, ts: float, products: ProductTable) -> PopSnapshot:
        day = int(ts // DAY_S) if ts == ts else self.day0 + self.n_days
        snap = self._cache.get(day)
        if snap is None:
            if len(self._cache) > 200:
                self._cache.clear()
            snap = self._cache[day] = PopSnapshot.build(self.counts_before_day(day), products)
        return snap


def counts_from_rows(rows: list[dict], index: dict[str, int], n: int, suffix: str) -> dict[str, np.ndarray]:
    """Filas (product_id, event_type, n) de pop_window() -> arrays {view_1d, cart_1d, buy_1d}."""
    out = {f"{s}_{suffix}": np.zeros(n, dtype=np.float32) for s in EVENTS.values()}
    for r in rows:
        i = index.get(str(r.get("product_id", "")))
        short = EVENTS.get(str(r.get("event_type", "")))
        if i is None or short is None or i >= n:
            continue
        try:
            out[f"{short}_{suffix}"][i] += float(r.get("n", 0))
        except (TypeError, ValueError):
            pass
    return out


# ----------------------------------------------------------------------------------------------
# Paquete de artefactos de un corte temporal
# ----------------------------------------------------------------------------------------------
@dataclass
class Artifacts:
    product_ids: list[str]
    until_ms: int
    covis: dict[str, Neighbors]
    content_knn: Neighbors
    carousel: CarouselStats
    content: ContentModel | None
    emb: np.ndarray
    buy_all: np.ndarray
    best_sellers: list[str]
    meta: dict = field(default_factory=dict)
    # transiciones "vista -> compra en el carrusel" (None/{} en artefactos antiguos)
    trans_prod: "Neighbors | None" = None              # producto visto -> producto comprado
    trans_sub: dict = field(default_factory=dict)      # subcategoría vista -> {subcategoría comprada: tasa}

    @classmethod
    def empty(cls, product_ids: list[str], emb: np.ndarray | None = None) -> "Artifacts":
        n = len(product_ids)
        e = emb if emb is not None else np.zeros((n, 1), dtype=np.float32)
        return cls(product_ids, 0, {k: Neighbors.empty(n) for k in ("view", "buy", "b2b")},
                   Neighbors.empty(n), CarouselStats.empty(n), None, e,
                   np.zeros(n, dtype=np.float32), [], {"degraded": True})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)          # escritura atómica: el servicio nunca lee un fichero a medias

    @staticmethod
    def load(path: Path) -> "Artifacts":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Artifacts):
            raise TypeError(f"{path} no contiene Artifacts")
        return obj
