"""Catálogo en memoria y embeddings de contenido.

ProductTable es una foto inmutable: cada cambio (precio, stock, producto nuevo) crea una nueva y se
sustituye de golpe, así el hilo que responde nunca ve una tabla a medio actualizar. Los productos
solo se añaden al final, de modo que el índice de un producto no cambia nunca y todos los arrays
precalculados (co-visitación, popularidad...) siguen siendo válidos.

Embeddings: TF-IDF sobre categoría, subcategoría, marca y descripción, reducido con SVD a 64
dimensiones y normalizado. Un producto nuevo tiene vector desde el primer segundo (solo necesita su
texto): es lo que resuelve el cold start de producto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .timeutil import to_float

DEVICE_CODES = {"web": 0, "ios": 1, "android": 2}
PAGE_CODES = {"home": 0, "product": 1, "category": 2}


def product_text(category: str, subcategory: str, brand: str, description: str) -> str:
    return f"{category} {subcategory} {subcategory} {brand} {description}".lower()


@dataclass
class ContentModel:
    """Vectorizador TF-IDF + SVD ajustado con el catálogo; embebe textos nuevos igual."""
    vectorizer: object
    svd: object | None

    def embed(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        if self.svd is not None:
            e = self.svd.transform(x).astype(np.float32)
        else:
            e = x.toarray().astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.maximum(norms, 1e-9)


def fit_content_model(texts: list[str], dim: int = 64, seed: int = 0) -> ContentModel:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    x = vec.fit_transform(texts)
    k = min(dim, x.shape[1] - 1, x.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(x) if k >= 2 else None
    return ContentModel(vec, svd)


def knn(emb: np.ndarray, k: int = 30, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Los k vecinos más parecidos (coseno) de cada producto, sin él mismo. Por bloques para no
    crear la matriz n x n entera."""
    n = emb.shape[0]
    k = max(0, min(k, n - 1))
    idx = np.zeros((n, k), dtype=np.int32)
    sim = np.zeros((n, k), dtype=np.float32)
    if k == 0:
        return idx, sim
    for start in range(0, n, block):
        s = emb[start:start + block] @ emb.T
        rows = np.arange(s.shape[0])
        s[rows, rows + start] = -np.inf
        part = np.argpartition(-s, k, axis=1)[:, :k]
        part_sim = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        idx[start:start + s.shape[0]] = np.take_along_axis(part, order, axis=1)
        sim[start:start + s.shape[0]] = np.take_along_axis(part_sim, order, axis=1)
    return idx, sim


class _Vocab:
    """Texto -> código entero estable (los nuevos valores reciben el siguiente código)."""

    def __init__(self, values: list[str] | None = None) -> None:
        self.codes: dict[str, int] = {}
        for v in values or []:
            self.code(v)

    def code(self, value: str) -> int:
        c = self.codes.get(value)
        if c is None:
            c = self.codes[value] = len(self.codes)
        return c

    def copy(self) -> "_Vocab":
        out = _Vocab()
        out.codes = dict(self.codes)
        return out


@dataclass
class ProductTable:
    ids: list[str]
    index: dict[str, int]
    cat: np.ndarray            # int32, código de categoría
    sub: np.ndarray            # int32, código de subcategoría
    brand: np.ndarray          # int32, código de marca
    price: np.ndarray          # float32
    created: np.ndarray        # float64, epoch s
    active: np.ndarray         # bool, is_active del último registro
    in_stock: np.ndarray       # bool, último out_of_stock / back_in_stock
    live_new: np.ndarray       # bool, llegó como new_product durante el evento
    emb: np.ndarray            # float32 (n, d), normalizado
    vocab_cat: _Vocab = field(default_factory=_Vocab)
    vocab_sub: _Vocab = field(default_factory=_Vocab)
    vocab_brand: _Vocab = field(default_factory=_Vocab)
    available: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.available = self.active & self.in_stock

    def __len__(self) -> int:
        return len(self.ids)

    def sub_names(self) -> list[str]:
        """Código de subcategoría -> nombre (se cachea; la tabla es inmutable)."""
        names = self.__dict__.get("_sub_names")
        if names is None or len(names) != len(self.vocab_sub.codes):
            names = [""] * len(self.vocab_sub.codes)
            for name, code in self.vocab_sub.codes.items():
                names[code] = name
            self.__dict__["_sub_names"] = names
        return names

    @classmethod
    def from_frame(cls, df: pd.DataFrame, emb: np.ndarray | None = None,
                   order: list[str] | None = None) -> "ProductTable":
        """Construye la tabla desde shop.catalog. `order` fija el orden (el de los artefactos
        offline); los productos que no estén en `order` van al final."""
        df = df.drop_duplicates("product_id", keep="last").set_index("product_id", drop=False)
        if order is not None:
            order = [str(p) for p in order]
            order_set = set(order)
            # productos de `order` que ya no están en el catálogo: se conservan, inactivos
            missing = [p for p in order if p not in df.index]
            if missing:
                filler = pd.DataFrame({"product_id": missing, "category": "", "subcategory": "", "brand": "",
                                       "price_eur": 0.0, "description": "", "created_ms": 0,
                                       "is_active": False}).set_index("product_id", drop=False)
                df = pd.concat([df, filler])
            extra = [p for p in df.index if p not in order_set]
            df = df.loc[order + extra]
        vc, vs, vb = _Vocab(), _Vocab(), _Vocab()
        n = len(df)
        if emb is None or emb.shape[0] < n:
            d = emb.shape[1] if emb is not None and emb.ndim == 2 else 1
            full = np.zeros((n, d), dtype=np.float32)
            if emb is not None and emb.shape[0]:
                full[: emb.shape[0]] = emb
            emb = full
        return cls(
            ids=[str(p) for p in df["product_id"]],
            index={str(p): i for i, p in enumerate(df["product_id"])},
            cat=np.array([vc.code(str(v)) for v in df["category"]], dtype=np.int32),
            sub=np.array([vs.code(str(v)) for v in df["subcategory"]], dtype=np.int32),
            brand=np.array([vb.code(str(v)) for v in df["brand"]], dtype=np.int32),
            price=pd.to_numeric(df["price_eur"], errors="coerce").fillna(0).to_numpy(np.float32),
            created=pd.to_numeric(df["created_ms"], errors="coerce").fillna(0).to_numpy(np.float64) / 1000.0,
            active=df["is_active"].map(_to_bool).to_numpy(bool),
            in_stock=np.ones(n, dtype=bool),
            live_new=np.zeros(n, dtype=bool),
            emb=np.ascontiguousarray(emb[:n], dtype=np.float32),
            vocab_cat=vc, vocab_sub=vs, vocab_brand=vb,
        )

    def apply_updates(self, rows: list[dict], content: ContentModel | None,
                      live_since_s: float = math.inf) -> "ProductTable":
        """Aplica shop.catalog_updates en orden de ts y devuelve una tabla NUEVA."""
        if not rows:
            return self
        ids, index = list(self.ids), dict(self.index)
        cat, sub, brand = list(self.cat), list(self.sub), list(self.brand)
        price, created = self.price.copy(), self.created.copy()
        active, in_stock, live_new = self.active.copy(), self.in_stock.copy(), self.live_new.copy()
        vc, vs, vb = self.vocab_cat.copy(), self.vocab_sub.copy(), self.vocab_brand.copy()
        new_texts: list[str] = []
        extra = {"price": [], "created": [], "active": [], "in_stock": [], "live_new": []}

        for r in sorted(rows, key=lambda r: to_float(r.get("ts_ms"), 0.0)):
            pid = str(r.get("product_id", ""))
            if not pid:
                continue
            change = str(r.get("change", ""))
            is_active = _to_bool(r.get("is_active", True))
            p = to_float(r.get("price_eur"), math.nan)
            i = index.get(pid)
            if i is None:                              # producto nuevo: se añade al final
                index[pid] = len(ids)
                ids.append(pid)
                cat.append(vc.code(str(r.get("category", ""))))
                sub.append(vs.code(str(r.get("subcategory", ""))))
                brand.append(vb.code(str(r.get("brand", ""))))
                extra["price"].append(0.0 if math.isnan(p) else p)
                extra["created"].append(to_float(r.get("created_ms"), 0.0) / 1000.0)
                extra["active"].append(is_active)
                extra["in_stock"].append(change != "out_of_stock")
                ts_s = to_float(r.get("ts_ms"), 0.0) / 1000.0
                extra["live_new"].append(change == "new_product" or ts_s >= live_since_s)
                new_texts.append(product_text(str(r.get("category", "")), str(r.get("subcategory", "")),
                                              str(r.get("brand", "")), str(r.get("description", ""))))
                continue
            j = i - len(self.ids)
            if j >= 0:                                 # nuevo en este mismo lote
                if not math.isnan(p):
                    extra["price"][j] = p
                extra["active"][j] = is_active
                if change in ("out_of_stock", "back_in_stock"):
                    extra["in_stock"][j] = change == "back_in_stock"
            else:
                if not math.isnan(p):
                    price[i] = p
                active[i] = is_active
                if change in ("out_of_stock", "back_in_stock"):
                    in_stock[i] = change == "back_in_stock"

        emb = self.emb
        if new_texts:
            d = self.emb.shape[1]
            if content is not None:
                try:
                    new_emb = content.embed(new_texts)
                except Exception:              # sin vector: el producto sigue siendo recomendable
                    new_emb = np.zeros((len(new_texts), d), dtype=np.float32)
            else:
                new_emb = np.zeros((len(new_texts), d), dtype=np.float32)
            if new_emb.shape[1] != d:
                new_emb = np.zeros((len(new_texts), d), dtype=np.float32)
            emb = np.vstack([self.emb, new_emb]).astype(np.float32)
            price = np.concatenate([price, np.array(extra["price"], dtype=np.float32)])
            created = np.concatenate([created, np.array(extra["created"], dtype=np.float64)])
            active = np.concatenate([active, np.array(extra["active"], dtype=bool)])
            in_stock = np.concatenate([in_stock, np.array(extra["in_stock"], dtype=bool)])
            live_new = np.concatenate([live_new, np.array(extra["live_new"], dtype=bool)])

        return ProductTable(ids=ids, index=index, cat=np.array(cat, dtype=np.int32),
                            sub=np.array(sub, dtype=np.int32), brand=np.array(brand, dtype=np.int32),
                            price=price, created=created, active=active, in_stock=in_stock,
                            live_new=live_new, emb=emb, vocab_cat=vc, vocab_sub=vs, vocab_brand=vb)

    def all_available(self) -> "ProductTable":
        """Copia con todo disponible (evaluación offline: el stock de hoy no aplica al pasado)."""
        n = len(self.ids)
        return ProductTable(ids=self.ids, index=self.index, cat=self.cat, sub=self.sub, brand=self.brand,
                            price=self.price, created=self.created, active=np.ones(n, dtype=bool),
                            in_stock=np.ones(n, dtype=bool), live_new=self.live_new, emb=self.emb,
                            vocab_cat=self.vocab_cat, vocab_sub=self.vocab_sub, vocab_brand=self.vocab_brand)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "yes")
