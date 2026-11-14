"""Mundo sintético para probar todo sin RawTree: catálogo, histórico, impresiones y un cliente falso
que responde a las consultas de reco/sql.py con datos coherentes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from reco.timeutil import DAY_MS

DAY0 = 20400                      # día epoch del inicio del histórico sintético


def make_catalog(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = [f"Cat{c}" for c in range(6)]
    adjs = ["vintage", "luxury", "minimalist", "recycled", "wireless", "organic", "bold", "compact"]
    rows = []
    for i in range(n):
        c = int(rng.integers(len(cats)))
        sub = f"Sub{c}_{int(rng.integers(5))}"
        brand = f"Brand{int(rng.integers(25))}"
        rows.append({
            "product_id": f"p_{i:06x}", "category": cats[c], "subcategory": sub, "brand": brand,
            "price_eur": float(np.round(rng.lognormal(3.3, 0.7), 2)),
            "description": f"{brand} {sub} with a {rng.choice(adjs)} look and {rng.choice(adjs)} finish",
            "created_ms": int((DAY0 - rng.integers(0, 400)) * DAY_MS), "is_active": bool(rng.random() > 0.03),
        })
    return pd.DataFrame(rows)


def make_history(catalog: pd.DataFrame, n_imps: int = 6000, days: int = 60, seed: int = 1):
    """Impresiones del carrusel antiguo + frames de co-visitación, carrusel y conteos diarios.
    Regla oculta: se compra sobre todo lo de la subcategoría preferida, y más arriba que abajo."""
    rng = np.random.default_rng(seed)
    ids = catalog["product_id"].tolist()
    sub = catalog["subcategory"].to_numpy()
    by_sub = {s: np.flatnonzero(sub == s) for s in np.unique(sub)}
    subs = list(by_sub)
    pop = rng.zipf(1.6, len(ids)).astype(float)
    pop /= pop.sum()
    best = np.argsort(-pop)[:40]

    imps = []
    for k in range(n_imps):
        day = DAY0 + int(rng.integers(0, days))
        ts_ms = day * DAY_MS + int(rng.integers(0, DAY_MS))
        pref = subs[int(rng.integers(len(subs)))]
        nv = int(rng.integers(0, 6))
        views = [ids[int(rng.choice(by_sub[pref]))] for _ in range(nv)]
        rv = [{"product_id": p, "ts": pd.Timestamp(ts_ms - 60_000 * (nv - j), unit="ms").strftime("%Y-%m-%d %H:%M:%S.%f")}
              for j, p in enumerate(views)]
        shown_idx = list(rng.choice(best, 6, replace=False)) + list(rng.choice(len(ids), 4, replace=False))
        shown_idx = list(dict.fromkeys(int(i) for i in shown_idx))[:10]
        rng.shuffle(shown_idx)
        purchased = clicked = ""
        for pos, i in enumerate(shown_idx):
            p = 0.004 * (12.0 if sub[i] == pref and nv else 1.0) * (1.0 / (1 + 0.4 * pos)) * (1 + 50 * pop[i])
            if rng.random() < p:
                purchased = clicked = ids[i]
                break
            if not clicked and rng.random() < 3 * p:
                clicked = ids[i]
        imps.append({"request_id": f"r_{k:08x}", "ts_ms": ts_ms, "user_id": f"u_{int(rng.integers(3000)):05x}",
                     "context": {"device": rng.choice(["web", "ios", "android"]),
                                 "page": rng.choice(["home", "product", "category"]), "region": "Catalunya"},
                     "recent_views": rv, "shown": [ids[i] for i in shown_idx],
                     "clicked": clicked, "purchased": purchased})
    imps = pd.DataFrame(imps)

    # co-visitación: productos de la misma subcategoría se ven juntos
    pairs = []
    for s, members in by_sub.items():
        for a in members:
            for b in rng.choice(members, min(15, len(members)), replace=False):
                if a != b:
                    pairs.append({"p": ids[a], "q": ids[b], "n": int(rng.integers(1, 50))})
    covis = pd.DataFrame(pairs)

    car = []
    for r in imps.itertuples():
        for pos, p in enumerate(r.shown, start=1):
            car.append({"product_id": p, "pos": pos, "shown": 1, "clicks": int(r.clicked == p), "buys": int(r.purchased == p)})
    car = pd.DataFrame(car).groupby(["product_id", "pos"], as_index=False).sum()

    daily = {}
    for event, scale in (("view", 200), ("add_to_cart", 20), ("purchase", 5)):
        rows = []
        for d in range(days + 1):
            counts = rng.poisson(pop * scale * 30)
            for i in np.flatnonzero(counts):
                rows.append({"day": DAY0 + d, "product_id": ids[i], "n": int(counts[i])})
        daily[event] = pd.DataFrame(rows)
    sales = daily["purchase"].groupby("product_id")["n"].sum()
    return imps, covis, car, daily, sales


class FakeClient:
    """Responde a las consultas de reco/sql.py reconociendo su texto. Guarda los inserts."""

    def __init__(self, catalog: pd.DataFrame, sales: pd.Series, team: str = "team_99",
                 history: tuple | None = None) -> None:
        self.team = team
        self.history = history          # (imps, covis, car, daily) para las consultas offline
        self.catalog = catalog
        self.sales = sales
        self.updates: list[dict] = []
        self.requests: list[dict] = []
        self.feedback: list[dict] = []
        self.inserted: list[dict] = []
        self.fail_inserts = 0
        self.insert_delays: list[float] = []   # segundos de espera de los próximos inserts
        self.query_delays: list[float] = []    # lo mismo para las consultas de peticiones
        self.queries: list[str] = []
        self._lock = __import__("threading").Lock()

    def query(self, q: str, **kw) -> list[dict]:
        self.queries.append(q)
        now_ms = int(time.time() * 1000)
        if self.history is not None:
            out = self._offline(q)
            if out is not None:
                return out
        if "AS now_ms" in q:
            return [{"now_ms": str(now_ms)}]
        if "FROM shop.catalog_updates" in q:
            return [dict(u) for u in self.updates]
        if "FROM shop.catalog" in q:
            return self.catalog.to_dict("records")
        if "FROM shop.requests" in q:
            with self._lock:
                d = self.query_delays.pop(0) if self.query_delays else 0.0
            time.sleep(d)
            return list(self.requests)
        if "UNION ALL" in q:
            return [{"product_id": p, "event_type": "purchase", "n": str(int(n))} for p, n in self.sales.items()]
        if "AS v15" in q:
            return [{"product_id": p, "v15": "3", "v60": "10", "b60": "1"} for p in list(self.sales.index[:20])]
        if ".feedback" in q:
            return list(self.feedback)
        if "event_type = 'purchase' AND" in q and "user_id::String" in q:
            return []
        if "count() AS n" in q:
            return [{"product_id": p, "n": str(int(n))} for p, n in self.sales.items()]
        raise AssertionError(f"consulta no prevista: {q[:200]}")

    def _offline(self, q: str):
        import re
        imps, covis, car, daily = self.history
        if "min(toUnixTimestamp64Milli" in q:
            return [{"t0": str(int(imps.ts_ms.min())), "t1": str(int(imps.ts_ms.max()))}]
        if "intDiv(" in q:
            if "shop.events" in q:
                return []
            event = re.search(r"event_type = '(\w+)'", q).group(1)
            return daily[event].to_dict("records")
        if "INNER JOIN" in q:
            return covis.to_dict("records")
        if "ARRAY JOIN" in q:
            return car.to_dict("records")
        if "cityHash64" in q:
            lo, hi = map(int, re.search(r">= (\d+) AND \S+ < (\d+)", q).groups())
            sel = imps[(imps.ts_ms >= lo) & (imps.ts_ms < hi)]
            return sel.to_dict("records")
        return None

    def query_df(self, q: str, **kw) -> pd.DataFrame:
        return pd.DataFrame(self.query(q, **kw))

    def insert(self, table: str, rows: list[dict], **kw) -> None:
        with self._lock:
            d = self.insert_delays.pop(0) if self.insert_delays else 0.0
        time.sleep(d)
        if self.fail_inserts > 0:
            self.fail_inserts -= 1
            raise RuntimeError("fallo simulado")
        assert table == "submissions"
        with self._lock:
            self.inserted.extend(rows)
