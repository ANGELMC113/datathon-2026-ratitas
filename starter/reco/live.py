"""Estado en vivo y su refresco en segundo plano (rol A).

El hilo que responde peticiones nunca espera a RawTree para nada que no sea leer peticiones y
escribir respuestas: catálogo, stock, popularidad, tendencias, compras y feedback se refrescan en
dos hilos aparte y se sustituyen de golpe (asignar un atributo en Python es atómico).

- Hilo rápido: catálogo/stock (cada 2 s, cada 0,5 s cerca de los múltiplos de 5 min, cuando llegan
  los cambios), tendencias, compras, feedback y reloj.
- Hilo lento: popularidad de las últimas 24 h y 7 días (consulta más pesada).
Si una tarea falla se registra y se reintenta en su siguiente turno; el hilo no muere nunca.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from . import sql
from .artifacts import Artifacts, PopSnapshot, counts_from_rows
from .catalog import ProductTable
from .client import RawTreeClient
from .config import Settings
from .state import BanditStats, State
from .timeutil import to_float

PURCHASE_MEMORY_S = 7 * 86400.0
REQUEST_USER_MEMORY_S = 600.0


class LiveState(State):
    def __init__(self, client: RawTreeClient, artifacts: Artifacts, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._pop: PopSnapshot | None = None
        self._lock = threading.Lock()
        self._user_bought: dict[str, dict[str, float]] = defaultdict(dict)   # user -> {pid: ts}
        self._request_user: dict[str, tuple[str, float]] = {}                # request_id -> (user, t)
        self.clock_offset = 0.0           # reloj del servidor - reloj local (s)
        self.catalog_ms = 0               # último ts aplicado de catalog_updates
        self.purchases_ms = 0
        self.feedback_ms = 0
        self.health: dict[str, float] = {}  # tarea -> último éxito (time.time())
        self.intervals: dict[str, float] = {}  # tarea -> cada cuánto se ejecuta

        products = self._load_catalog(artifacts)
        super().__init__(products, artifacts, lambda ts, p: self._pop_or_empty(p), BanditStats())

    # ------------------------------------------------------------------ interfaz de State
    def pop_at(self, ts: float) -> PopSnapshot:
        return self._pop_or_empty(self.products)

    def _pop_or_empty(self, products: ProductTable) -> PopSnapshot:
        pop = self._pop
        if pop is None:
            pop = self._pop = PopSnapshot.build({"buy_all": self.artifacts.buy_all}, products)
        return pop

    def bought(self, user_id: str) -> set[str]:
        if not user_id:
            return set()
        with self._lock:
            d = self._user_bought.get(user_id)
            return set(d) if d else set()

    def remember_request(self, request_id: str, user_id: str) -> None:
        """Para atribuir las compras del feedback (que no trae user_id) al usuario."""
        with self._lock:
            self._request_user[request_id] = (user_id, time.time())

    def now_server(self) -> float:
        return time.time() + self.clock_offset

    # ------------------------------------------------------------------ arranque
    def _load_catalog(self, artifacts: Artifacts) -> ProductTable:
        df = self.client.query_df(sql.CATALOG, timeout=60)
        products = ProductTable.from_frame(df, emb=artifacts.emb, order=artifacts.product_ids)
        rows = self.client.query(sql.catalog_updates_since(0), timeout=60)
        if rows:
            self.catalog_ms = int(max(to_float(r.get("ts_ms"), 0) for r in rows))
            products = products.apply_updates(rows, artifacts.content)
        print(f"[live] catálogo: {len(products)} productos, {int(products.available.sum())} disponibles, "
              f"{len(rows)} cambios aplicados")
        return products

    def warm_up(self) -> None:
        """Primera carga síncrona de todo (si algo falla, se sigue: hay valores por defecto)."""
        for name, fn in self.tasks_fast() + self.tasks_slow():
            self._run(name, fn[0])

    # ------------------------------------------------------------------ tareas de refresco
    def refresh_clock(self) -> None:
        t0 = time.time()
        rows = self.client.query(sql.CLOCK, timeout=5)
        t1 = time.time()
        server = to_float(rows[0]["now_ms"]) / 1000.0
        self.clock_offset = server - (t0 + t1) / 2.0

    def refresh_catalog(self) -> None:
        rows = self.client.query(sql.catalog_updates_since(self.catalog_ms), timeout=5, retries=0)
        if not rows:
            return
        last = int(max(to_float(r.get("ts_ms"), 0) for r in rows))
        if last == self.catalog_ms and len(rows) <= getattr(self, "_last_catalog_batch", 0):
            return                                   # mismos cambios del mismo instante: nada nuevo
        self._last_catalog_batch = len(rows) if last == self.catalog_ms else 0
        self.products = self.products.apply_updates(rows, self.artifacts.content)
        self.catalog_ms = last

    def refresh_pop(self) -> None:
        products = self.products
        n, index = len(products), products.index
        counts = counts_from_rows(self.client.query(sql.pop_window(24), timeout=60), index, n, "1d")
        counts.update(counts_from_rows(self.client.query(sql.pop_window(24 * 7), timeout=120), index, n, "7d"))
        counts["buy_all"] = self.artifacts.buy_all
        old = self._pop
        self._pop = PopSnapshot.build(counts, products, trending=old.trending if old is not None else None)
        self._update_fallback()

    def refresh_trends(self) -> None:
        products = self.products
        df = self.client.query_df(sql.TRENDS, timeout=20)
        pop = self._pop_or_empty(products)
        if df.empty:
            self._pop = pop.with_trending(np.zeros(0, dtype=np.int64))
            return
        idx = df["product_id"].astype(str).map(products.index)
        ok = idx.notna()
        idx = idx[ok].astype(np.int64).to_numpy()
        v15 = pd.to_numeric(df.loc[ok, "v15"], errors="coerce").fillna(0).to_numpy()
        v60 = pd.to_numeric(df.loc[ok, "v60"], errors="coerce").fillna(0).to_numpy()
        b60 = pd.to_numeric(df.loc[ok, "b60"], errors="coerce").fillna(0).to_numpy()
        share60 = v60 / max(v60.sum(), 1.0) * 1e4
        base = np.zeros(len(idx), dtype=np.float64)
        ok7 = idx < len(pop.shares["view_7d"])
        base[ok7] = pop.shares["view_7d"][idx[ok7]]
        lift = np.minimum((share60 + 1.0) / (base + 1.0), 5.0)            # cuánto más se ve ahora que de normal
        momentum = (4.0 * v15 + 1.0) / (v60 + 4.0)                          # se está acelerando
        score = (v60 + 5.0 * b60) * lift * momentum
        order = idx[np.argsort(-score, kind="stable")][:50]
        self._pop = pop.with_trending(order)

    def refresh_purchases(self) -> None:
        since = self.purchases_ms or int((self.now_server() - PURCHASE_MEMORY_S) * 1000)
        rows: list[dict] = []
        if not self.purchases_ms:                    # al arrancar: también las del histórico reciente
            rows += self.client.query(sql.purchases_since("shop.history_events", since), timeout=60)
        rows += self.client.query(sql.purchases_since("shop.events", since), timeout=20)
        last = since
        with self._lock:
            for r in rows:
                t = to_float(r.get("ts_ms"), 0.0)
                self._user_bought[str(r.get("user_id", ""))][str(r.get("product_id", ""))] = t / 1000.0
                last = max(last, int(t))
        self.purchases_ms = last

    def refresh_feedback(self) -> None:
        since = self.feedback_ms or int((self.now_server() - 6 * 3600) * 1000)
        rows = self.client.query(sql.feedback_since(self.settings.team, since), timeout=30)
        last = since
        for r in rows:
            last = max(last, int(to_float(r.get("ts_ms"), 0)))
            shown = r.get("shown") or []
            if isinstance(shown, str):
                continue
            purchased = str(r.get("purchased") or "")
            if r.get("served_by") == "team":
                self.bandit.update([str(p) for p in shown], purchased)
            if purchased:
                with self._lock:
                    user = self._request_user.get(str(r.get("request_id")), ("", 0.0))[0]
                    if user:
                        self._user_bought[user][purchased] = time.time()
        self.feedback_ms = last
        self._prune()

    def decay_bandit(self) -> None:
        self.bandit.decay(0.9)

    def _prune(self) -> None:
        now = time.time()
        with self._lock:
            for rid in [k for k, (_, t) in self._request_user.items() if now - t > REQUEST_USER_MEMORY_S]:
                del self._request_user[rid]
            horizon = self.now_server() - PURCHASE_MEMORY_S
            for user in list(self._user_bought):
                d = self._user_bought[user]
                for pid in [p for p, t in d.items() if t < horizon]:
                    del d[pid]
                if not d:
                    del self._user_bought[user]

    def _update_fallback(self) -> None:
        pop = self._pop
        if pop is None:
            return
        products = self.products
        ids = [products.ids[i] for i in pop.rank_global if i < len(products)]
        ids += [p for p in self.artifacts.best_sellers if p not in set(ids)]
        self.fallback_ids = ids[:500]

    # ------------------------------------------------------------------ planificación
    def tasks_fast(self) -> list[tuple[str, tuple[Callable[[], None], float]]]:
        s = self.settings
        return [("clock", (self.refresh_clock, s.every_clock)),
                ("catalog", (self.refresh_catalog, s.every_catalog)),
                ("trends", (self.refresh_trends, s.every_trends)),
                ("purchases", (self.refresh_purchases, s.every_purchases)),
                ("feedback", (self.refresh_feedback, s.every_feedback)),
                ("bandit_decay", (self.decay_bandit, 3600.0))]

    def tasks_slow(self) -> list[tuple[str, tuple[Callable[[], None], float]]]:
        return [("pop", (self.refresh_pop, self.settings.every_pop))]

    def _run(self, name: str, fn: Callable[[], None]) -> bool:
        if not self.intervals:
            self.intervals = {n: iv for n, (_, iv) in self.tasks_fast() + self.tasks_slow()}
        try:
            fn()
            self.health[name] = time.time()
            return True
        except Exception as exc:
            print(f"[live] {name} falló: {exc!r}")
            if not isinstance(exc, (OSError, RuntimeError)):
                traceback.print_exc()
            return False

    def start(self) -> None:
        for label, tasks in (("fast", self.tasks_fast()), ("slow", self.tasks_slow())):
            threading.Thread(target=self._loop, args=(tasks,), name=f"refresh-{label}", daemon=True).start()

    def _loop(self, tasks) -> None:
        due = {name: time.time() + interval for name, (_, interval) in tasks}
        while True:
            now = time.time()
            for name, (fn, interval) in tasks:
                if now >= due[name]:
                    self._run(name, fn)
                    due[name] = time.time() + self._next_interval(name, interval)
            time.sleep(0.05)

    def _next_interval(self, name: str, interval: float) -> float:
        """Los cambios de catálogo llegan en múltiplos de 5 minutos: alrededor de esos instantes
        miramos cada 0,5 s para no recomendar un producto recién agotado."""
        if name == "catalog":
            t = self.now_server() % 300.0
            if t > 297.0 or t < 8.0:
                return 0.5
        return interval
