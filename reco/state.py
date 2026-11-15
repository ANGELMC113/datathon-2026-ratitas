"""Estado que necesitan candidatos, features y rerank. La misma interfaz sirve offline (popularidad
del día de cada impresión) y en vivo (LiveState en live.py, popularidad de las últimas horas)."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable

import numpy as np

from .artifacts import Artifacts, PopSnapshot
from .catalog import ProductTable

PopProvider = Callable[[float, ProductTable], PopSnapshot]


class BanditStats:
    """Impresiones y compras por producto en los huecos de exploración (8-10), sacadas del feedback.
    Solo se cuentan esos huecos para que todos los productos se comparen en igualdad de posición."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.shown: dict[str, float] = defaultdict(float)
        self.buys: dict[str, float] = defaultdict(float)

    def update(self, shown: list[str], purchased: str, slots: range = range(7, 10)) -> None:
        with self._lock:
            for pos in slots:
                if pos < len(shown):
                    pid = shown[pos]
                    self.shown[pid] += 1.0
                    if purchased and purchased == pid:
                        self.buys[pid] += 1.0

    def decay(self, factor: float) -> None:
        """Olvida poco a poco lo antiguo para seguir tendencias (p. ej. x0.9 cada hora)."""
        with self._lock:
            for d in (self.shown, self.buys):
                for k in list(d):
                    d[k] *= factor

    def lookup(self, pids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            shown = np.array([self.shown.get(p, 0.0) for p in pids], dtype=np.float64)
            buys = np.array([self.buys.get(p, 0.0) for p in pids], dtype=np.float64)
        return shown, np.minimum(buys, shown)


class State:
    def __init__(self, products: ProductTable, artifacts: Artifacts, pop_provider: PopProvider,
                 bandit: BanditStats | None = None, tt=None) -> None:
        self.products = products
        self.tt = tt                       # reco.tt.TwoTower o None
        self.artifacts = artifacts
        self._pop_provider = pop_provider
        self.bandit = bandit or BanditStats()
        self.fallback_ids: list[str] = list(artifacts.best_sellers[:500])

    def pop_at(self, ts: float) -> PopSnapshot:
        return self._pop_provider(ts, self.products)

    def bought(self, user_id: str) -> set[str]:
        """Productos que el usuario ya compró (no se los volvemos a ofrecer). Offline: desconocido."""
        return set()

    def frozen(self, ts: float) -> "State":
        """Foto del estado para UNA petición: aunque un hilo de refresco cambie el catálogo o la
        popularidad a mitad del cálculo, candidatos, features y rerank ven los mismos datos."""
        return _Frozen(self, ts)

    def fallback_items(self, n: int, exclude: set[str] = frozenset()) -> list[str]:
        """Lista de emergencia: best-sellers recientes disponibles."""
        products = self.products
        out: list[str] = []
        for pid in self.fallback_ids:
            i = products.index.get(pid)
            if i is not None and products.available[i] and pid not in exclude and pid not in out:
                out.append(pid)
                if len(out) == n:
                    return out
        for i in np.flatnonzero(products.available):         # último recurso: cualquiera disponible
            pid = products.ids[i]
            if pid not in exclude and pid not in out:
                out.append(pid)
                if len(out) == n:
                    break
        return out


class _Frozen(State):
    def __init__(self, parent: State, ts: float) -> None:
        self.products = parent.products
        self.artifacts = parent.artifacts
        self.bandit = parent.bandit
        self.tt = parent.tt                # el mismo modelo durante toda la petición
        self.fallback_ids = parent.fallback_ids
        self._parent = parent
        self._pop = parent.pop_at(ts)
        self._pop_provider = lambda t, p: self._pop

    def pop_at(self, ts: float) -> PopSnapshot:
        return self._pop

    def bought(self, user_id: str) -> set[str]:
        return self._parent.bought(user_id)
