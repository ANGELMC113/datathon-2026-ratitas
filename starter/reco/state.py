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


class LiftStats:
    """Corrección por producto de lo que el modelo no sabe, aprendida del feedback en vivo.

    Por cada producto que mostramos guardamos cuántas compras esperaba el modelo (su probabilidad,
    por posición) y cuántas hubo. lift = (compras + alpha) / (esperadas + alpha): 1 si no hay datos,
    >1 si se vende más de lo que el modelo cree (tendencia nueva, producto nuevo), <1 si menos.
    El sesgo de posición se corrige con un factor por hueco (compras / esperadas en ese hueco).
    alpha = cuántas compras esperadas de evidencia hacen falta para fiarse a medias."""

    N_POS = 10

    def __init__(self, alpha: float = 3.0, lo: float = 0.33, hi: float = 3.0) -> None:
        self._lock = threading.Lock()
        self.alpha, self.lo, self.hi = alpha, lo, hi
        self.exp: dict[str, np.ndarray] = {}            # pid -> esperadas por hueco (sin corregir)
        self.obs: dict[str, float] = defaultdict(float)
        self.pos_exp = np.zeros(self.N_POS)
        self.pos_buys = np.zeros(self.N_POS)

    def update(self, shown: list[str], preds: dict[str, float], purchased: str) -> None:
        with self._lock:
            for k, pid in enumerate(shown[: self.N_POS]):
                p = preds.get(pid)
                if p is None or not np.isfinite(p):
                    continue
                v = self.exp.get(pid)
                if v is None:
                    v = self.exp[pid] = np.zeros(self.N_POS)
                v[k] += p
                self.pos_exp[k] += p
                if purchased and purchased == pid:
                    self.obs[pid] += 1.0
                    self.pos_buys[k] += 1.0

    def decay(self, factor: float) -> None:
        with self._lock:
            for v in self.exp.values():
                v *= factor
            for k in list(self.obs):
                self.obs[k] *= factor
            self.pos_exp *= factor
            self.pos_buys *= factor

    def lookup(self, pids: list[str]) -> np.ndarray:
        out = np.ones(len(pids))
        with self._lock:
            if not self.exp:
                return out
            # factor por hueco, con prior hacia 1 (10 compras esperadas de evidencia)
            c = (self.pos_buys + 10.0) / (self.pos_exp + 10.0)
            for j, pid in enumerate(pids):
                v = self.exp.get(pid)
                if v is not None:
                    e = float(v @ c)
                    out[j] = (self.obs.get(pid, 0.0) + self.alpha) / (e + self.alpha)
        return np.clip(out, self.lo, self.hi)


class State:
    def __init__(self, products: ProductTable, artifacts: Artifacts, pop_provider: PopProvider,
                 bandit: BanditStats | None = None) -> None:
        self.products = products
        self.artifacts = artifacts
        self._pop_provider = pop_provider
        self.bandit = bandit or BanditStats()
        self.lift = LiftStats()
        self.fallback_ids: list[str] = list(artifacts.best_sellers[:500])

    def pop_at(self, ts: float) -> PopSnapshot:
        return self._pop_provider(ts, self.products)

    def bought(self, user_id: str) -> set[str]:
        """Productos que el usuario ya compró (no se los volvemos a ofrecer). Offline: desconocido."""
        return set()

    def remember_preds(self, request_id: str, preds: dict) -> None:
        """En vivo guarda las predicciones para aprender el lift; offline no hace nada."""

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
        self.lift = parent.lift
        self.fallback_ids = parent.fallback_ids
        self._parent = parent
        self._pop = parent.pop_at(ts)
        self._pop_provider = lambda t, p: self._pop

    def pop_at(self, ts: float) -> PopSnapshot:
        return self._pop

    def bought(self, user_id: str) -> set[str]:
        return self._parent.bought(user_id)
