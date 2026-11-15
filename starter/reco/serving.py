"""Piezas del servicio en vivo que no son la lógica de recomendar (rol A):

- Stats: contadores y latencias que se imprimen cada 30 s.
- ServeLog: una línea JSON por respuesta (para reentrenar con el feedback y analizar el cold start).
- Pusher: inserta las respuestas en lotes, en paralelo y con copia de respaldo.
- Arm / parse_arm / pick_arm: variantes del test A/B y reparto estable de peticiones.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .client import RawTreeClient


class Stats:
    def __init__(self, every: float = 30.0) -> None:
        self.lock = threading.Lock()
        self.every = every
        self.last = time.time()
        self.c: Counter = Counter()
        self.compute_ms: list[float] = []
        self.age_ms: list[float] = []

    def add(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.c[key] += n

    def sample(self, compute_ms: float | None = None, age_ms: float | None = None) -> None:
        with self.lock:
            if compute_ms is not None:
                self.compute_ms.append(compute_ms)
            if age_ms is not None:
                self.age_ms.append(age_ms)

    def maybe_print(self, extra: str = "") -> None:
        if time.time() - self.last < self.every:
            return
        with self.lock:
            c, cm, am = self.c, self.compute_ms, self.age_ms
            self.c, self.compute_ms, self.age_ms = Counter(), [], []
            self.last = time.time()

        arms = {k[4:]: v for k, v in c.items() if k.startswith("arm:")}
        ab = ("modelos=" + ",".join(f"{k}:{v}" for k, v in sorted(arms.items())) + " ") if len(arms) > 1 else ""

        def pct(a: list[float], q: float) -> str:
            return f"{np.percentile(a, q):.0f}" if a else "-"

        print(f"[serve {datetime.now():%H:%M:%S}] respondidas={c['answered']} modelo={c['model']} "
              f"fallback={c['fallback']} ({', '.join(f'{k[3:]}={v}' for k, v in c.items() if k.startswith('fb_')) or '-'}) "
              f"viejas={c['dropped']} insertadas={c['pushed']} fallo_insert={c['push_failed']} "
              f"respaldos={c['hedged']} lecturas_fallidas={c['pull_failed']} "
              f"{ab}"
              f"cálculo p50/p95={pct(cm, 50)}/{pct(cm, 95)} ms  edad al insertar p50/p95={pct(am, 50)}/{pct(am, 95)} ms"
              f"{extra}", flush=True)


class ServeLog:
    """Una línea JSON por respuesta: permite reentrenar con el feedback (train.py --with-feedback)
    y analizar el cold start, porque el feedback no trae la petición original."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._day = ""
        self._f = None

    def write(self, record: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if day != self._day or self._f is None:
            if self._f:
                self._f.close()
            self._f = open(self.directory / f"served-{day}.jsonl", "a", buffering=1 << 16)
            self._day = day
        self._f.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if self._f:
            self._f.flush()


class Pusher(threading.Thread):
    """Inserta lotes en team_NN.submissions sin que un lote lento bloquee a los demás.

    - Cada lote se inserta en su propio hilo (varios a la vez).
    - Inserción con respaldo: si RawTree no responde en `hedge_after` s, se envía una copia idéntica
      por otra conexión y vale la primera que conteste. Una copia repetida no hace daño: la
      organización cuenta una sola respuesta por petición y las dos son iguales.
    - Tras un fallo solo se reintentan las respuestas que aún pueden llegar a tiempo.
    """

    def __init__(self, client: RawTreeClient, state, stats: Stats, timeout: float,
                 dry_run: bool, hedge_after: float = 0.6, workers: int = 8) -> None:
        super().__init__(name="pusher", daemon=True)
        self.client, self.state, self.stats = client, state, stats
        self.timeout, self.dry_run, self.hedge_after = timeout, dry_run, hedge_after
        self.q: queue.Queue = queue.Queue()
        # dos grupos de hilos: uno por lote y otro para las llamadas HTTP (si compartieran grupo, los
        # lotes podrían ocupar todos los hilos esperando inserciones que nunca llegan a ejecutarse)
        self.batches = ThreadPoolExecutor(max_workers=4, thread_name_prefix="push-batch")
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="push-http")

    def submit(self, items: list[tuple[dict, float, float]]) -> None:
        """items: (respuesta, plazo en reloj local, ts de la petición en reloj del servidor)."""
        self.q.put(items)

    def run(self) -> None:
        while True:
            batch = list(self.q.get())
            while True:                                   # junta todo lo que esté en cola
                try:
                    batch += self.q.get_nowait()
                except queue.Empty:
                    break
            self.batches.submit(self._push_safe, batch)

    def _push_safe(self, batch: list[tuple[dict, float, float]]) -> None:
        try:
            self._push(batch)
        except Exception:                                 # un hilo de inserción no puede tirar nada
            traceback.print_exc()

    def _insert(self, rows: list[dict]) -> None:
        """Una inserción con copia de respaldo si la primera tarda."""
        if self.dry_run:
            return
        first = self.pool.submit(self.client.insert, "submissions", rows, timeout=self.timeout, retries=0)
        done, _ = wait([first], timeout=self.hedge_after)
        if done:
            first.result()                                # propaga el error si falló
            return
        self.stats.add("hedged")
        second = self.pool.submit(self.client.insert, "submissions", rows, timeout=self.timeout, retries=0)
        last: BaseException | None = None
        try:
            for f in as_completed([first, second], timeout=self.timeout + 0.2):
                try:
                    f.result()
                    return                                # la primera que llegue vale
                except Exception as exc:
                    last = exc
        except FuturesTimeout as exc:
            last = exc
        raise last if last is not None else RuntimeError("inserción sin respuesta")

    def _push(self, batch: list[tuple[dict, float, float]]) -> None:
        attempt = 0
        while batch:
            if attempt > 0:
                # en los reintentos solo las que aún pueden llegar (1 s de margen por si el reloj miente)
                now = time.time()
                batch = [b for b in batch if b[1] + 1.0 > now]
                if not batch:
                    self.stats.add("push_late")
                    return
            try:
                self._insert([b[0] for b in batch])
            except Exception as exc:
                attempt += 1
                print(f"[push] intento {attempt} falló ({len(batch)} respuestas): {str(exc)[:160]}", flush=True)
                if attempt >= 4:
                    self.stats.add("push_failed", len(batch))
                    return
                time.sleep(0.05 * attempt)
                continue
            done = self.state.now_server()
            self.stats.add("pushed", len(batch))
            for _, _, req_ts in batch:
                self.stats.sample(age_ms=(done - req_ts) * 1000.0)
            return


# ----------------------------------------------------------------------------------------------
def bucket(request_id: str) -> float:
    """Número estable en [0, 1) a partir del request_id: la misma petición, el mismo brazo."""
    return (zlib.crc32(request_id.encode()) % 10000) / 10000.0


class Arm:
    """Una variante del A/B: un modelo (recargable en caliente) y cómo se ordena la lista final."""

    def __init__(self, watcher, fraction: float, n_explore: int | None = None,
                 max_per_sub: int | None = None, old_candidates: bool = False,
                 path: Path | None = None, rebuy: bool = False, adapt: bool = False) -> None:
        self.watcher, self.fraction, self.path = watcher, fraction, path
        self.n_explore, self.max_per_sub = n_explore, max_per_sub
        self.old_candidates = old_candidates
        self.rebuy = rebuy                # no excluir lo ya comprado por el usuario
        self.adapt = adapt                # corregir el score con el lift aprendido del feedback

    def suffix(self) -> str:
        """Lo que distingue la variante en model_version (p. ej. v2-lgbm+div2+noexp)."""
        out = ""
        if self.max_per_sub:
            out += f"+div{self.max_per_sub}"
        if self.n_explore == 0:
            out += "+noexp"
        if self.old_candidates:
            out += "+oldcand"
        if self.rebuy:
            out += "+rebuy"
        if self.adapt:
            out += "+adapt"
        return out


def parse_arm(spec: str, fraction: float) -> "Arm":
    """'artifacts/model_b' o 'artifacts/model:div=2,noexp' -> Arm (el watcher se crea fuera)."""
    path, _, opts = spec.partition(":")
    n_explore, max_per_sub, old, rebuy, adapt = None, None, False, False, False
    for opt in filter(None, (o.strip() for o in opts.split(","))):
        if opt == "noexp":
            n_explore = 0
        elif opt == "oldcand":
            old = True
        elif opt == "rebuy":
            rebuy = True
        elif opt == "adapt":
            adapt = True
        elif opt.startswith("div="):
            max_per_sub = int(opt[4:])
        elif opt.startswith("explore="):
            n_explore = int(opt[8:])
        else:
            raise SystemExit(f"opción desconocida en --ab-model-dir: {opt!r} (usa div=N, noexp, explore=N, oldcand, rebuy, adapt)")
    return Arm(None, fraction, n_explore, max_per_sub, old, Path(path), rebuy, adapt)


def pick_arm(request_id: str, fractions: list[float]) -> int:
    """Índice del brazo extra (0..k-1) o -1 para el modelo principal (A)."""
    u, acc = bucket(request_id), 0.0
    for k, f in enumerate(fractions):
        acc += f
        if u < acc:
            return k
    return -1


