"""Servicio en vivo (rol A): pull -> recommend -> push, sin parar nunca.

    python serve_v2.py               # responde de verdad
    python serve_v2.py --dry-run     # calcula pero no inserta (para probar contra el stream real)

Garantías:
- Cada petición recibe 10 IDs distintos y disponibles: si el modelo falla, tarda o devuelve algo
  inválido, respondemos con nuestra lista de emergencia (best-sellers recientes con stock), no con la
  de la organización.
- El cálculo de un lote tiene un presupuesto (0,6 s): lo que no quepa va con la lista de emergencia.
- Las respuestas se insertan desde otro hilo, así el bucle vuelve a leer peticiones enseguida; si la
  inserción falla se reintenta mientras quede plazo para cada respuesta.
- Catálogo, stock, popularidad, feedback y modelo se actualizan en segundo plano y en caliente.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reco import config, sql
from reco.artifacts import Artifacts
from reco.client import RawTreeClient
from reco.live import LiveState
from reco.model import ModelWatcher
from reco.recommender import Recommender, is_valid, repair
from reco.session import request_ts
from reco.timeutil import to_float


# ----------------------------------------------------------------------------------------------
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

        def pct(a: list[float], q: float) -> str:
            return f"{np.percentile(a, q):.0f}" if a else "-"

        print(f"[serve {datetime.now():%H:%M:%S}] respondidas={c['answered']} modelo={c['model']} "
              f"fallback={c['fallback']} ({', '.join(f'{k[3:]}={v}' for k, v in c.items() if k.startswith('fb_')) or '-'}) "
              f"viejas={c['dropped']} insertadas={c['pushed']} fallo_insert={c['push_failed']} "
              f"respaldos={c['hedged']} lecturas_fallidas={c['pull_failed']} "
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

    def __init__(self, client: RawTreeClient, state: LiveState, stats: Stats, timeout: float,
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
class Server:
    def __init__(self, client: RawTreeClient, state: LiveState, recommender: Recommender,
                 watcher: ModelWatcher | None, settings: config.Settings, *, dry_run: bool = False,
                 log: ServeLog | None = None) -> None:
        self.client, self.state, self.rec, self.watcher = client, state, recommender, watcher
        self.s = settings
        self.stats = Stats()
        self.log = log
        self.seen: dict[str, float] = {}
        self.pusher = Pusher(client, state, self.stats, settings.push_timeout_s, dry_run,
                             hedge_after=settings.hedge_after_s)
        self._pull_failures = 0
        self._plan_b_until = 0.0
        self._last_prune = time.time()
        self._last_trace = 0.0

    # ---------------------------------------------------------------- lectura
    def fetch(self) -> list[dict]:
        """Una lectura de shop.requests (se ejecuta en hilos: varias pueden ir solapadas)."""
        use_plan_b = time.time() < self._plan_b_until
        q = sql.requests_latest(60) if use_plan_b else sql.requests_recent(self.s.request_window_s)
        try:
            rows = self.client.query(q, timeout=self.s.pull_timeout_s, retries=0)
            self._pull_failures = 0
            return rows
        except Exception as exc:
            self._pull_failures += 1
            self.stats.add("pull_failed")
            if self._pull_failures in (1, 3) or self._pull_failures % 20 == 0:
                print(f"[pull] falló ({self._pull_failures} seguidas): {str(exc)[:160]}", flush=True)
            if self._pull_failures >= 6 and not use_plan_b:
                print("[pull] paso 60 s a la consulta del starter", flush=True)
                self._plan_b_until = time.time() + 60.0
            return []

    def fresh(self, rows: list[dict]) -> list[dict]:
        """Peticiones nuevas (sin responder y aún a tiempo), las más urgentes primero."""
        now = self.state.now_server()
        out: list[tuple[float, dict]] = []
        for r in rows:
            rid = str(r.get("request_id") or "")
            if not rid or rid in self.seen:
                continue
            self.seen[rid] = time.time()
            t = request_ts(r)
            if t == t and now - t > self.s.drop_after_s:      # t == t: no es NaN
                self.stats.add("dropped")
                continue
            out.append((t if t == t else now, r))
        out.sort(key=lambda x: x[0])
        return [r for _, r in out]

    def pull(self) -> list[dict]:
        return self.fresh(self.fetch())

    # ---------------------------------------------------------------- cálculo
    def answer(self, reqs: list[dict]) -> list[tuple[dict, float, float]]:
        if self.watcher is not None:
            self.rec.scorer = self.watcher.scorer
        version = self.rec.scorer.version
        start = time.perf_counter()
        out: list[tuple[dict, float, float]] = []
        for r in reqs:
            rid = str(r["request_id"])
            user = str(r.get("user_id") or "")
            t_req = request_ts(r)
            if t_req != t_req:
                t_req = self.state.now_server()
            t0 = time.perf_counter()
            items, source, cold = None, "model", False
            if t0 - start > self.s.batch_budget_s:
                source = "fb_budget"
            else:
                try:
                    rec = self.rec.recommend(r)
                    items, cold = rec.items, rec.cold
                    if not is_valid(items, self.state, self.s.n_items):
                        items = repair(items, self.state, self.s.n_items, exclude=self.state.bought(user))
                        self.stats.add("repaired")
                        if not is_valid(items, self.state, self.s.n_items):
                            source = "fb_invalid"
                except Exception:
                    source = "fb_error"
                    if time.time() - self._last_trace > 60:
                        self._last_trace = time.time()
                        traceback.print_exc()
            if source != "model":
                items = self.state.fallback_items(self.s.n_items, exclude=self.state.bought(user))
                if not is_valid(items, self.state, self.s.n_items):
                    self.stats.add("fb_unanswerable")      # imposible en la práctica: <10 disponibles
                    continue
                self.stats.add("fallback")
                self.stats.add(source)
            else:
                self.stats.add("model")
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.sample(compute_ms=compute_ms)
            self.stats.add("answered")
            self.state.remember_request(rid, user)
            answer = {"request_id": rid, "items": items, "model_version": version}
            deadline_local = t_req + self.s.deadline_s - self.state.clock_offset
            out.append((answer, deadline_local, t_req))
            if self.log is not None:
                self.log.write({"request_id": rid, "ts": r.get("ts"), "user_id": user,
                                "is_new_user": r.get("is_new_user"), "context": r.get("context"),
                                "recent_views": r.get("recent_views"), "items": items,
                                "model_version": version, "source": source, "cold": cold,
                                "compute_ms": round(compute_ms, 2)})
        return out

    # ---------------------------------------------------------------- bucle
    def handle(self, reqs: list[dict]) -> int:
        if not reqs:
            return 0
        answers = self.answer(reqs)
        if answers:
            self.pusher.submit(answers)
        if self.log is not None:
            self.log.flush()
        return len(answers)

    def step(self) -> int:
        """Una vuelta síncrona (lectura -> respuesta). La usan las pruebas."""
        return self.handle(self.pull())

    def prune(self) -> None:
        now = time.time()
        if now - self._last_prune < 5.0:
            return
        self._last_prune = now
        horizon = now - max(120.0, 3 * self.s.request_window_s)
        for rid in [k for k, t in self.seen.items() if t < horizon]:
            del self.seen[rid]

    def run(self, max_seconds: float | None = None) -> None:
        """Lanza una lectura cada `poll_interval` sin esperar a la anterior (hasta
        `max_inflight_pulls` a la vez): si una se cuelga, la siguiente ya está en camino. Todo lo que
        no es red (deduplicar, calcular, registrar) ocurre en este hilo, en orden."""
        self.pusher.start()
        pool = ThreadPoolExecutor(max_workers=self.s.max_inflight_pulls, thread_name_prefix="pull")
        inflight: set = set()
        next_launch = time.monotonic()
        t_end = time.time() + max_seconds if max_seconds else None
        print(f"[serve] en marcha con {self.rec.scorer.kind} {self.rec.scorer.version}", flush=True)
        while t_end is None or time.time() < t_end:
            try:
                now = time.monotonic()
                if now >= next_launch and len(inflight) < self.s.max_inflight_pulls:
                    inflight.add(pool.submit(self.fetch))
                    next_launch = now + self.s.poll_interval
                timeout = max(0.0, next_launch - time.monotonic())
                if inflight:
                    done, _ = wait(inflight, timeout=timeout, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(timeout)
                    done = set()
                for f in done:
                    inflight.discard(f)
                    self.handle(self.fresh(f.result()))
                self.prune()
                stale = [k for k, t in self.state.health.items()
                         if time.time() - t > max(600.0, 2 * self.state.intervals.get(k, 0.0))]
                self.stats.maybe_print(f"  sin refrescar >10 min: {', '.join(stale)}" if stale else "")
            except Exception:                         # el bucle no puede morir
                traceback.print_exc()
                time.sleep(0.1)


# ----------------------------------------------------------------------------------------------
def load_artifacts(path: Path, client: RawTreeClient) -> Artifacts:
    try:
        art = Artifacts.load(path)
        print(f"[serve] artefactos {path} (histórico hasta {to_float(art.until_ms) / 1000:.0f})")
        return art
    except FileNotFoundError:
        print(f"[serve] AVISO: no existe {path}; modo degradado (sin co-visitación ni contenido). "
              f"Ejecuta train.py --build para generarlo.")
    except Exception as exc:
        print(f"[serve] AVISO: no se pudo leer {path} ({exc!r}); modo degradado")
    ids = [str(r["product_id"]) for r in client.query(sql.CATALOG, timeout=60)]
    art = Artifacts.empty(ids)
    try:
        sales = client.query(sql.best_sellers(int(time.time() * 1000)), timeout=120)
        ranked = sorted(((int(to_float(r["n"], 0)), str(r["product_id"])) for r in sales), reverse=True)
        art.best_sellers = [p for _, p in ranked]
        index = {p: i for i, p in enumerate(ids)}
        for c, p in ranked:
            if p in index:
                art.buy_all[index[p]] = c
    except Exception as exc:
        print(f"[serve] best-sellers no disponibles: {exc!r}")
    return art


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="calcula pero no inserta en submissions")
    ap.add_argument("--artifacts", type=Path, default=config.LIVE_ARTIFACTS)
    ap.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    ap.add_argument("--no-explore", action="store_true", help="desactiva Thompson sampling")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    s = config.settings()
    fast = RawTreeClient(s.api_url, s.api_key, s.team, timeout=s.http_timeout_s, retries=0)
    slow = RawTreeClient(s.api_url, s.api_key, s.team, timeout=60, retries=2)

    artifacts = load_artifacts(args.artifacts, slow)
    state = LiveState(slow, artifacts, s)
    state.warm_up()
    state.start()
    watcher = ModelWatcher(args.model_dir, s.every_model_check).start()
    rec = Recommender(state, watcher.scorer, n_candidates=s.n_candidates, n_items=s.n_items,
                      n_explore=0 if args.no_explore else s.n_explore, explore_prior=s.explore_prior,
                      max_recent=s.max_recent, decay=s.recency_decay)
    server = Server(fast, state, rec, watcher, s, dry_run=args.dry_run, log=ServeLog(config.LOGS))
    try:
        server.run(args.max_seconds)
    except KeyboardInterrupt:
        print("\n[serve] parado por el usuario")
        sys.exit(0)


if __name__ == "__main__":
    main()
