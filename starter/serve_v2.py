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
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from reco import config, sql
from reco.artifacts import Artifacts
from reco.client import RawTreeClient
from reco.live import LiveState
from reco.model import ModelWatcher
from reco.candidates import QUOTAS_OLD
from reco.recommender import Recommender, is_valid, repair
from reco.serving import Arm, Pusher, ServeLog, Stats, bucket, parse_arm, pick_arm  # noqa: F401
from reco.session import request_ts
from reco.timeutil import to_float


class Server:
    def __init__(self, client: RawTreeClient, state: LiveState, recommender: Recommender,
                 watcher: ModelWatcher | None, settings: config.Settings, *, dry_run: bool = False,
                 log: ServeLog | None = None, arms: list[Arm] | None = None) -> None:
        self.client, self.state, self.rec, self.watcher = client, state, recommender, watcher
        # test A/B/n: cada variante extra recibe una fracción fija de peticiones (por hash del
        # request_id); el resto va al modelo principal
        self.arms: list[Arm] = list(arms or [])
        self.s = settings
        self.stats = Stats()
        self.log = log
        self.seen: dict[str, float] = {}
        self.main_suffix = ""                     # opciones del modelo principal en model_version
        self.main_adapt = False                   # el modelo principal usa el lift del feedback
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
        scorer_a = self.rec.scorer
        # un brazo sin modelo cargado (heurística), o que sería idéntico a A, no se usa: va a A
        arm_scorers = []
        for arm in self.arms:
            sc = arm.watcher.scorer
            same_as_a = sc.version == scorer_a.version and arm.suffix() == self.main_suffix
            arm_scorers.append(sc if sc.kind != "heuristic" and not same_as_a else None)
        fractions = [a.fraction for a in self.arms]
        start = time.perf_counter()
        out: list[tuple[dict, float, float]] = []
        for r in reqs:
            rid = str(r["request_id"])
            user = str(r.get("user_id") or "")
            t_req = request_ts(r)
            if t_req != t_req:
                t_req = self.state.now_server()
            t0 = time.perf_counter()
            k = pick_arm(rid, fractions) if self.arms else -1
            arm = self.arms[k] if k >= 0 and arm_scorers[k] is not None else None
            scorer = arm_scorers[k] if arm is not None else scorer_a
            version = scorer.version + (arm.suffix() if arm is not None else self.main_suffix)
            items, source, cold = None, "model", False
            if t0 - start > self.s.batch_budget_s:
                source = "fb_budget"
            else:
                try:
                    rec = self.rec.recommend(r, scorer=scorer,
                                             n_explore=arm.n_explore if arm is not None else None,
                                             max_per_sub=arm.max_per_sub if arm is not None else None,
                                             quotas=(QUOTAS_OLD if arm.old_candidates else None)
                                             if arm is not None else None,
                                             rebuy=arm is not None and arm.rebuy,
                                             adapt=(arm.adapt if arm is not None else self.main_adapt))
                    items, cold = rec.items, rec.cold
                    if not is_valid(items, self.state, self.s.n_items):
                        exclude = set() if arm is not None and arm.rebuy else self.state.bought(user)
                        items = repair(items, self.state, self.s.n_items, exclude=exclude)
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
                self.state.remember_preds(rid, {p: rec.raw[p] for p in items if p in rec.raw})
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.sample(compute_ms=compute_ms)
            self.stats.add("answered")
            if self.arms:
                self.stats.add(f"arm:{version}")
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
    ap.add_argument("--ab-model-dir", nargs="+", default=None,
                    help="variantes extra a probar a la vez: CARPETA[:opciones], p. ej. artifacts/model_b "
                         "'artifacts/model:div=2' 'artifacts/model:div=2,noexp' (div=N: máx. N por "
                         "subcategoría; noexp: sin exploración). Cada modelo se recarga en caliente")
    ap.add_argument("--old-candidates", action="store_true", default=None,
                    help="modelo principal con la búsqueda de candidatos sin transiciones (la de antes de las 21:43)")
    ap.add_argument("--max-per-sub", type=int, default=None,
                    help="diversidad del modelo principal: máx. N productos por subcategoría (0 = sin límite)")
    ap.add_argument("--adapt", action="store_true",
                    help="modelo principal con el lift aprendido del feedback en vivo (LiftStats)")
    ap.add_argument("--ab-fraction", type=float, default=None,
                    help="fracción de peticiones para CADA modelo extra (por defecto, reparto a partes iguales)")
    args = ap.parse_args(argv)

    s = config.settings()
    if args.max_per_sub is None:
        args.max_per_sub = s.max_per_sub
    if args.old_candidates is None:
        args.old_candidates = s.old_candidates
    fast = RawTreeClient(s.api_url, s.api_key, s.team, timeout=s.http_timeout_s, retries=0)
    slow = RawTreeClient(s.api_url, s.api_key, s.team, timeout=60, retries=2)

    artifacts = load_artifacts(args.artifacts, slow)
    state = LiveState(slow, artifacts, s)
    state.warm_up()
    state.start()
    watcher = ModelWatcher(args.model_dir, s.every_model_check).start()
    rec = Recommender(state, watcher.scorer, n_candidates=s.n_candidates, n_items=s.n_items,
                      n_explore=0 if args.no_explore else s.n_explore, explore_prior=s.explore_prior,
                      max_recent=s.max_recent, decay=s.recency_decay, max_per_sub=args.max_per_sub)
    if args.old_candidates:
        rec.quotas = QUOTAS_OLD
    arms: list[Arm] = []
    if args.ab_model_dir:
        frac = args.ab_fraction if args.ab_fraction is not None else 1.0 / (len(args.ab_model_dir) + 1)
        if frac * len(args.ab_model_dir) >= 1.0:
            raise SystemExit("--ab-fraction demasiado alta: no quedaría tráfico para el modelo principal")
        watchers: dict[str, ModelWatcher] = {}
        for spec in args.ab_model_dir:
            arm = parse_arm(spec, frac)
            key = str(arm.path.resolve())
            if key == str(args.model_dir.resolve()):
                arm.watcher = watcher                       # mismo modelo que el principal
            else:
                if key not in watchers:
                    watchers[key] = ModelWatcher(arm.path, s.every_model_check).start()
                arm.watcher = watchers[key]
            arms.append(arm)
            print(f"[serve] test A/B: {frac:.0%} de las peticiones a {arm.path}{arm.suffix() or ''}")
        print(f"[serve] el {1 - frac * len(arms):.0%} restante, al modelo principal {args.model_dir}")
    server = Server(fast, state, rec, watcher, s, dry_run=args.dry_run, log=ServeLog(config.LOGS),
                    arms=arms)
    server.main_suffix = Arm(None, 0, 0 if args.no_explore else None, args.max_per_sub,
                             args.old_candidates, adapt=args.adapt).suffix()
    server.main_adapt = args.adapt
    try:
        server.run(args.max_seconds)
    except KeyboardInterrupt:
        print("\n[serve] parado por el usuario")
        sys.exit(0)


if __name__ == "__main__":
    main()
