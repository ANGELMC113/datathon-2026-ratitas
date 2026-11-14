"""Reentrenamiento automático con puerta de calidad (rol C/A).

    python auto_retrain.py --every-hours 3 --freeze-at "2026-11-15 08:30"

Cada `--every-hours` horas:
1. Ejecuta `train.py --with-feedback` en un proceso aparte, con prioridad baja y pocos hilos para no
   quitarle CPU al servicio. Los últimos `--holdout-minutes` de feedback se reservan: ningún
   modelo los ha visto.
2. Compara en esos minutos reservados el modelo nuevo con el que está desplegado.
3. Solo si el nuevo es mejor (logloss menor y AUC no peor), lo copia a artifacts/model/; el servicio
   lo carga en caliente en ~10 s. Si no, el desplegado sigue intacto.
4. A partir de `--freeze-at` (hora local) ya no despliega nada: la versión final queda estable antes
   de la ronda final, como pide la guía del equipo.

Cada decisión queda en logs/retrain_history.csv (para el CHANGELOG y la presentación).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from reco import config

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[auto {datetime.now():%H:%M:%S}] {msg}", flush=True)


def decide(meta: dict, min_buys: int = 30, auc_tolerance: float = 0.005) -> tuple[bool, str]:
    """¿Desplegar el candidato? Solo con evidencia en datos que ninguno de los dos ha visto."""
    hold = (meta.get("metrics") or {}).get("holdout")
    if not hold:
        return False, "sin feedback reservado para comparar"
    if hold.get("n_buys", 0) < min_buys:
        return False, f"solo {hold.get('n_buys', 0)} compras en lo reservado (mínimo {min_buys})"
    cand, cur = hold.get("candidate", {}), hold.get("current", {})
    if "logloss" not in cand or "auc" not in cand:
        return False, "métricas del candidato incompletas"
    if "logloss" not in cur or "auc" not in cur:
        return True, "el desplegado no tiene métricas comparables"
    detail = (f"logloss {cand['logloss']:.5f} vs {cur['logloss']:.5f}, "
              f"AUC {cand['auc']:.4f} vs {cur['auc']:.4f} ({hold['current_version']})")
    if cand["logloss"] <= cur["logloss"] and cand["auc"] >= cur["auc"] - auc_tolerance:
        return True, "mejor: " + detail
    return False, "no mejora: " + detail


def deploy(candidate_dir: Path, model_dir: Path) -> None:
    """Copia el modelo y DESPUÉS model.json (lo que vigila el servicio), cada uno de forma atómica."""
    meta = json.loads((candidate_dir / "model.json").read_text())
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (meta["file"], "model.json"):
        tmp = model_dir / (name + ".tmp")
        shutil.copy2(candidate_dir / name, tmp)
        tmp.replace(model_dir / name)


def record(row: dict) -> None:
    path = config.LOGS / "retrain_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run_once(args) -> bool:
    version = f"{args.prefix}-{datetime.now():%d%H%M}"
    cand = config.ARTIFACTS / "candidate"
    shutil.rmtree(cand, ignore_errors=True)
    config.LOGS.mkdir(parents=True, exist_ok=True)
    out = config.LOGS / f"train-{version}.log"
    cmd = [sys.executable, str(HERE / "train.py"), "--with-feedback", "--holdout-minutes", str(args.holdout_minutes),
           "--model-dir", str(cand), "--version", version, "--threads", str(args.threads),
           "--backend", args.backend, "--eval-imps", "1500"]
    log(f"entrenando {version} (salida en {out}) ...")
    t = time.time()
    with open(out, "w") as f:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, timeout=args.timeout_min * 60,
                                preexec_fn=(lambda: os.nice(10)) if hasattr(os, "nice") else None).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    if rc != 0 or not (cand / "model.json").exists():
        log(f"el entrenamiento falló (código {rc}); el modelo desplegado sigue igual. Mira {out}")
        record({"time": datetime.now().isoformat(timespec="seconds"), "version": version, "deployed": False,
                "reason": f"entrenamiento falló ({rc})", "minutes": round((time.time() - t) / 60, 1)})
        return False
    meta = json.loads((cand / "model.json").read_text())
    ok, reason = decide(meta, args.min_buys)
    if ok and args.dry_run:
        reason = "(simulación) " + reason
        ok = False
    if ok:
        deploy(cand, config.MODEL_DIR)
        log(f"DESPLEGADO {version}: {reason}")
    else:
        log(f"no se despliega {version}: {reason}")
    hold = meta.get("metrics", {}).get("holdout", {})
    record({"time": datetime.now().isoformat(timespec="seconds"), "version": version, "deployed": ok,
            "reason": reason, "minutes": round((time.time() - t) / 60, 1),
            "holdout_buys": hold.get("n_buys", ""),
            "cand_auc": hold.get("candidate", {}).get("auc", ""), "cur_auc": hold.get("current", {}).get("auc", ""),
            "cand_logloss": hold.get("candidate", {}).get("logloss", ""),
            "cur_logloss": hold.get("current", {}).get("logloss", "")})
    return ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every-hours", type=float, default=3.0)
    ap.add_argument("--holdout-minutes", type=int, default=60)
    ap.add_argument("--min-buys", type=int, default=30, help="compras mínimas en lo reservado para decidir")
    ap.add_argument("--freeze-at", default="", help='hora local "AAAA-MM-DD HH:MM" a partir de la cual no se despliega')
    ap.add_argument("--threads", type=int, default=2, help="hilos de LightGBM (deja CPU al servicio)")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--prefix", default="v3-auto")
    ap.add_argument("--timeout-min", type=int, default=90)
    ap.add_argument("--once", action="store_true", help="un solo ciclo y salir")
    ap.add_argument("--dry-run", action="store_true", help="entrena y compara, pero no despliega")
    args = ap.parse_args(argv)
    freeze = datetime.strptime(args.freeze_at, "%Y-%m-%d %H:%M") if args.freeze_at else None

    while True:
        if freeze and datetime.now() >= freeze:
            log(f"congelado desde {freeze:%H:%M}: no se despliega nada más. Fin.")
            return
        try:
            run_once(args)
        except Exception as exc:                       # nunca tocar el modelo desplegado por un error
            log(f"error inesperado: {exc!r}; el modelo desplegado sigue igual")
        if args.once:
            return
        next_run = time.time() + args.every_hours * 3600
        if freeze and datetime.fromtimestamp(next_run) >= freeze:
            log(f"el siguiente ciclo caería después de {freeze:%H:%M}: fin.")
            return
        log(f"próximo ciclo a las {datetime.fromtimestamp(next_run):%H:%M}")
        time.sleep(max(0.0, next_run - time.time()))


if __name__ == "__main__":
    main()
