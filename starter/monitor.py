"""Monitorización (rol A): el leaderboard estará a oscuras en la ronda final, esto es lo que queda.

    python monitor.py                # resumen de los últimos 60 minutos
    python monitor.py --minutes 15 --watch 60   # se repite cada 60 s

Muestra: semáforo (como el de la organización: verde si >90 % válidas y a tiempo en 5 min),
motivos de fallback, latencia, compras por 100 peticiones, revenue por petición y, cruzando con
logs/served-*.jsonl, el desglose por versión, cold start y origen (modelo o lista de emergencia).
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from reco import config, sql
from reco.client import from_env
from reco.timeutil import to_float


def load_logs(minutes: int) -> pd.DataFrame:
    rows = []
    for path in sorted(config.LOGS.glob("served-*.jsonl"))[-2:]:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rows.append({"request_id": r.get("request_id"), "model_version": r.get("model_version"),
                             "source": r.get("source"), "cold": bool(r.get("cold"))})
    return pd.DataFrame(rows).drop_duplicates("request_id", keep="last") if rows else pd.DataFrame()


def report(client, minutes: int) -> None:
    now_ms = int(to_float(client.query(sql.CLOCK)[0]["now_ms"]))
    fb = client.query_df(sql.feedback_since(client.team, now_ms - minutes * 60_000, limit=1_000_000), timeout=120)
    print(f"\n=== {time.strftime('%H:%M:%S')} · últimos {minutes} min · {len(fb)} peticiones ===")
    if fb.empty:
        print("sin feedback todavía")
        return
    fb["ts_ms"] = pd.to_numeric(fb["ts_ms"], errors="coerce")
    fb["latency_ms"] = pd.to_numeric(fb["latency_ms"], errors="coerce")
    fb["revenue_eur"] = pd.to_numeric(fb["revenue_eur"], errors="coerce").fillna(0)
    fb["ok"] = fb["served_by"] == "team"
    fb["buy"] = fb["purchased"].astype(str) != ""
    fb["click"] = fb["clicked"].astype(str) != ""

    last5 = fb[fb.ts_ms > now_ms - 5 * 60_000]
    share = last5["ok"].mean() if len(last5) else float("nan")
    light = "VERDE" if share > 0.9 else "ROJO"
    print(f"semáforo (5 min): {light}  {100 * share:.1f}% válidas y a tiempo")
    print(f"fiabilidad: {100 * fb['ok'].mean():.1f}%   motivos de fallback: "
          f"{fb.loc[~fb.ok, 'fallback_reason'].value_counts().to_dict() or '-'}")
    lat = fb.loc[fb.ok, "latency_ms"]
    if len(lat):
        print(f"latencia p50/p95/max: {lat.quantile(.5):.0f}/{lat.quantile(.95):.0f}/{lat.max():.0f} ms")
    print(f"compras por 100 peticiones: {100 * fb['buy'].mean():.2f}   CTR: {100 * fb['click'].mean():.2f}%   "
          f"revenue/petición: {fb['revenue_eur'].mean():.3f} €")
    print("por quién sirvió:")
    print(fb.groupby("served_by").agg(peticiones=("ok", "size"), compras_x100=("buy", lambda s: 100 * s.mean()),
                                      revenue=("revenue_eur", "mean")).round(3).to_string())

    logs = load_logs(minutes)
    if not logs.empty:
        m = fb.merge(logs, on="request_id", how="left")
        print("\npor versión, cold start y origen (nuestros logs):")
        print(m.groupby(["model_version", "cold", "source"], dropna=False)
               .agg(peticiones=("ok", "size"), compras_x100=("buy", lambda s: 100 * s.mean()),
                    ctr=("click", lambda s: 100 * s.mean()))
               .round(2).to_string())
        # compras por hueco: ¿se compra lo que ponemos arriba? ¿funcionan los huecos de exploración?
        pos = []
        for shown, p in zip(m["shown"], m["purchased"]):
            if p and isinstance(shown, list) and p in shown:
                pos.append(shown.index(p) + 1)
        if pos:
            print("\ncompras por hueco:", pd.Series(pos).value_counts().sort_index().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--watch", type=float, default=0, help="repetir cada N segundos")
    args = ap.parse_args()
    client = from_env(timeout=120, retries=1)
    while True:
        try:
            report(client, args.minutes)
        except Exception as exc:
            print(f"monitor: {exc}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
