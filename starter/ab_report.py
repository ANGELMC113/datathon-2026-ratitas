"""Informe del test A/B con la métrica oficial (rol A/C).

Lee la API pública del leaderboard (la misma que usa la página) y muestra, para nuestro equipo,
la conversión esperada de cada model_version en varias ventanas consecutivas. Es la métrica sin
suerte del reto (compras esperadas según el modelo oculto), así que sirve para decidir variantes.

    python ab_report.py                    # 4 ventanas de 15 min hasta ahora
    python ab_report.py --length 30 --windows 3
    python ab_report.py --team team_30

Una variante solo gana si queda por delante en (casi) todas las ventanas, no en una suelta.
OJO: en la ronda final el leaderboard está oculto; esto solo sirve durante la práctica.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd
import requests

from reco import config  # noqa: F401  (carga .env para TEAM_ID)

API = "https://datathon.rawtree.dev/api/state"
STEP = 300


def fetch(start: int, end: int) -> dict:
    r = requests.get(API, params={"by": "model", "from": start, "to": end}, timeout=20)
    r.raise_for_status()
    return r.json()


def window_table(state: dict, team: str) -> tuple[pd.DataFrame, dict]:
    base = state.get("baseline") or {}
    ref = state.get("reference") or {}
    teams = state.get("teams") or []
    rank = next((k + 1 for k, t in enumerate(teams) if t.get("team_id") == team), None)
    mine = next((t for t in teams if t.get("team_id") == team), None)
    rows = []
    for v in (mine or {}).get("versions", []):
        rows.append({"model_version": v["model_version"], "peticiones": v["requests"],
                     "conversion": v["conversion"], "x_baseline": v["conversion"] / max(base.get("conversion", 0), 1e-9),
                     "cold": v.get("cold_conversion"), "eur_peticion": v.get("revenue_per_request")})
    info = {"rank": rank, "n_teams": len(teams), "baseline": base.get("conversion"),
            "reference": ref.get("conversion"), "team_conv": (mine or {}).get("conversion"),
            "on_time": (mine or {}).get("on_time"),
            "leader": teams[0]["team_id"] if teams else None,
            "leader_conv": teams[0].get("conversion") if teams else None}
    return pd.DataFrame(rows), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default=os.environ.get("TEAM_ID", "team_30"))
    ap.add_argument("--length", type=int, default=15, help="minutos por ventana (mínimo 15, múltiplo de 5)")
    ap.add_argument("--windows", type=int, default=4, help="ventanas consecutivas hacia atrás")
    args = ap.parse_args()
    length = max(15, args.length // 5 * 5) * 60
    end = int(time.time()) // STEP * STEP
    per_window = []
    for k in range(args.windows, 0, -1):
        start = end - k * length
        state = fetch(start, start + length)
        if state.get("hidden") and state.get("mode") == "final":
            print("Ronda final: el leaderboard está oculto, no hay datos por versión.")
            return
        df, info = window_table(state, args.team)
        label = f"{datetime.fromtimestamp(start):%H:%M}-{datetime.fromtimestamp(start + length):%H:%M}"
        print(f"\n=== {label} · {args.team} puesto {info['rank']}/{info['n_teams']} · conversión equipo "
              f"{info['team_conv']:.3f} · a tiempo {info['on_time']:.1f}% · baseline {info['baseline']:.3f} · "
              f"referencia {info['reference']:.3f} · líder {info['leader']} {info['leader_conv']:.3f}")
        if df.empty:
            print("  (sin versiones con suficientes peticiones)")
            continue
        print(df.sort_values("conversion", ascending=False).round(3).to_string(index=False))
        df["ventana"] = label
        per_window.append(df)

    if len(per_window) >= 2:
        allw = pd.concat(per_window)
        piv = allw.pivot_table(index="model_version", columns="ventana", values="x_baseline")
        print("\n=== Multiplicador sobre el baseline por ventana (lo que hay que comparar) ===")
        print(piv.round(2).to_string())
        ranks = allw.pivot_table(index="model_version", columns="ventana", values="conversion").rank(ascending=False)
        wins = (ranks == 1).sum(axis=1)
        print("\nVentanas en las que cada versión fue la mejor:")
        print(wins.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
