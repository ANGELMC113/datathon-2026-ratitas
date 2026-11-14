"""Prueba cada consulta de reco/sql.py contra RawTree y dice cuál falla y cuánto tarda.
Ejecútalo antes de train.py --build y de serve_v2.py:

    python check_sql.py            # rápidas (en vivo + muestras pequeñas)
    python check_sql.py --heavy    # también co-visitación y carrusel sobre 1 día de histórico
"""
from __future__ import annotations

import argparse
import time

from reco import sql
from reco.client import from_env
from reco.timeutil import DAY_MS, to_float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", action="store_true")
    args = ap.parse_args()
    client = from_env(timeout=300, retries=0)

    t1 = int(time.time() * 1000)
    try:
        r = client.query(sql.HISTORY_RANGE)[0]
        t1 = int(to_float(r["t1"]))
    except Exception as exc:
        print(f"HISTORY_RANGE falló: {exc}")
    day = (t1 - DAY_MS, t1)

    checks = [
        ("CLOCK", sql.CLOCK),
        ("HISTORY_RANGE", sql.HISTORY_RANGE),
        ("requests_recent", sql.requests_recent(60)),
        ("requests_latest", sql.requests_latest(5)),
        ("CATALOG", sql.CATALOG + "\nLIMIT 3"),
        ("catalog_updates_since", sql.catalog_updates_since(0) + "\nLIMIT 3"),
        ("pop_window(24)", sql.pop_window(24) + "\nLIMIT 3"),
        ("TRENDS", sql.TRENDS + "\nLIMIT 3"),
        ("purchases_since(events)", sql.purchases_since("shop.events", 0) + "\nLIMIT 3"),
        ("purchases_since(history)", sql.purchases_since("shop.history_events", t1 - DAY_MS) + "\nLIMIT 3"),
        ("feedback_since", sql.feedback_since(client.team, 0, limit=3)),
        ("best_sellers", sql.best_sellers(t1) + "\nLIMIT 3"),
        ("daily_counts", sql.daily_counts("shop.history_events", "purchase") + "\nLIMIT 3"),
        ("impressions", sql.impressions(*day, keep_permille=5, limit=3)),
    ]
    if args.heavy:
        checks += [(f"covis({k}) 1 día", sql.covis(k, *day, top_k=5) + "\nLIMIT 3") for k in sql.COVIS_KINDS]
        checks.append(("carousel_stats 1 día", sql.carousel_stats(*day) + "\nLIMIT 3"))

    ok = 0
    for name, q in checks:
        t = time.time()
        try:
            rows = client.query(q)
            ok += 1
            sample = rows[0] if rows else "(sin filas)"
            print(f"OK    {name:28s} {time.time() - t:6.2f} s  {str(sample)[:150]}")
        except Exception as exc:
            print(f"FALLA {name:28s} {time.time() - t:6.2f} s  {str(exc)[:300]}")
    print(f"\n{ok}/{len(checks)} consultas correctas")


if __name__ == "__main__":
    main()
