"""¿Recompra la gente lo que ya compró? Mide si la regla «no recomendar lo ya comprado» nos cuesta compras.

Solo lee tablas, no toca el servicio. Responde a tres preguntas:

1. Carrusel histórico: de las compras en el carrusel, ¿cuántas son de un producto que ese mismo
   usuario ya había comprado en los 7 días anteriores? (la memoria de compras del servicio es de 7 días)
2. Carrusel histórico: cuando se mostró un producto ya comprado, ¿se compró más o menos que uno nuevo?
   Esta es la prueba directa: compara tasas, no solo cuántas hay.
3. Compras orgánicas (histórico y en vivo): ¿cuántas son repetidas?

    python diagnose_rebuy.py              # unos segundos o un par de minutos
    python diagnose_rebuy.py --days 7     # ventana más corta si va lento
"""
from __future__ import annotations

import argparse
import time

from reco import sql
from reco.client import from_env
from reco.timeutil import DAY_MS, to_float

TS = sql.TS_MS
MEM_MS = 7 * DAY_MS          # igual que PURCHASE_MEMORY_S en reco/live.py
GAP_MS = 60_000              # una compra «previa» debe ser 1 min anterior (evita contar la misma compra dos veces)
BOUGHT = "ifNull(CAST(purchased, 'Nullable(String)'), '')"


def prior_purchases(lo: int, hi: int, live: bool = False) -> str:
    """Todas las compras (orgánicas y del carrusel) entre lo y hi: (u, p, t)."""
    parts = [
        f"SELECT user_id::String AS u, product_id::String AS p, {TS} AS t FROM shop.history_events "
        f"WHERE event_type::String = 'purchase' AND {TS} >= {lo} AND {TS} < {hi}",
        f"SELECT user_id::String AS u, {BOUGHT} AS p, {TS} AS t FROM shop.history_sessions "
        f"WHERE {BOUGHT} != '' AND {TS} >= {lo} AND {TS} < {hi}",
    ]
    if live:
        parts.append(f"SELECT user_id::String AS u, product_id::String AS p, {TS} AS t FROM shop.events "
                     f"WHERE event_type::String = 'purchase' AND {TS} >= {lo} AND {TS} < {hi}")
    return "(" + "\nUNION ALL\n".join(parts) + ")"


def repeat_share(target: str, prior: str) -> str:
    """Qué parte de las compras de `target` (u, p, t) tienen otra compra previa del mismo usuario y producto."""
    return f"""
SELECT count() AS n, countIf(prev) AS rep FROM (
  SELECT c.u, c.p, c.t,
         max(pr.t < c.t - {GAP_MS} AND pr.t >= c.t - {MEM_MS}) AS prev
  FROM {target} AS c
  LEFT JOIN {prior} AS pr ON c.u = pr.u AND c.p = pr.p
  WHERE c.u != ''
  GROUP BY c.u, c.p, c.t
)"""


def shown_rates(since: int, until: int, keep_permille: int) -> str:
    """Tasa de compra de los productos mostrados, separando los que el usuario ya había comprado."""
    shown = f"""(
  SELECT request_id::String AS rid, user_id::String AS u, {TS} AS t, {BOUGHT} AS b, p
  FROM shop.history_sessions
  ARRAY JOIN CAST(shown, 'Array(String)') AS p
  WHERE {TS} >= {since} AND {TS} < {until} AND user_id::String != ''
    AND cityHash64(request_id::String) % 1000 < {keep_permille}
)"""
    return f"""
SELECT prev, count() AS n, countIf(p = b) AS buys FROM (
  SELECT s.rid, s.p, s.b,
         max(pr.t < s.t - {GAP_MS} AND pr.t >= s.t - {MEM_MS}) AS prev
  FROM {shown} AS s
  LEFT JOIN {prior_purchases(since - MEM_MS, until)} AS pr ON s.u = pr.u AND s.p = pr.p
  GROUP BY s.rid, s.p, s.b
)
GROUP BY prev
ORDER BY prev"""


def run(client, name: str, query: str) -> list[dict] | None:
    t0 = time.time()
    try:
        rows = client.query(query, timeout=300)
    except Exception as exc:
        print(f"  [{name}] falló: {exc}")
        return None
    print(f"  ({name}: {time.time() - t0:.1f} s)")
    return rows


def pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.2f} %" if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="días finales del histórico a analizar")
    ap.add_argument("--live-hours", type=float, default=6.0)
    ap.add_argument("--keep-permille", type=int, default=50, help="muestra de impresiones para la pregunta 2")
    args = ap.parse_args()
    client = from_env(timeout=300, retries=0)

    t1 = int(to_float(client.query(sql.HISTORY_RANGE)[0]["t1"]))
    since = t1 - args.days * DAY_MS
    now = int(to_float(client.query(sql.CLOCK)[0]["now_ms"]))
    live_since = now - int(args.live_hours * 3600 * 1000)

    print(f"\n1) Compras en el carrusel histórico (últimos {args.days} días)")
    carousel = (f"(SELECT user_id::String AS u, {BOUGHT} AS p, {TS} AS t FROM shop.history_sessions "
                f"WHERE {BOUGHT} != '' AND {TS} >= {since} AND {TS} < {t1})")
    r = run(client, "recompras carrusel", repeat_share(carousel, prior_purchases(since - MEM_MS, t1)))
    share_carousel = None
    if r:
        n, rep = to_float(r[0]["n"]), to_float(r[0]["rep"])
        share_carousel = rep / n if n else 0.0
        print(f"   {int(n)} compras; {int(rep)} eran de algo que el usuario ya había comprado: {pct(rep, n)}")

    print(f"\n2) Tasa de compra al mostrar un producto ya comprado (muestra {args.keep_permille / 10:.1f} %)")
    r = run(client, "tasas por producto mostrado", shown_rates(since, t1, args.keep_permille))
    lift = None
    if r:
        by = {int(to_float(x["prev"])): (to_float(x["n"]), to_float(x["buys"])) for x in r}
        n0, b0 = by.get(0, (0.0, 0.0))
        n1, b1 = by.get(1, (0.0, 0.0))
        print(f"   producto nuevo para el usuario: {int(n0)} mostrados, compra {pct(b0, n0)}")
        print(f"   producto ya comprado:           {int(n1)} mostrados, compra {pct(b1, n1)}")
        if n0 and n1 and b0:
            lift = (b1 / n1) / (b0 / n0)
            print(f"   -> un producto ya comprado se compra {lift:.2f}× lo que uno nuevo"
                  f" (con {int(b1)} compras, {'pocas: tómalo con cautela' if b1 < 30 else 'suficientes'})")

    print("\n3) Compras orgánicas repetidas")
    organic_hist = (f"(SELECT user_id::String AS u, product_id::String AS p, {TS} AS t FROM shop.history_events "
                    f"WHERE event_type::String = 'purchase' AND {TS} >= {t1 - 7 * DAY_MS} AND {TS} < {t1})")
    r = run(client, "orgánicas histórico", repeat_share(organic_hist, prior_purchases(t1 - 14 * DAY_MS, t1)))
    if r:
        n, rep = to_float(r[0]["n"]), to_float(r[0]["rep"])
        print(f"   histórico (7 últimos días): {int(n)} compras, repetidas {pct(rep, n)}")
    organic_live = (f"(SELECT user_id::String AS u, product_id::String AS p, {TS} AS t FROM shop.events "
                    f"WHERE event_type::String = 'purchase' AND {TS} >= {live_since})")
    r = run(client, "orgánicas en vivo",
            repeat_share(organic_live, prior_purchases(live_since - MEM_MS, now + 60_000, live=True)))
    if r:
        n, rep = to_float(r[0]["n"]), to_float(r[0]["rep"])
        print(f"   en vivo (últimas {args.live_hours:g} h): {int(n)} compras, repetidas {pct(rep, n)}")

    print("\nConclusión")
    if lift is None and share_carousel is None:
        print("   No hay datos suficientes; revisa los errores de arriba.")
    elif (lift is not None and lift >= 0.8) or (share_carousel is not None and share_carousel >= 0.05):
        print("   La gente sí recompra: la regla nos puede estar costando compras.")
        print("   Prueba la variante rebuy en el A/B, por ejemplo:")
        print("   ./run_serve.sh --ab-model-dir artifacts/model_d artifacts/model_e artifacts/model_e:rebuy")
    else:
        print("   Casi nadie recompra (o se compra mucho menos lo ya comprado): la regla está bien, no la toques.")


if __name__ == "__main__":
    main()
