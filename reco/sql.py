"""Todas las consultas SQL en un solo sitio (dialecto ClickHouse, el de RawTree).

Las columnas son Dynamic: se castea (::String, ::DateTime64(3)) en joins, agregados, IN y
comparaciones de tiempo (`IN` sobre una columna Dynamic sin castear da "Illegal type Dynamic"). Los tiempos se comparan en milisegundos epoch para no depender de la zona horaria del
servidor. Los enteros de 64 bits llegan como texto en el JSON: se convierten en Python.

`python check_sql.py` ejecuta cada consulta con datos reales y dice cuál falla.
"""
from __future__ import annotations

TS_MS = "toUnixTimestamp64Milli(ts::DateTime64(3))"


def _between(since_ms: int, until_ms: int) -> str:
    return f"{TS_MS} >= {int(since_ms)} AND {TS_MS} < {int(until_ms)}"


# ----------------------------------------------------------------------------------------------
# En vivo
# ----------------------------------------------------------------------------------------------
CLOCK = "SELECT toUnixTimestamp64Milli(now64(3)) AS now_ms"


def requests_recent(window_s: int, limit: int = 2000) -> str:
    """Peticiones de los últimos `window_s` segundos (filtrar por tiempo evita ordenar la tabla entera)."""
    return f"""
SELECT request_id::String AS request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
WHERE ts::DateTime64(3) > now64(3) - INTERVAL {int(window_s)} SECOND
ORDER BY ts::DateTime64(3) DESC
LIMIT {int(limit)}"""


def requests_latest(limit: int = 60) -> str:
    """Plan B, la consulta del starter (ya probada): las N peticiones más recientes."""
    return f"""
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT {int(limit)}"""


_PRODUCT_COLS = """
       product_id::String AS product_id,
       ifNull(category::Nullable(String), '') AS category,
       ifNull(subcategory::Nullable(String), '') AS subcategory,
       ifNull(brand::Nullable(String), '') AS brand,
       ifNull(price_eur::Nullable(Float64), 0) AS price_eur,
       ifNull(description::Nullable(String), '') AS description,
       ifNull(toUnixTimestamp64Milli(created_at::Nullable(DateTime64(3))), 0) AS created_ms,
       ifNull(is_active::Nullable(Bool), true) AS is_active"""

CATALOG = f"SELECT {_PRODUCT_COLS}\nFROM shop.catalog"


def catalog_updates_since(since_ms: int) -> str:
    """Cambios de catálogo desde `since_ms` (>=: los repetidos se aplican de forma idempotente)."""
    return f"""
SELECT {_PRODUCT_COLS},
       change::String AS change,
       {TS_MS} AS ts_ms
FROM shop.catalog_updates
WHERE {TS_MS} >= {int(since_ms)}
ORDER BY ts_ms"""


def pop_window(hours: int) -> str:
    """Actividad orgánica por producto en las últimas `hours` horas (histórico + vivo)."""
    return f"""
SELECT product_id, event_type, count() AS n
FROM (
    SELECT product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
    WHERE ts::DateTime64(3) > now64(3) - INTERVAL {int(hours)} HOUR
    UNION ALL
    SELECT product_id::String AS product_id, event_type::String AS event_type
    FROM shop.events
    WHERE ts::DateTime64(3) > now64(3) - INTERVAL {int(hours)} HOUR
)
GROUP BY product_id, event_type"""


TRENDS = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view' AND ts::DateTime64(3) > now64(3) - INTERVAL 15 MINUTE) AS v15,
       countIf(event_type::String = 'view') AS v60,
       countIf(event_type::String IN ('add_to_cart', 'purchase')) AS b60
FROM shop.events
WHERE ts::DateTime64(3) > now64(3) - INTERVAL 1 HOUR
GROUP BY product_id"""


def purchases_since(table: str, since_ms: int) -> str:
    return f"""
SELECT user_id::String AS user_id, product_id::String AS product_id, {TS_MS} AS ts_ms
FROM {table}
WHERE event_type = 'purchase' AND {TS_MS} > {int(since_ms)}"""


def feedback_since(team: str, since_ms: int, limit: int = 50000) -> str:
    return f"""
SELECT request_id::String AS request_id, {TS_MS} AS ts_ms, shown,
       ifNull(served_by::Nullable(String), '') AS served_by,
       ifNull(fallback_reason::Nullable(String), '') AS fallback_reason,
       latency_ms,
       ifNull(clicked::Nullable(String), '') AS clicked,
       ifNull(purchased::Nullable(String), '') AS purchased,
       ifNull(revenue_eur::Nullable(Float64), 0) AS revenue_eur
FROM {team}.feedback
WHERE request_id != '__init__' AND {TS_MS} > {int(since_ms)}
ORDER BY ts_ms
LIMIT {int(limit)}"""


# ----------------------------------------------------------------------------------------------
# Offline (entrenamiento y artefactos)
# ----------------------------------------------------------------------------------------------
HISTORY_RANGE = f"""
SELECT min({TS_MS}) AS t0, max({TS_MS}) AS t1
FROM shop.history_sessions"""


def daily_counts(table: str, event_type: str) -> str:
    """Eventos por día UTC y producto: la popularidad de cada día se reconstruye sin fuga de futuro."""
    return f"""
SELECT intDiv({TS_MS}, 86400000) AS day, product_id::String AS product_id, count() AS n
FROM {table}
WHERE event_type = '{event_type}'
GROUP BY day, product_id"""


def best_sellers(until_ms: int) -> str:
    return f"""
SELECT product_id::String AS product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase' AND {TS_MS} < {int(until_ms)}
GROUP BY product_id"""


def _distinct_pairs(key: str, event_types: tuple[str, ...], since_ms: int, until_ms: int) -> str:
    types = ", ".join(f"'{t}'" for t in event_types)
    return (f"SELECT DISTINCT {key}::String AS k, product_id::String AS p FROM shop.history_events "
            f"WHERE event_type::String IN ({types}) AND {_between(since_ms, until_ms)}")


# tipo de co-visitación -> (clave de agrupación, eventos del producto origen, eventos del destino)
COVIS_KINDS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "view": ("session_id", ("view",), ("view",)),                                # vista -> vista
    "buy": ("session_id", ("view", "add_to_cart"), ("add_to_cart", "purchase")),  # vista -> carrito/compra
    "b2b": ("user_id", ("purchase",), ("purchase",)),                            # compra -> compra
}


def covis(kind: str, since_ms: int, until_ms: int, top_k: int) -> str:
    """Matriz de co-visitación top-K: cuántas sesiones (o usuarios) comparten el par p -> q."""
    key, left, right = COVIS_KINDS[kind]
    return f"""
SELECT a.p AS p, b.p AS q, count() AS n
FROM ({_distinct_pairs(key, left, since_ms, until_ms)}) AS a
INNER JOIN ({_distinct_pairs(key, right, since_ms, until_ms)}) AS b ON a.k = b.k
WHERE a.p != b.p
GROUP BY p, q
ORDER BY p, n DESC
LIMIT {int(top_k)} BY p"""


def carousel_stats(since_ms: int, until_ms: int) -> str:
    """Impresiones, clics y compras del carrusel antiguo por producto y posición (para quitar el
    sesgo de posición: un producto en el hueco 1 vende más por estar ahí, no por ser mejor)."""
    return f"""
SELECT p AS product_id, pos, count() AS shown, countIf(c = p) AS clicks, countIf(b = p) AS buys
FROM (
    SELECT CAST(shown, 'Array(String)') AS sh,
           ifNull(CAST(clicked, 'Nullable(String)'), '') AS c,
           ifNull(CAST(purchased, 'Nullable(String)'), '') AS b
    FROM shop.history_sessions
    WHERE {_between(since_ms, until_ms)}
)
ARRAY JOIN sh AS p, arrayEnumerate(sh) AS pos
GROUP BY product_id, pos"""


def click_impressions(since_ms: int, until_ms: int) -> str:
    """Impresiones del histórico con clic (para el two-tower: cada clic es un ejemplo positivo)."""
    return f"""
SELECT request_id::String AS request_id, {TS_MS} AS ts_ms, user_id::String AS user_id,
       context, recent_views, CAST(shown, 'Array(String)') AS shown,
       ifNull(CAST(clicked, 'Nullable(String)'), '') AS clicked,
       ifNull(CAST(purchased, 'Nullable(String)'), '') AS purchased
FROM shop.history_sessions
WHERE {_between(since_ms, until_ms)}
  AND ifNull(CAST(clicked, 'Nullable(String)'), '') != ''"""


def impressions(since_ms: int, until_ms: int, keep_permille: int, limit: int | None = None) -> str:
    """Impresiones del histórico para entrenar: todas las que acaban en compra y una muestra
    determinista (hash del request_id) del resto. El peso 1000/keep_permille compensa el muestreo."""
    lim = f"\nLIMIT {int(limit)}" if limit else ""
    return f"""
SELECT request_id::String AS request_id, {TS_MS} AS ts_ms, user_id::String AS user_id,
       context, recent_views, CAST(shown, 'Array(String)') AS shown,
       ifNull(CAST(clicked, 'Nullable(String)'), '') AS clicked,
       ifNull(CAST(purchased, 'Nullable(String)'), '') AS purchased
FROM shop.history_sessions
WHERE {_between(since_ms, until_ms)}
  AND (ifNull(CAST(purchased, 'Nullable(String)'), '') != ''
       OR cityHash64(request_id::String) % 1000 < {int(keep_permille)}){lim}"""
