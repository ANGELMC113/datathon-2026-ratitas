"""Tiempos. RawTree devuelve timestamps UTC como texto ('2026-11-14 09:00:00.014000000');
internamente trabajamos siempre con segundos epoch (float)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

DAY_S = 86400.0
DAY_MS = 86_400_000


def parse_ts(value) -> float:
    """Timestamp -> segundos epoch UTC. Acepta texto con espacio o 'T', 'Z', offset '+01:00',
    hasta 9 decimales, solo fecha, o números (s o ms). Devuelve NaN si no se puede leer."""
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v
    s = str(value).strip()
    if not s:
        return math.nan
    if s.lstrip("-").replace(".", "", 1).isdigit():          # número como texto (p. ej. ms)
        return parse_ts(float(s))
    s = s.replace("T", " ")
    offset = 0.0
    if s.endswith("Z"):
        s = s[:-1]
    elif len(s) > 19 and s[-6] in "+-" and s[-3] == ":":
        sign = 1.0 if s[-6] == "+" else -1.0
        offset = sign * (int(s[-5:-3]) * 3600 + int(s[-2:]) * 60)
        s = s[:-6]
    main, _, frac = s.partition(".")
    try:
        fmt = "%Y-%m-%d %H:%M:%S" if len(main) > 10 else "%Y-%m-%d"
        dt = datetime.strptime(main, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return math.nan
    out = dt.timestamp() - offset
    digits = "".join(ch for ch in frac if ch.isdigit())[:9]
    if digits:
        out += int(digits.ljust(9, "0")) / 1e9
    return out


def to_float(value, default: float = math.nan) -> float:
    """Número que puede llegar como texto (ClickHouse entrecomilla los enteros de 64 bits)."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
