"""Cliente RawTree compartido. Dueño: rol A (Shiqi). Lo usa todo el equipo.

Interfaz acordada (no cambiar sin avisar):
    query(sql: str) -> list[dict]          # filas de un SELECT
    insert(table: str, rows: list[dict])   # inserta en la base del equipo (TEAM_ID)

Punto de partida: query() y push() de starter/recommender.py.

TODO (paso 2 de la guía):
- Leer TEAM_ID, RAWTREE_API_KEY y RAWTREE_API_URL de .env / entorno (nunca la key en el código).
- Timeout HTTP corto (el starter usa 10 s: ¿cuánto te puedes permitir dentro de 3 s?).
- Reintentar solo los errores que tiene sentido repetir (red, 5xx, 429), con espera creciente.
- Mensajes de error con el código HTTP y el principio de la respuesta.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests


def load_config(env_path: Path = Path(__file__).resolve().parent / ".env") -> dict[str, str]:
    # 1. Si env_path existe, recorre sus líneas
    if not env_path.exists():
        return
    else :
        for line in env_path.read_text().splitlines():
        # 2. Salta las líneas vacías y las que empiezan por "#"
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                    continue
            # 3. Parte cada línea por el primer "="
            var, _, value = line.partition("=")
            # 4. Guárdala en os.environ sin pisar lo que ya exista
            os.environ.setdefault(var.strip(), value.strip())
    # 5. Devuelve un dict con value, RAWTREE_API_KEY y RAWTREE_API_URL
    return {
            "TEAM_ID": os.environ["TEAM_ID"],
            "RAWTREE_API_KEY": os.environ["RAWTREE_API_KEY"],
            "RAWTREE_API_URL": os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
        }
    

CONFIG = load_config()
API = CONFIG["RAWTREE_API_URL"]
TEAM = CONFIG["TEAM_ID"]

http = requests.Session()
http.headers["Authorization"] = f"Bearer {CONFIG['RAWTREE_API_KEY']}"


def query(sql: str, timeout: float = 10) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)

