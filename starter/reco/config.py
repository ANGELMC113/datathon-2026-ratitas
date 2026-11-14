"""Configuración: rutas, credenciales (.env) y parámetros del sistema, todo en un sitio.

La API key nunca va en el código: se lee de .env o del entorno.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Carga KEY=VALUE de .env en os.environ sin pisar lo que ya exista. Busca el .env en esta
    carpeta y, si no está, en la carpeta padre (el repo, si ratitas_v2/ está dentro)."""
    if path is None:
        path = next((p for p in (ROOT / ".env", ROOT.parent / ".env") if p.exists()), ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

ARTIFACTS = Path(os.environ.get("RECO_ARTIFACTS", str(ROOT / "artifacts")))
LOGS = Path(os.environ.get("RECO_LOGS", str(ROOT / "logs")))
LIVE_ARTIFACTS = ARTIFACTS / "live.pkl"   # datos precalculados que usa el servicio en vivo
MODEL_DIR = ARTIFACTS / "model"           # modelo entrenado (model.json + model.txt / model.pkl)


@dataclass(frozen=True)
class Settings:
    api_url: str
    api_key: str
    team: str

    # --- bucle en vivo ---
    poll_interval: float = 0.25      # s entre lecturas de shop.requests
    request_window_s: int = 20       # ventana de la consulta de peticiones (deduplicamos con `seen`)
    drop_after_s: float = 4.0        # peticiones más viejas ya no se pueden servir a tiempo: se ignoran
    deadline_s: float = 3.0          # plazo oficial desde el ts de la petición
    batch_budget_s: float = 0.6      # cómputo máximo por lote; si se supera, el resto va con fallback
    http_timeout_s: float = 2.0      # timeout de las consultas en segundo plano rápidas
    pull_timeout_s: float = 1.0      # una lectura colgada se abandona pronto: la siguiente ya va en camino
    max_inflight_pulls: int = 3      # lecturas solapadas como máximo
    push_timeout_s: float = 1.5      # timeout de cada inserción (nunca más corto: la cortaríamos sin motivo)
    hedge_after_s: float = 0.6       # si la inserción no responde en este tiempo, se envía otra copia

    # --- recomendador ---
    n_candidates: int = 200
    n_items: int = 10
    n_explore: int = 3               # huecos 8-10 para Thompson sampling
    explore_prior: float = 50.0      # peso del score del modelo como prior del bandit
    max_recent: int = 10             # vistas recientes que usamos (las más nuevas)
    recency_decay: float = 0.7       # peso de la vista k-ésima más reciente = decay**k

    # --- refresco del estado en vivo (segundos) ---
    every_catalog: float = 2.0
    every_trends: float = 30.0
    every_purchases: float = 15.0
    every_feedback: float = 15.0
    every_pop: float = 180.0
    every_clock: float = 300.0
    every_model_check: float = 10.0


def settings() -> Settings:
    missing = [k for k in ("TEAM_ID", "RAWTREE_API_KEY") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Faltan variables en .env o en el entorno: {', '.join(missing)}")
    return Settings(
        api_url=os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").rstrip("/"),
        api_key=os.environ["RAWTREE_API_KEY"],
        team=os.environ["TEAM_ID"],
    )
