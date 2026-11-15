"""Cliente RawTree: query(sql) e insert(table, rows), con reintentos solo cuando tiene sentido.

- Una sesión HTTP por hilo (requests.Session no es segura entre hilos) con conexiones reutilizadas,
  que ahorran el apretón de manos TLS en cada llamada: importante para la latencia.
- Se reintenta en errores de red, 429 y 5xx, con espera creciente. Los 4xx no: el SQL está mal.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from . import config

Record = dict[str, Any]
RETRY_STATUS = {429, 500, 502, 503, 504}


class RawTreeError(RuntimeError):
    pass


class RawTreeClient:
    def __init__(self, api_url: str, api_key: str, team: str, *, timeout: float = 30.0,
                 retries: int = 2, backoff: float = 0.2) -> None:
        if not api_key:
            raise ValueError("RAWTREE_API_KEY vacía")
        self.api_url = api_url.rstrip("/")
        self.team = team
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._api_key = api_key
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {self._api_key}"
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return session

    def _post(self, url: str, *, payload: Any, params: dict | None, timeout: float | None,
              retries: int | None) -> requests.Response:
        timeout = self.timeout if timeout is None else timeout
        retries = self.retries if retries is None else retries
        last: Exception = RawTreeError("sin intentos")
        for attempt in range(retries + 1):
            try:
                resp = self._session().post(url, json=payload, params=params, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = RawTreeError(f"error de red en {url}: {exc}")
            else:
                if resp.status_code < 400:
                    return resp
                msg = f"HTTP {resp.status_code} en {url}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUS:
                    raise RawTreeError(msg)
                last = RawTreeError(msg)
            if attempt < retries:
                time.sleep(min(self.backoff * (2 ** attempt), 2.0))
        raise last

    def query(self, sql: str, *, timeout: float | None = None, retries: int | None = None) -> list[Record]:
        resp = self._post(f"{self.api_url}/v1/query", payload={"sql": sql}, params=None,
                          timeout=timeout, retries=retries)
        try:
            return resp.json()["data"]
        except (ValueError, KeyError) as exc:
            raise RawTreeError(f"respuesta inesperada de /v1/query: {resp.text[:300]}") from exc

    def query_df(self, sql: str, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(self.query(sql, **kwargs))

    def insert(self, table: str, rows: list[Record], *, timeout: float | None = None,
               retries: int | None = None) -> None:
        """Inserta en team_NN.<table>. Un reintento tras un timeout puede duplicar filas idénticas;
        para submissions la organización cuenta solo una respuesta por petición."""
        if not rows:
            return
        self._post(f"{self.api_url}/v1/tables/{table}", payload=rows, params={"database": self.team},
                   timeout=timeout, retries=retries)


def from_env(**kwargs) -> RawTreeClient:
    s = config.settings()
    return RawTreeClient(s.api_url, s.api_key, s.team, **kwargs)
