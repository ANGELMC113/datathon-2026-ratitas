"""Etapa 2b: probabilidad de compra de cada fila (rol C).

- LightGBM binario (etiqueta = compra) entrenado con train.py.
- Si lightgbm no está instalado, train.py usa HistGradientBoosting de scikit-learn (el mismo tipo de
  modelo: árboles potenciados por gradiente).
- Si no hay modelo o no cuadra con las features actuales, una heurística con pesos fijos: el
  servicio nunca se queda sin poder ordenar.
"""
from __future__ import annotations

import json
import os
import pickle
import threading
import time
from pathlib import Path

import numpy as np

from .features import F, FEATURE_NAMES


class Scorer:
    kind = "base"
    version = "v0"

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HeuristicScorer(Scorer):
    """Suma ponderada de señales (la versión v1 sin modelo). Pesos elegidos a mano, explicables."""
    kind = "heuristic"

    WEIGHTS = {"covis_buy": 3.0, "covis_view": 2.0, "covis_b2b": 1.0, "content_knn": 1.0,
               "content_sim": 0.5, "in_recent": 0.3, "cat_share": 0.3, "sub_share": 0.4}

    def __init__(self, version: str = "v1-heuristic") -> None:
        self.version = version

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = np.zeros(len(X), dtype=np.float64)
        for name, w in self.WEIGHTS.items():
            s += w * np.nan_to_num(X[:, F[name]].astype(np.float64), nan=0.0)
        # popularidad en escala log para que no domine, y un pequeño peso a la conversión
        s += 0.15 * np.log1p(np.nan_to_num(X[:, F["pop_buy_1d"]], nan=0.0))
        s += 0.10 * np.log1p(np.nan_to_num(X[:, F["pop_buy_7d"]], nan=0.0))
        s += 0.05 * np.log1p(np.nan_to_num(X[:, F["pop_buy_all"]], nan=0.0))
        s += 2.0 * np.nan_to_num(X[:, F["conv_7d"]], nan=0.0)
        return 1.0 / (1.0 + np.exp(-(s - 3.0)))          # a (0, 1) para que el bandit pueda usarlo


class LightGBMScorer(Scorer):
    kind = "lightgbm"

    def __init__(self, model_file: Path, version: str) -> None:
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(model_file))
        self.version = version

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 200 filas: un hilo es más rápido que repartir el trabajo
        return self.booster.predict(X, num_threads=1)


class SklearnScorer(Scorer):
    kind = "sklearn"

    def __init__(self, model_file: Path, version: str) -> None:
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
        self.version = version

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def load_scorer(model_dir: Path) -> Scorer:
    """Carga el modelo de `model_dir` o devuelve la heurística si falta o no es compatible."""
    meta_path = model_dir / "model.json"
    if not meta_path.exists():
        return HeuristicScorer()
    meta = json.loads(meta_path.read_text())
    if tuple(meta.get("features", ())) != FEATURE_NAMES:
        # columnas distintas = predicciones basura; mejor la heurística y avisar
        print(f"[model] {meta_path}: las features no coinciden con FEATURE_NAMES; uso la heurística")
        return HeuristicScorer()
    version = str(meta.get("version", "v2-model"))
    kind = meta.get("kind")
    if kind == "lightgbm":
        return LightGBMScorer(model_dir / meta.get("file", "model.txt"), version)
    if kind == "sklearn":
        return SklearnScorer(model_dir / meta.get("file", "model.pkl"), version)
    raise ValueError(f"tipo de modelo desconocido: {kind}")


class ModelWatcher:
    """Recarga en caliente: si model.json cambia, carga el modelo nuevo en segundo plano y lo
    sustituye de golpe. Si la carga falla, se queda con el anterior."""

    def __init__(self, model_dir: Path, interval: float = 10.0) -> None:
        self.model_dir = model_dir
        self.interval = interval
        self.scorer: Scorer = HeuristicScorer()
        self._mtime = -1.0
        self.check()
        self._thread = threading.Thread(target=self._loop, name="model-watcher", daemon=True)

    def start(self) -> "ModelWatcher":
        self._thread.start()
        return self

    def check(self) -> None:
        path = self.model_dir / "model.json"
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime == self._mtime:
            return
        try:
            new = load_scorer(self.model_dir)
            new.predict(np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32))   # prueba antes de usarlo
        except Exception as exc:
            print(f"[model] no se pudo cargar {self.model_dir}: {exc!r}; sigo con {self.scorer.version}")
            self._mtime = mtime
            return
        self._mtime = mtime
        self.scorer = new
        print(f"[model] cargado {new.kind} {new.version}")

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval)
            self.check()
