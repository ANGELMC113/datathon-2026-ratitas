"""Entrenamiento y validación offline (rol C), con split temporal y sin fuga de futuro.

    python train.py --build              # descarga de RawTree + artefactos + entrena (la primera vez)
    python train.py                      # reentrena con lo descargado (artifacts/offline/)
    python train.py --with-feedback      # añade lo aprendido del stream (logs + team_NN.feedback)

Esquema temporal (días del histórico de shop.history_sessions):

    días 1-40  -> artefactos A (co-visitación, carrusel...)   | días 41-50 -> filas de ENTRENAMIENTO
    días 1-50  -> artefactos B                                 | días 51-60 -> filas de VALIDACIÓN
    días 1-60  -> artefactos LIVE (los que usa serve_v2.py)

Cada fila solo ve datos anteriores a su impresión: los artefactos se cortan antes y la popularidad se
calcula con los días previos. El modelo final se reentrena con A+B (y el feedback si se pide) usando
el número de árboles elegido en validación.

Salida: artifacts/live.pkl (para el servicio) y artifacts/model/ (model.json + modelo). El servicio
recarga el modelo en caliente en cuanto cambia model.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reco import config, sql
from reco.artifacts import Artifacts, DailyPop
from reco.catalog import ProductTable
from reco.client import from_env
from reco.features import FEATURE_NAMES
from reco.model import HeuristicScorer, Scorer
from reco.offline import (build_artifacts, download_daily, download_purchase_paths, download_impressions, evaluate_policies,
                          feedback_impressions, fit_content, history_range, impressions_to_rows,
                          download_purchases, load_catalog, log, mark_prev_bought, offline_state, row_metrics)
from reco.timeutil import DAY_MS

OFFLINE = config.ARTIFACTS / "offline"


# ----------------------------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------------------------
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": 7, "num_threads": 0,
}


def backend_name(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except ImportError:
        log("lightgbm no está instalado: uso HistGradientBoosting de scikit-learn (pip install lightgbm)")
        return "sklearn"


# Espacio de búsqueda de hiperparámetros (train.py --tune N). El primer punto es la configuración
# por defecto, así que la búsqueda nunca empeora en validación.
SEARCH_SPACE = {
    "num_leaves": [15, 31, 63, 127],          # complejidad de cada árbol
    "min_data_in_leaf": [50, 100, 200, 500],  # filas mínimas por hoja: más = menos sobreajuste
    "learning_rate": [0.03, 0.05, 0.1],
    "feature_fraction": [0.6, 0.8, 1.0],      # fracción de features que ve cada árbol
    "lambda_l2": [0.0, 1.0, 5.0],             # regularización
}
BEST_PARAMS = "best_params.json"


def sample_params(n: int, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [{k: LGB_PARAMS[k] for k in SEARCH_SPACE}]
    seen = {json.dumps(out[0], sort_keys=True)}
    while len(out) < n and len(seen) < 500:
        cand = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        key = json.dumps(cand, sort_keys=True, default=float)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def _sklearn_params(params: dict | None) -> dict:
    """Traducción a HistGradientBoosting (solo si falta lightgbm)."""
    p = {**{k: LGB_PARAMS[k] for k in SEARCH_SPACE}, **(params or {})}
    out = dict(learning_rate=float(p["learning_rate"]), max_leaf_nodes=int(p["num_leaves"]),
               min_samples_leaf=int(p["min_data_in_leaf"]), l2_regularization=float(p["lambda_l2"]),
               random_state=7)
    if float(p["feature_fraction"]) < 1.0:
        out["max_features"] = float(p["feature_fraction"])
    return out


def fit(backend: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, Xv: np.ndarray | None = None,
        yv: np.ndarray | None = None, wv: np.ndarray | None = None, rounds: int | None = None,
        params: dict | None = None, verbose: bool = True):
    """Devuelve (modelo, n_árboles). Con validación usa parada temprana; sin ella, `rounds` fijo.
    `params` sobrescribe LGB_PARAMS (los del ajuste de hiperparámetros)."""
    if backend == "lightgbm":
        import lightgbm as lgb
        p = {**LGB_PARAMS, **(params or {})}
        dtrain = lgb.Dataset(X, y, weight=w, feature_name=list(FEATURE_NAMES), free_raw_data=False)
        if Xv is not None:
            dvalid = lgb.Dataset(Xv, yv, weight=wv, reference=dtrain)
            callbacks = [lgb.early_stopping(100, verbose=False)]
            if verbose:
                callbacks.append(lgb.log_evaluation(100))
            booster = lgb.train(p, dtrain, num_boost_round=3000, valid_sets=[dvalid], callbacks=callbacks)
            return booster, max(int(booster.best_iteration), 1)
        booster = lgb.train(p, dtrain, num_boost_round=int(rounds or 300))
        return booster, int(rounds or 300)

    from sklearn.ensemble import HistGradientBoostingClassifier
    params = _sklearn_params(params)
    if Xv is not None:
        # sklearn no acepta un conjunto de validación externo: buscamos el nº de árboles a mano
        best, best_iter, model = np.inf, 50, None
        from sklearn.metrics import log_loss
        clf = HistGradientBoostingClassifier(max_iter=50, warm_start=True, early_stopping=False, **params)
        stall = 0
        for it in range(50, 1501, 50):
            clf.set_params(max_iter=it)
            clf.fit(X, y, sample_weight=w)
            loss = log_loss(yv, np.clip(clf.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6), sample_weight=wv)
            if loss < best - 1e-6:
                best, best_iter, stall = loss, it, 0
            else:
                stall += 1
                if stall >= 3:
                    break
        model = HistGradientBoostingClassifier(max_iter=best_iter, early_stopping=False, **params)
        model.fit(X, y, sample_weight=w)
        return model, best_iter
    model = HistGradientBoostingClassifier(max_iter=int(rounds or 300), early_stopping=False, **params)
    model.fit(X, y, sample_weight=w)
    return model, int(rounds or 300)


class _InMemoryScorer(Scorer):
    def __init__(self, backend: str, model, version: str) -> None:
        self.kind, self.model, self.version = backend, model, version

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "lightgbm":
            return self.model.predict(X, num_threads=1)
        return self.model.predict_proba(X)[:, 1]


def predict(backend: str, model, X: np.ndarray) -> np.ndarray:
    return _InMemoryScorer(backend, model, "tmp").predict(X)


def importance(backend: str, model) -> pd.DataFrame | None:
    if backend != "lightgbm":
        return None
    gain = model.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": FEATURE_NAMES, "gain": gain})
    df["gain_pct"] = 100 * df["gain"] / max(df["gain"].sum(), 1e-9)
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def save_model(backend: str, model, version: str, meta: dict, model_dir: Path) -> None:
    """Escribe el modelo y DESPUÉS model.json (que es lo que vigila el servicio), cada uno de forma
    atómica, y deja una copia en artifacts/models/<version>/ para poder volver atrás."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if backend == "lightgbm":
        fname = "model.txt"
        tmp = model_dir / (fname + ".tmp")
        model.save_model(str(tmp))
    else:
        fname = "model.pkl"
        tmp = model_dir / (fname + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(model, f)
    tmp.replace(model_dir / fname)
    meta = {**meta, "kind": backend, "version": version, "file": fname, "features": list(FEATURE_NAMES),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp_meta = model_dir / "model.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2, default=float))
    tmp_meta.replace(model_dir / "model.json")
    archive = model_dir.parent / "models" / version
    archive.mkdir(parents=True, exist_ok=True)
    for f in (fname, "model.json"):
        shutil.copy2(model_dir / f, archive / f)
    log(f"modelo guardado en {model_dir} (copia en {archive})")


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
def build(args) -> None:
    client = from_env(timeout=600, retries=2)
    t0, t1 = history_range(client)
    day0 = t0 // DAY_MS
    cut_a = (day0 + args.train_from_day) * DAY_MS
    cut_b = (day0 + args.valid_from_day) * DAY_MS
    end = t1 + 1
    log(f"histórico {pd.Timestamp(t0, unit='ms')} -> {pd.Timestamp(t1, unit='ms')}; "
        f"entrenar con [{pd.Timestamp(cut_a, unit='ms')}, {pd.Timestamp(cut_b, unit='ms')}), validar con el resto")

    catalog = load_catalog(client)
    product_ids = catalog["product_id"].tolist()
    content, emb = fit_content(catalog)
    log(f"catálogo {len(catalog)} productos; embeddings {emb.shape}")
    tables = ("shop.history_events", "shop.events") if args.include_live_events else ("shop.history_events",)
    daily = download_daily(client, product_ids, tables)

    OFFLINE.mkdir(parents=True, exist_ok=True)
    with open(OFFLINE / "base.pkl", "wb") as f:
        pickle.dump({"catalog": catalog, "daily": daily, "daily_tables": tables,
                     "cuts": (t0, cut_a, cut_b, end)}, f)

    paths = download_purchase_paths(client, t0, end)
    sub_of = dict(zip(catalog["product_id"].astype(str), catalog["subcategory"].astype(str)))
    for name, until in (("A", cut_a), ("B", cut_b), ("LIVE", end)):
        log(f"artefactos {name} (histórico < {pd.Timestamp(until, unit='ms')})")
        art = build_artifacts(client, product_ids, content, emb, until, paths=paths, sub_of=sub_of)
        art.save(OFFLINE / f"artifacts_{name}.pkl")
        if name == "LIVE":
            art.save(config.LIVE_ARTIFACTS)
            log(f"artefactos del servicio en {config.LIVE_ARTIFACTS}")

    imps_a = download_impressions(client, cut_a, cut_b, args.keep_permille)
    imps_b = download_impressions(client, cut_b, end, args.keep_permille)
    imps_a.to_pickle(OFFLINE / "imps_A.pkl")
    imps_b.to_pickle(OFFLINE / "imps_B.pkl")
    with open(OFFLINE / "build.json", "w") as f:
        json.dump({"keep_permille": args.keep_permille, "cuts": [t0, cut_a, cut_b, end]}, f)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="descargar datos y construir artefactos")
    ap.add_argument("--version", default=None, help="model_version, p. ej. v2-lgbm")
    ap.add_argument("--backend", choices=("auto", "lightgbm", "sklearn"), default="auto")
    ap.add_argument("--train-from-day", type=int, default=40)
    ap.add_argument("--valid-from-day", type=int, default=50)
    ap.add_argument("--keep-permille", type=int, default=250,
                    help="‰ de impresiones SIN compra que se guardan (las de compra, todas)")
    ap.add_argument("--with-feedback", action="store_true", help="añadir impresiones propias del stream")
    ap.add_argument("--include-live-events", action="store_true",
                    help="incluir shop.events en la popularidad diaria (necesario con --with-feedback)")
    ap.add_argument("--keep-rebuys", action="store_true",
                    help="no quitar del entrenamiento lo que el usuario ya había comprado (antes de v5)")
    ap.add_argument("--default-params", action="store_true",
                    help="ignorar best_params.json y usar los hiperparámetros de serie (como v2/v3)")
    ap.add_argument("--eval-imps", type=int, default=5000)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="dónde guardar el modelo (por defecto artifacts/model, el que usa el servicio)")
    ap.add_argument("--holdout-minutes", type=int, default=0,
                    help="con --with-feedback: deja fuera los últimos N minutos de feedback para comparar "
                         "el modelo nuevo con el que está desplegado")
    ap.add_argument("--threads", type=int, default=0, help="hilos de LightGBM (0 = todos)")
    ap.add_argument("--feedback-weight", type=float, default=1.0,
                    help="peso de las impresiones propias del stream frente al histórico (p. ej. 5)")
    ap.add_argument("--feedback-half-life", type=float, default=0.0,
                    help="horas: el feedback pierde la mitad de su peso cada N horas (0 = todo igual)")
    ap.add_argument("--compare-dir", type=Path, default=None,
                    help="modelo con el que comparar en lo reservado (por defecto artifacts/model)")
    ap.add_argument("--tune", type=int, default=0,
                    help="probar N combinaciones de hiperparámetros en validación y guardar la mejor "
                         f"(artifacts/offline/{BEST_PARAMS}); los entrenamientos siguientes la usan")
    args = ap.parse_args(argv)
    if args.with_feedback:
        args.include_live_events = True
    return args


def load_folds(args) -> dict:
    """Datos descargados + filas de entrenamiento (días 41-50) y validación (51-60)."""
    if args.build or not (OFFLINE / "base.pkl").exists():
        build(args)
    with open(OFFLINE / "base.pkl", "rb") as f:
        base = pickle.load(f)
    keep = json.loads((OFFLINE / "build.json").read_text())["keep_permille"]
    neg_w = 1000.0 / keep                       # compensa el muestreo de impresiones sin compra
    catalog: pd.DataFrame = base["catalog"]
    if args.with_feedback or (args.include_live_events and "shop.events" not in base.get("daily_tables", ())):
        # con feedback siempre se actualiza: los días del stream crecen y la popularidad debe incluirlos
        log("actualizando la popularidad diaria con shop.events (para las impresiones del stream)")
        base["daily"] = download_daily(from_env(timeout=600, retries=2), catalog["product_id"].tolist(),
                                       ("shop.history_events", "shop.events"))
        base["daily_tables"] = ("shop.history_events", "shop.events")
        with open(OFFLINE / "base.pkl", "wb") as f:
            pickle.dump(base, f)
    daily: DailyPop = base["daily"]
    art_a = Artifacts.load(OFFLINE / "artifacts_A.pkl")
    art_b = Artifacts.load(OFFLINE / "artifacts_B.pkl")
    products = ProductTable.from_frame(catalog, emb=art_b.emb, order=art_b.product_ids)
    imps_a = pd.read_pickle(OFFLINE / "imps_A.pkl")
    imps_b = pd.read_pickle(OFFLINE / "imps_B.pkl")
    if not args.keep_rebuys:
        imps_a, imps_b = drop_rebuys(args, base, imps_a, imps_b)
    st_a = offline_state(products, art_a, daily)
    st_b = offline_state(products, art_b, daily)
    t = time.time()
    rows_a = impressions_to_rows(st_a, imps_a, neg_w)
    rows_b = impressions_to_rows(st_b, imps_b, neg_w)
    log(f"filas: entrenamiento {len(rows_a['y'])} ({int(rows_a['y'].sum())} compras), "
        f"validación {len(rows_b['y'])} ({int(rows_b['y'].sum())} compras) en {time.time() - t:.0f} s")
    return {"base": base, "catalog": catalog, "daily": daily, "st_b": st_b, "imps_b": imps_b,
            "rows_a": rows_a, "rows_b": rows_b}


def drop_rebuys(args, base: dict, imps_a: pd.DataFrame, imps_b: pd.DataFrame):
    """Marca en cada impresión lo que el usuario ya había comprado para quitarlo de las filas.
    Las compras se descargan una vez y se guardan en artifacts/offline/purchases.pkl."""
    path = OFFLINE / "purchases.pkl"
    _, cut_a, _, end = base["cuts"]
    if args.build or not path.exists():
        download_purchases(from_env(timeout=600, retries=2), cut_a - 7 * DAY_MS, end).to_pickle(path)
    purchases = pd.read_pickle(path)
    imps_a, imps_b = mark_prev_bought(imps_a, purchases), mark_prev_bought(imps_b, purchases)
    for name, imps in (("entrenamiento", imps_a), ("validación", imps_b)):
        rebuy = sum(1 for b, prev in zip(imps["purchased"], imps["prev_bought"]) if b and b in prev)
        n_buy = int((imps["purchased"] != "").sum())
        log(f"recompras quitadas de {name}: {rebuy} de {n_buy} compras "
            f"({100 * rebuy / max(n_buy, 1):.1f} %; el simulador evaluado casi no las tiene)")
    return imps_a, imps_b


def tune(backend: str, rows_a: dict, rows_b: dict, n: int) -> dict:
    """Búsqueda aleatoria de hiperparámetros; gana el menor logloss en validación (queremos
    probabilidades bien calibradas, no solo buen orden). Guarda la mejor en best_params.json."""
    results = []
    for k, params in enumerate(sample_params(n)):
        t = time.time()
        model, it = fit(backend, rows_a["X"], rows_a["y"], rows_a["w"], rows_b["X"], rows_b["y"], rows_b["w"],
                        params=params, verbose=False)
        m = row_metrics(predict(backend, model, rows_b["X"]), rows_b)
        results.append({**params, "trees": it, **{x: m.get(x) for x in ("logloss", "auc", "mrr_in_shown_model")},
                        "seconds": round(time.time() - t)})
        log(f"  ajuste {k + 1}/{n}: {params} -> logloss {m.get('logloss', float('nan')):.6f} "
            f"AUC {m.get('auc', float('nan')):.4f} ({it} árboles)")
    table = pd.DataFrame(results).sort_values("logloss")
    print(table.to_string(index=False))
    table.to_csv(OFFLINE / "tuning.csv", index=False)
    best = {k: table.iloc[0][k] for k in SEARCH_SPACE}
    best = {k: (int(v) if k in ("num_leaves", "min_data_in_leaf") else float(v)) for k, v in best.items()}
    (OFFLINE / BEST_PARAMS).write_text(json.dumps(best, indent=2))
    log(f"mejores hiperparámetros: {best} (guardados en {OFFLINE / BEST_PARAMS})")
    return best


def load_best_params() -> dict | None:
    path = OFFLINE / BEST_PARAMS
    if path.exists():
        params = json.loads(path.read_text())
        log(f"uso los hiperparámetros ajustados de {path}: {params}")
        return params
    return None


def validate(args, backend: str, folds: dict, params: dict | None) -> tuple:
    """Modelo entrenado con días 41-50 y evaluado en 51-60: métricas y nº de árboles."""
    rows_a, rows_b = folds["rows_a"], folds["rows_b"]
    model, best_iter = fit(backend, rows_a["X"], rows_a["y"], rows_a["w"], rows_b["X"], rows_b["y"], rows_b["w"],
                           params=params)
    metrics = {"valid_rows": row_metrics(predict(backend, model, rows_b["X"]), rows_b), "best_iter": best_iter,
               "params": params or {k: LGB_PARAMS[k] for k in SEARCH_SPACE}}
    log(f"validación (filas mostradas) modelo: {metrics['valid_rows']}")
    log(f"validación (filas mostradas) heurística: {row_metrics(HeuristicScorer().predict(rows_b['X']), rows_b)}")
    imp = importance(backend, model)
    if imp is not None:
        print(imp.head(15).to_string(index=False))
    log("evaluación de la lista completa (candidatos + orden) en días 51-60 ...")
    summary = evaluate_policies(folds["st_b"], folds["imps_b"],
                                {"heuristica": HeuristicScorer(), "modelo": _InMemoryScorer(backend, model, "valid")},
                                max_imps=args.eval_imps)
    print(summary.round(4).to_string())
    metrics["policies"] = summary.round(5).to_dict()
    return model, best_iter, metrics, imp


def feedback_rows(args, folds: dict) -> tuple[dict | None, dict | None]:
    """Nuestras propias impresiones del stream (las últimas `holdout_minutes` se reservan)."""
    client = from_env(timeout=600, retries=2)
    imps_fb = feedback_impressions(client, client.team, config.LOGS)
    if not len(imps_fb):
        return None, None
    art_live = Artifacts.load(OFFLINE / "artifacts_LIVE.pkl")
    prod_live = ProductTable.from_frame(folds["catalog"], emb=art_live.emb, order=art_live.product_ids)
    prod_live = prod_live.apply_updates(client.query(sql.catalog_updates_since(0), timeout=120),
                                        art_live.content, live_since_s=folds["base"]["cuts"][3] / 1000.0)
    st_fb = offline_state(prod_live, art_live, folds["daily"])
    rows_hold = None
    if args.holdout_minutes > 0:
        cut = float(imps_fb["ts_ms"].max()) - args.holdout_minutes * 60_000
        imps_hold, imps_fb = imps_fb[imps_fb["ts_ms"] > cut], imps_fb[imps_fb["ts_ms"] <= cut]
        rows_hold = impressions_to_rows(st_fb, imps_hold, 1.0)
        log(f"feedback: {len(imps_fb)} impresiones para entrenar, {len(imps_hold)} reservadas "
            f"(últimos {args.holdout_minutes} min, {int(rows_hold['y'].sum())} compras) para comparar")
    rows_fb = impressions_to_rows(st_fb, imps_fb, 1.0)
    rows_fb["w"] = rows_fb["w"] * args.feedback_weight
    if args.feedback_half_life > 0 and len(rows_fb["y"]):
        # lo reciente pesa más: cada `half_life` horas de antigüedad, la mitad de peso
        age_h = (float(imps_fb["ts_ms"].max()) - imps_fb["ts_ms"].to_numpy(dtype=float)) / 3.6e6
        decay = 0.5 ** (age_h / args.feedback_half_life)
        rows_fb["w"] = rows_fb["w"] * decay[rows_fb["g"]].astype(np.float32)
        log(f"feedback con vida media de {args.feedback_half_life} h: peso medio {decay.mean():.2f}")
    return (rows_fb if len(rows_fb["y"]) else None), rows_hold


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    LGB_PARAMS["num_threads"] = args.threads
    folds = load_folds(args)
    backend = backend_name(args.backend)

    params = tune(backend, folds["rows_a"], folds["rows_b"], args.tune) if args.tune > 0 else (None if args.default_params else load_best_params())
    model, best_iter, metrics, imp = validate(args, backend, folds, params)

    # --- modelo final: A + B (+ feedback), con el nº de árboles elegido en validación ---
    rows = [folds["rows_a"], folds["rows_b"]]
    rows_hold = None
    if args.with_feedback:
        rows_fb, rows_hold = feedback_rows(args, folds)
        if rows_fb is not None:
            metrics["feedback_rows"] = row_metrics(predict(backend, model, rows_fb["X"]), rows_fb)
            log(f"modelo de validación sobre el feedback propio: {metrics['feedback_rows']}")
            rows.append(rows_fb)
    rounds = max(int(best_iter * 1.1), 50)
    final, _ = fit(backend, np.vstack([r["X"] for r in rows]), np.concatenate([r["y"] for r in rows]),
                   np.concatenate([r["w"] for r in rows]), rounds=rounds, params=params)
    version = args.version or f"v2-{backend}-{time.strftime('%d%H%M')}"

    # --- comparación justa con el modelo desplegado: los últimos minutos que ninguno ha visto ---
    if rows_hold is not None and len(rows_hold["y"]):
        from reco.model import load_scorer
        current = load_scorer(args.compare_dir or config.MODEL_DIR)
        metrics["holdout"] = {
            "n_rows": int(len(rows_hold["y"])), "n_buys": int(rows_hold["y"].sum()),
            "candidate": row_metrics(predict(backend, final, rows_hold["X"]), rows_hold),
            "current": row_metrics(np.asarray(current.predict(rows_hold["X"])), rows_hold),
            "current_version": current.version,
        }
        log(f"reservado · nuevo {version}: {metrics['holdout']['candidate']}")
        log(f"reservado · desplegado {current.version}: {metrics['holdout']['current']}")

    if imp is not None:
        imp.to_csv(OFFLINE / f"importance_{version}.csv", index=False)
    if not args.no_save:
        save_model(backend, final, version, {"metrics": metrics, "rounds": rounds},
                   args.model_dir or config.MODEL_DIR)
    log(f"listo: {version}. Añade una línea a CHANGELOG.md con estas métricas.")


if __name__ == "__main__":
    main()
