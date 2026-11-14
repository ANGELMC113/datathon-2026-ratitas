"""Pruebas sin red: python -m pytest -q tests/test_reco.py"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reco import candidates as cand_mod                                   # noqa: E402
from reco.artifacts import Artifacts, CarouselStats, DailyPop, Neighbors   # noqa: E402
from reco.catalog import ProductTable, knn                                 # noqa: E402
from reco.config import Settings                                           # noqa: E402
from reco.features import F, FEATURE_NAMES, build_matrix                   # noqa: E402
from reco.model import HeuristicScorer, Scorer, load_scorer                # noqa: E402
from reco.offline import evaluate_policies, fit_content, impressions_to_rows, offline_state, row_metrics  # noqa: E402
from reco.recommender import Recommender, is_valid                         # noqa: E402
from reco.rerank import rerank_idx                                         # noqa: E402
from reco.session import make_session                                      # noqa: E402
from reco.state import State                                               # noqa: E402
from reco.timeutil import DAY_MS, parse_ts                                 # noqa: E402
from synth import DAY0, FakeClient, make_catalog, make_history             # noqa: E402


# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world():
    catalog = make_catalog(400)
    imps, covis, car, daily, sales = make_history(catalog)
    ids = catalog["product_id"].tolist()
    content, emb = fit_content(catalog)
    index = {p: i for i, p in enumerate(ids)}
    nb_idx, nb_sim = knn(emb, 20)
    buy_all = np.zeros(len(ids), dtype=np.float32)
    for p, n in sales.items():
        buy_all[index[p]] = n
    art = Artifacts(product_ids=ids, until_ms=(DAY0 + 60) * DAY_MS,
                    covis={"view": Neighbors.from_frame(covis, index, len(ids), 40),
                           "buy": Neighbors.from_frame(covis.sample(frac=0.5, random_state=0), index, len(ids), 40),
                           "b2b": Neighbors.empty(len(ids))},
                    content_knn=Neighbors.from_dense(nb_idx, np.maximum(nb_sim, 0)),
                    carousel=CarouselStats.from_frame(car, index, len(ids)), content=content, emb=emb,
                    buy_all=buy_all, best_sellers=sales.sort_values(ascending=False).index.tolist())
    dp = DailyPop.from_frames(daily, ids)
    products = ProductTable.from_frame(catalog, emb=emb, order=ids)
    return {"catalog": catalog, "imps": imps, "art": art, "daily": dp, "products": products, "sales": sales}


def live_like_state(world) -> State:
    return State(world["products"], world["art"], world["daily"].snapshot)


def a_request(world, n_views=3, seed=0, **kw) -> dict:
    rng = np.random.default_rng(seed)
    ids = world["products"].ids
    views = [{"product_id": ids[int(rng.integers(len(ids)))], "ts": f"2025-11-14 09:00:0{j}.123000000"}
             for j in range(n_views)]
    req = {"request_id": f"r_{seed}", "ts": "2025-11-14 09:00:10.000000000", "user_id": "u_1",
           "is_new_user": False, "context": {"device": "ios", "page": "product", "region": "X"},
           "recent_views": views}
    req.update(kw)
    return req


# ----------------------------------------------------------------------------------------------
def test_parse_ts():
    base = parse_ts("2026-11-14 09:00:00")
    assert base == pytest.approx(1794646800.0)
    assert parse_ts("2026-11-14 09:00:00.014000000") == pytest.approx(base + 0.014)
    assert parse_ts("2026-11-14T09:00:00Z") == pytest.approx(base)
    assert parse_ts("2026-11-14T10:00:00+01:00") == pytest.approx(base)
    assert parse_ts(1794646800000) == pytest.approx(base)
    assert parse_ts("1794646800000") == pytest.approx(base)
    assert parse_ts("2026-11-14") == pytest.approx(base - 9 * 3600)
    assert math.isnan(parse_ts("")) and math.isnan(parse_ts(None)) and math.isnan(parse_ts("basura"))


def test_neighbors_csr():
    index = {"a": 0, "b": 1, "c": 2}
    df = pd.DataFrame({"p": ["a", "a", "b", "x"], "q": ["b", "c", "a", "a"], "n": [3, 1, 5, 9]})
    nb = Neighbors.from_frame(df, index, 3, top_k=10)
    q, w = nb.get(0)
    assert list(q) == [1, 2] and w.sum() == pytest.approx(1.0) and w[0] == pytest.approx(0.75)
    assert nb.get(2) is None and nb.get(7) is None


def test_product_table_updates(world):
    pt = world["products"]
    pid = next(p for p in pt.ids if pt.available[pt.index[p]])
    i = pt.index[pid]
    base = world["catalog"].set_index("product_id").loc[pid].to_dict()
    upd = lambda change, ts, **kw: {**base, "product_id": pid, "change": change, "ts_ms": str(ts), **kw}
    pt2 = pt.apply_updates([upd("out_of_stock", 1, is_active=False)], world["art"].content)
    assert not pt2.available[i] and pt.available[i]          # la tabla original no cambia
    pt3 = pt2.apply_updates([upd("back_in_stock", 2, is_active=True), upd("price_change", 3, price_eur=1.5)],
                            world["art"].content)
    assert pt3.available[i] and pt3.price[i] == pytest.approx(1.5)
    new = {"product_id": "p_nuevo", "category": "Cat1", "subcategory": "Sub1_2", "brand": "BrandX",
           "price_eur": 9.99, "description": "BrandX Sub1_2 with a vintage look", "created_ms": "0",
           "is_active": True, "change": "new_product", "ts_ms": "5"}
    pt4 = pt3.apply_updates([new], world["art"].content)
    j = pt4.index["p_nuevo"]
    assert j == len(pt3) and pt4.available[j] and pt4.live_new[j]
    assert np.linalg.norm(pt4.emb[j]) == pytest.approx(1.0, abs=1e-3)   # tiene embedding de contenido
    assert pt4.ids[:len(pt3)] == pt3.ids                                  # los índices no se mueven


def test_candidates_and_features(world):
    st = live_like_state(world)
    req = a_request(world, 4)
    sess = make_session(st, req)
    cand = cand_mod.generate(st, sess, 200)
    assert len(cand) == len(set(cand.tolist())) <= 200
    assert all(st.products.available[i] for i in cand)
    # los vecinos de co-visitación de la última vista aparecen
    nb = st.artifacts.covis["view"].get(int(sess.recent[0]))
    assert len(set(nb[0][:5].tolist()) & set(cand.tolist())) > 0
    X = build_matrix(st, sess, cand)
    assert X.shape == (len(cand), len(FEATURE_NAMES)) and X.dtype == np.float32
    assert (X[:, F["position"]] == 0).all()
    rec_rows = X[:, F["in_recent"]] == 1
    assert rec_rows.sum() == len(set(sess.recent.tolist()) & set(cand.tolist()))
    assert np.isfinite(X[:, F["covis_view"]]).all()


def test_cold_user_still_gets_candidates(world):
    st = live_like_state(world)
    sess = make_session(st, a_request(world, 0, is_new_user=True))
    assert sess.cold and len(sess.recent) == 0
    cand = cand_mod.generate(st, sess, 200)
    assert len(cand) == 200


def test_rerank_always_valid(world):
    st = live_like_state(world)
    rng = np.random.default_rng(0)
    for k in range(200):
        req = a_request(world, int(rng.integers(0, 6)), seed=k)
        sess = make_session(st, req)
        n = int(rng.integers(0, 60))
        ranked = rng.choice(len(st.products), n, replace=True)        # con repetidos a propósito
        scores = np.sort(rng.random(n))[::-1]
        sess.bought = set(rng.choice(len(st.products), 5).tolist())
        out = rerank_idx(st, ranked, scores, sess, rng)
        ids = [st.products.ids[i] for i in out]
        assert is_valid(ids, st)
        assert not (set(out) & sess.bought)


def test_recommender_robust_to_bad_input(world):
    st = live_like_state(world)
    rec = Recommender(st, HeuristicScorer(), seed=0)
    bad = [
        a_request(world, 3),
        a_request(world, 0),
        a_request(world, 2, context='{"device": "web", "page": "home"}'),
        a_request(world, 2, context=None),
        a_request(world, 0, recent_views=[{"product_id": "p_no_existe", "ts": "x"}, "p_000001"]),
        a_request(world, 0, recent_views='[{"product_id": "p_000002", "ts": "2025-11-14 09:00:00"}]'),
        a_request(world, 0, recent_views=None, ts=None, user_id=None),
        {"request_id": "r", "context.device": "ios", "context.page": "home", "recent_views": []},
    ]
    for req in bad:
        r = rec.recommend(req)
        assert is_valid(r.items, st), req


def test_offline_training_learns(world):
    st = offline_state(world["products"], world["art"], world["daily"])
    imps = world["imps"]
    rows = impressions_to_rows(st, imps.iloc[:4000], neg_weight=1.0)
    rows_v = impressions_to_rows(st, imps.iloc[4000:], neg_weight=1.0)
    assert rows["X"].shape[1] == len(FEATURE_NAMES) and rows["y"].sum() > 20
    import train
    model, it = train.fit("sklearn", rows["X"], rows["y"], rows["w"], rows_v["X"], rows_v["y"], rows_v["w"])
    p = train.predict("sklearn", model, rows_v["X"])
    m = row_metrics(p, rows_v)
    assert m["auc"] > 0.65, m
    assert m["mrr_in_shown_model"] > m["mrr_in_shown_old_order"], m
    summary = evaluate_policies(st, imps.iloc[4000:], {"heuristica": HeuristicScorer(),
                                                       "modelo": train._InMemoryScorer("sklearn", model, "t")})
    assert summary.loc["modelo", "todas"] >= summary.loc["baseline", "todas"]


def test_model_save_and_load(world, tmp_path):
    import train
    st = offline_state(world["products"], world["art"], world["daily"])
    rows = impressions_to_rows(st, world["imps"].iloc[:1500], 1.0)
    model, _ = train.fit("sklearn", rows["X"], rows["y"], rows["w"], rounds=30)
    train.save_model("sklearn", model, "v-test", {}, tmp_path / "model")
    sc = load_scorer(tmp_path / "model")
    assert sc.kind == "sklearn" and sc.version == "v-test"
    assert sc.predict(rows["X"][:5]).shape == (5,)
    # si las features no cuadran, heurística (nunca columnas desalineadas)
    meta = tmp_path / "model" / "model.json"
    meta.write_text(meta.read_text().replace('"position"', '"otra_cosa"'))
    assert isinstance(load_scorer(tmp_path / "model"), HeuristicScorer)
    assert isinstance(load_scorer(tmp_path / "no_existe"), HeuristicScorer)


# ----------------------------------------------------------------------------------------------
# Servicio en vivo con un cliente falso
# ----------------------------------------------------------------------------------------------
def make_server(world, scorer: Scorer | None = None, **settings_kw):
    import serve_v2
    from reco.live import LiveState
    client = FakeClient(world["catalog"], world["sales"])
    s = Settings(api_url="http://x", api_key="k", team="team_99", **settings_kw)
    state = LiveState(client, world["art"], s)
    state.warm_up()
    rec = Recommender(state, scorer or HeuristicScorer(), seed=0)
    server = serve_v2.Server(client, state, rec, None, s, log=None)
    return server, client, state


def live_requests(world, n: int, start: int = 0, age_s: float = 0.2) -> list[dict]:
    now = time.time() - age_s
    reqs = []
    for k in range(start, start + n):
        r = a_request(world, k % 5, seed=k)
        r["request_id"] = f"live_{k}"
        r["ts"] = pd.Timestamp(now, unit="s").strftime("%Y-%m-%d %H:%M:%S.%f")
        reqs.append(r)
    return reqs


def test_server_answers_valid_and_once(world):
    server, client, state = make_server(world)
    client.requests = live_requests(world, 25)
    out = server.answer(server.pull())
    assert len(out) == 25
    assert server.pull() == []                      # no se responde dos veces
    for ans, deadline, _ in out:
        assert is_valid(ans["items"], state) and deadline > time.time()
    server.pusher._push(out)
    assert len(client.inserted) == 25


def test_server_drops_stale_requests(world):
    server, client, _ = make_server(world)
    client.requests = live_requests(world, 5, age_s=30)
    assert server.pull() == []
    assert server.stats.c["dropped"] == 5


def test_server_falls_back_on_errors_and_slowness(world):
    class Broken(Scorer):
        version = "broken"
        def predict(self, X):
            raise ValueError("boom")

    class Slow(Scorer):
        version = "slow"
        def predict(self, X):
            time.sleep(0.25)
            return np.zeros(len(X))

    server, client, state = make_server(world, Broken())
    client.requests = live_requests(world, 5)
    out = server.answer(server.pull())
    assert len(out) == 5 and server.stats.c["fb_error"] == 5
    assert all(is_valid(a["items"], state) for a, _, _ in out)

    server, client, state = make_server(world, Slow(), batch_budget_s=0.3)
    client.requests = live_requests(world, 6)
    t = time.time()
    out = server.answer(server.pull())
    assert time.time() - t < 1.0                    # el presupuesto corta el lote
    assert server.stats.c["fb_budget"] >= 3 and all(is_valid(a["items"], state) for a, _, _ in out)


def test_server_never_recommends_out_of_stock(world):
    server, client, state = make_server(world)
    top = state.fallback_items(30)
    base = world["catalog"].set_index("product_id")
    client.updates = [{**base.loc[p].to_dict(), "product_id": p, "change": "out_of_stock",
                       "is_active": False, "ts_ms": str(10 ** 12 + k)} for k, p in enumerate(top)]
    state.refresh_catalog()
    client.requests = live_requests(world, 30)
    out = server.answer(server.pull())
    shown = {p for a, _, _ in out for p in a["items"]}
    assert not (shown & set(top))


def test_push_retries_then_succeeds(world):
    server, client, _ = make_server(world)
    client.requests = live_requests(world, 3)
    out = server.answer(server.pull())
    client.fail_inserts = 2
    server.pusher._push(out)
    assert len(client.inserted) == 3


def test_live_purchases_are_excluded(world):
    server, client, state = make_server(world)
    req = live_requests(world, 1)[0]
    first = server.rec.recommend(req).items
    state._user_bought[req["user_id"]] = {p: time.time() for p in first}
    second = server.rec.recommend(req).items
    assert not set(first) & set(second)


def test_latency_per_request(world):
    """5.000 productos como en el reto: el cálculo por petición debe ser de milisegundos."""
    catalog = make_catalog(5000, seed=3)
    imps, covis, car, daily, sales = make_history(catalog, n_imps=500)
    ids = catalog["product_id"].tolist()
    index = {p: i for i, p in enumerate(ids)}
    content, emb = fit_content(catalog)
    nb_idx, nb_sim = knn(emb, 30)
    art = Artifacts(ids, 0, {k: Neighbors.from_frame(covis, index, len(ids), 40) for k in ("view", "buy", "b2b")},
                    Neighbors.from_dense(nb_idx, nb_sim), CarouselStats.from_frame(car, index, len(ids)),
                    content, emb, np.zeros(len(ids), np.float32), sales.index.tolist())
    st = State(ProductTable.from_frame(catalog, emb=emb, order=ids), art, DailyPop.from_frames(daily, ids).snapshot)
    import train
    rows = impressions_to_rows(offline_state(st.products, art, DailyPop.from_frames(daily, ids)), imps, 1.0)
    model, _ = train.fit("sklearn", rows["X"], rows["y"], rows["w"], rounds=200)
    rec = Recommender(st, train._InMemoryScorer("sklearn", model, "t"), seed=0)
    world5k = {"products": st.products}
    times = []
    for k in range(200):
        req = a_request(world5k, k % 8, seed=k)
        t = time.perf_counter()
        r = rec.recommend(req)
        times.append(time.perf_counter() - t)
        assert is_valid(r.items, st)
    p95 = float(np.percentile(times, 95)) * 1000
    print(f"\nlatencia por petición (5.000 productos, sklearn 200 árboles): "
          f"p50={np.percentile(times, 50) * 1000:.1f} ms p95={p95:.1f} ms")
    assert p95 < 100


def test_train_and_serve_end_to_end(world, tmp_path, monkeypatch):
    """train.py --build completo contra el cliente falso, y el servicio cargando lo que produce."""
    import train
    from reco import config
    catalog = world["catalog"]
    imps, covis, car, daily, sales = make_history(catalog, n_imps=5000, seed=5)
    fake = FakeClient(catalog, sales, history=(imps, covis, car, daily))
    monkeypatch.setattr(train, "from_env", lambda **kw: fake)
    monkeypatch.setattr(train, "OFFLINE", tmp_path / "offline")
    monkeypatch.setattr(config, "LIVE_ARTIFACTS", tmp_path / "live.pkl")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "model")
    train.main(["--build", "--backend", "sklearn", "--version", "v-e2e", "--keep-permille", "1000"])
    assert (tmp_path / "live.pkl").exists() and (tmp_path / "model" / "model.json").exists()

    from reco.live import LiveState
    import serve_v2
    sc = load_scorer(tmp_path / "model")
    assert sc.version == "v-e2e"
    s = Settings(api_url="http://x", api_key="k", team="team_99")
    art = Artifacts.load(tmp_path / "live.pkl")
    state = LiveState(fake, art, s)
    state.warm_up()
    server = serve_v2.Server(fake, state, Recommender(state, sc, seed=1), None, s, log=None)
    fake.requests = live_requests(world, 20)
    out = server.answer(server.pull())
    assert len(out) == 20 and server.stats.c["model"] == 20
    assert all(is_valid(a["items"], state) and a["model_version"] == "v-e2e" for a, _, _ in out)


def test_push_hedges_slow_insert(world):
    """Si la inserción tarda, se envía una copia y vale la primera que conteste."""
    server, client, _ = make_server(world)
    client.requests = live_requests(world, 4)
    out = server.answer(server.pull())
    client.insert_delays = [1.2, 0.0]              # la primera se cuelga, la copia va rápida
    t = time.time()
    server.pusher._push(out)
    assert time.time() - t < 1.0
    assert server.stats.c["hedged"] == 1 and server.stats.c["pushed"] == 4
    time.sleep(1.3)                                # la primera también acaba: copia duplicada idéntica
    ids = [r["request_id"] for r in client.inserted]
    assert set(ids) == {a["request_id"] for a, _, _ in out}


def test_run_loop_survives_slow_pulls(world):
    """Con lecturas solapadas, una lectura colgada no retrasa las peticiones siguientes."""
    server, client, state = make_server(world)
    client.query_delays = [0.9] + [0.0] * 1000     # la primera lectura se cuelga casi 1 s
    stop = time.time() + 3.0

    def feeder():
        k = 0
        while time.time() < stop:
            client.requests = live_requests(world, 3, start=k, age_s=0.05)
            k += 3
            time.sleep(0.33)
    th = __import__("threading").Thread(target=feeder, daemon=True)
    th.start()
    server.run(max_seconds=3.0)
    time.sleep(0.5)
    assert server.stats.c["dropped"] == 0
    assert len(client.inserted) >= 20
    assert all(is_valid(r["items"], state) for r in client.inserted)


def test_auto_retrain_gate_and_deploy(tmp_path):
    import json as _json
    import auto_retrain
    meta = lambda c_ll, c_auc, n=50: {"file": "model.pkl", "metrics": {"holdout": {
        "n_buys": n, "current_version": "v2", "candidate": {"logloss": c_ll, "auc": c_auc},
        "current": {"logloss": 0.010, "auc": 0.90}}}}
    assert auto_retrain.decide(meta(0.009, 0.91))[0]
    assert auto_retrain.decide(meta(0.009, 0.897))[0]            # AUC casi igual y mejor logloss: sí
    assert not auto_retrain.decide(meta(0.011, 0.95))[0]          # peor logloss: no
    assert not auto_retrain.decide(meta(0.009, 0.85))[0]          # AUC claramente peor: no
    assert not auto_retrain.decide(meta(0.009, 0.95, n=5))[0]     # pocas compras: no se decide
    assert not auto_retrain.decide({"metrics": {}})[0]
    cand, live = tmp_path / "cand", tmp_path / "model"
    cand.mkdir()
    (cand / "model.pkl").write_bytes(b"x")
    (cand / "model.json").write_text(_json.dumps({"file": "model.pkl"}))
    auto_retrain.deploy(cand, live)
    assert (live / "model.pkl").read_bytes() == b"x" and (live / "model.json").exists()


def test_train_with_feedback_holdout(world, tmp_path, monkeypatch):
    """Reentrenar con el feedback propio y comparar con el desplegado en los últimos minutos."""
    import json as _json
    import train
    from reco import config
    catalog = world["catalog"]
    imps, covis, car, daily, sales = make_history(catalog, n_imps=3000, seed=8)
    fake = FakeClient(catalog, sales, history=(imps, covis, car, daily))
    for name, val in (("LIVE_ARTIFACTS", tmp_path / "live.pkl"), ("MODEL_DIR", tmp_path / "model"),
                      ("LOGS", tmp_path / "logs")):
        monkeypatch.setattr(config, name, val)
    monkeypatch.setattr(train, "from_env", lambda **kw: fake)
    monkeypatch.setattr(train, "OFFLINE", tmp_path / "offline")
    train.main(["--build", "--backend", "sklearn", "--version", "v2", "--keep-permille", "1000"])

    # nuestro propio stream: 2 h de peticiones registradas + su feedback
    rng = np.random.default_rng(3)
    ids = catalog["product_id"].tolist()
    t0 = (DAY0 + 61) * DAY_MS
    (tmp_path / "logs").mkdir()
    fb = []
    with open(tmp_path / "logs" / "served-x.jsonl", "w") as f:
        for k in range(1500):
            ts = t0 + k * 5000
            views = [ids[int(i)] for i in rng.integers(0, len(ids), int(rng.integers(0, 4)))]
            shown = [ids[int(i)] for i in rng.choice(len(ids), 10, replace=False)]
            bought = shown[int(rng.integers(0, 3))] if rng.random() < 0.15 else ""
            f.write(_json.dumps({"request_id": f"q{k}", "ts": pd.Timestamp(ts, unit="ms").isoformat(),
                                 "user_id": f"u{k % 50}", "context": {"device": "web", "page": "home"},
                                 "recent_views": [{"product_id": v, "ts": ""} for v in views]}) + "\n")
            fb.append({"request_id": f"q{k}", "ts_ms": str(ts + 4000), "shown": shown,
                       "clicked": bought, "purchased": bought})
    fake.feedback = fb
    train.main(["--with-feedback", "--holdout-minutes", "30", "--backend", "sklearn", "--version", "v3",
                "--model-dir", str(tmp_path / "cand")])
    meta = _json.loads((tmp_path / "cand" / "model.json").read_text())
    hold = meta["metrics"]["holdout"]
    assert hold["current_version"] == "v2" and hold["n_buys"] > 0
    assert "logloss" in hold["candidate"] and "logloss" in hold["current"]
    assert _json.loads((tmp_path / "model" / "model.json").read_text())["version"] == "v2"   # no se tocó
