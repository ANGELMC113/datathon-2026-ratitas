import os
import time
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter

# 1. Carrega de credencials des de .env
for env_path in [Path(".env"), Path("../.env")]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        break

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM_ID = os.environ.get("TEAM_ID", "team_30")
API_KEY = os.environ.get("RAWTREE_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta RAWTREE_API_KEY al fitxer .env")

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"


def sql(q: str, timeout: int = 60) -> pd.DataFrame:
    r = http.post(f"{API}/v1/query", json={"sql": q}, timeout=timeout)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data", []))


class FastNumpyTwoTower:
    def __init__(self, text_dim: int = 32):
        self.text_dim = text_dim
        self.lock = threading.Lock()
        self.V = None
        self.item_ids = []
        self.id_to_idx = {}
        self.active_mask = None
        self.popularity_prior = None
        self.user_bought = {}

    def refresh_from_db(self):
        """Actualitza catàleg, preus, estoc, tendències i feedback sense aturar el servei."""
        # A. Catàleg vigent + tendències de l'última hora
        cat_df = sql("""
        WITH latest_updates AS (
            SELECT
                product_id::String AS product_id,
                argMax(category::String, ts::DateTime64(3)) AS category,
                argMax(subcategory::String, ts::DateTime64(3)) AS subcategory,
                argMax(brand::String, ts::DateTime64(3)) AS brand,
                argMax(price_eur::Float64, ts::DateTime64(3)) AS price_eur,
                argMax(description::String, ts::DateTime64(3)) AS description,
                argMax(is_active::Bool, ts::DateTime64(3)) AS is_active,
                argMax(change::String, ts::DateTime64(3)) AS last_change
            FROM shop.catalog_updates
            GROUP BY product_id
        ),
        merged AS (
            SELECT
                COALESCE(u.product_id, c.product_id::String) AS product_id,
                COALESCE(u.category, c.category::String) AS category,
                COALESCE(u.subcategory, c.subcategory::String) AS subcategory,
                COALESCE(u.brand, c.brand::String) AS brand,
                COALESCE(u.price_eur, c.price_eur::Float64) AS price_eur,
                COALESCE(u.description, c.description::String) AS description,
                COALESCE(u.is_active, c.is_active::Bool) AS is_active,
                COALESCE(u.last_change, 'initial') AS last_change
            FROM shop.catalog c
            FULL OUTER JOIN latest_updates u ON c.product_id::String = u.product_id
        ),
        trends_1h AS (
            SELECT
                product_id::String AS product_id,
                countIf(event_type::String = 'view') AS views_1h,
                countIf(event_type::String = 'add_to_cart') AS carts_1h,
                countIf(event_type::String = 'purchase') AS purchases_1h
            FROM shop.events
            WHERE ts::DateTime64(3) >= now() - INTERVAL 1 HOUR
            GROUP BY product_id
        ),
        hist_sales AS (
            SELECT
                product_id::String AS product_id,
                countIf(event_type::String = 'purchase') AS hist_buys
            FROM shop.history_events
            GROUP BY product_id
        )
        SELECT
            m.product_id AS product_id,
            m.category AS category,
            m.subcategory AS subcategory,
            m.brand AS brand,
            m.price_eur AS price_eur,
            m.description AS description,
            m.is_active AS is_active,
            m.last_change AS last_change,
            COALESCE(t.views_1h, 0) AS views_1h,
            COALESCE(t.carts_1h, 0) AS carts_1h,
            COALESCE(t.purchases_1h, 0) AS purchases_1h,
            COALESCE(h.hist_buys, 0) AS hist_buys
        FROM merged m
        LEFT JOIN trends_1h t ON m.product_id = t.product_id
        LEFT JOIN hist_sales h ON m.product_id = h.product_id
        """)

        cat_df["description"] = cat_df["description"].fillna("")
        cat_df["log_price"] = np.log1p(cat_df["price_eur"].fillna(20.0).clip(lower=0))
        cat_df["trend_score"] = (
            np.log1p(cat_df["purchases_1h"]) * 2.0
            + np.log1p(cat_df["carts_1h"])
        )

        # Codificació de la torre d'ítem
        ohe = OneHotEncoder(handle_unknown="ignore").fit_transform(
            cat_df[["category", "subcategory", "brand"]].fillna("unknown")
        ).toarray()
        tfidf = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(cat_df["description"])
        n_comp = min(self.text_dim, tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=max(1, n_comp), random_state=42).fit_transform(tfidf)
        nums = StandardScaler().fit_transform(cat_df[["log_price", "trend_score"]])

        raw_V = np.hstack([ohe * 1.2, svd * 1.0, nums * 0.5]).astype(np.float32)
        norms = np.linalg.norm(raw_V, axis=1, keepdims=True) + 1e-8
        V = raw_V / norms

        item_ids = cat_df["product_id"].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(item_ids)}
        active_mask = cat_df["is_active"].values.astype(bool)

        # B. Prior de popularitat + tendència + feedback propi
        pop = (
            np.log1p(cat_df["hist_buys"].values) * 0.4
            + np.log1p(cat_df["purchases_1h"].values) * 1.5
            + np.log1p(cat_df["carts_1h"].values) * 0.6
            + (cat_df["last_change"].values == "new_product").astype(float) * 0.3
        ).astype(np.float32)
        pop = pop / (np.max(pop) + 1e-8)

        # Incorporació del feedback real de team_30.feedback
        fb_df = sql(f"""
            SELECT shown, purchased
            FROM {TEAM_ID}.feedback
            WHERE request_id != '__init__' AND served_by = 'team'
            ORDER BY ts::DateTime64(3) DESC
            LIMIT 10000
        """)
        if not fb_df.empty:
            imp_counts = Counter()
            buy_counts = Counter()
            for r in fb_df.itertuples():
                if isinstance(r.shown, list):
                    imp_counts.update(r.shown)
                if isinstance(r.purchased, str) and r.purchased:
                    buy_counts[r.purchased] += 1

            total_imps = sum(imp_counts.values()) + 1
            for pid, imps in imp_counts.items():
                idx = id_to_idx.get(pid)
                if idx is not None:
                    buys = buy_counts.get(pid, 0)
                    emp_cr = buys / (imps + 5.0)
                    ucb = 0.05 * np.sqrt(np.log(total_imps) / (imps + 1.0))
                    pop[idx] += float(np.clip(emp_cr * 2.0 + ucb, 0.0, 0.4))

        # C. Compres per usuari (per evitar recomanar un producte que ja han comprat)
        bought_df = sql("""
            SELECT user_id::String AS user_id, groupArray(DISTINCT product_id::String) AS items
            FROM (
                SELECT user_id, product_id FROM shop.history_events WHERE event_type = 'purchase'
                UNION ALL
                SELECT user_id, product_id FROM shop.events WHERE event_type = 'purchase'
            )
            GROUP BY user_id
        """)
        bought_dict = {r.user_id: set(r.items) for r in bought_df.itertuples()}

        with self.lock:
            self.V = V
            self.item_ids = item_ids
            self.id_to_idx = id_to_idx
            self.active_mask = active_mask
            self.popularity_prior = pop
            self.user_bought = bought_dict

    def recommend(self, row) -> list[str]:
        """Retorna exactament 10 product_ids actius i diferents."""
        with self.lock:
            V = self.V
            item_ids = self.item_ids
            id_to_idx = self.id_to_idx
            active_mask = self.active_mask
            pop = self.popularity_prior
            already_bought = self.user_bought.get(str(row.user_id), set())

        rviews = row.recent_views if isinstance(row.recent_views, list) else []
        req_ts = pd.Timestamp(row.ts)

        if rviews:
            u_vec = np.zeros(V.shape[1], dtype=np.float32)
            for v in rviews[-10:]:
                if not isinstance(v, dict):
                    continue
                idx = id_to_idx.get(v.get("product_id"))
                if idx is not None:
                    dt = max((req_ts - pd.Timestamp(v.get("ts", row.ts))).total_seconds(), 0.0)
                    weight = np.exp(-dt / 1800.0)  # decaïment de 30 minuts
                    u_vec += weight * V[idx]
            norm = np.linalg.norm(u_vec)
            if norm > 1e-8:
                u_vec /= norm
                scores = (V @ u_vec) + 0.30 * pop
            else:
                scores = pop.copy()
        else:
            scores = pop.copy()

        # Només productes actius
        scores = scores.copy()
        scores[~active_mask] = -1e9

        # Penalitzem productes ja comprats pel mateix usuari
        for pid in already_bought:
            idx = id_to_idx.get(pid)
            if idx is not None:
                scores[idx] -= 2.0

        # Lleugera penalització a l'últim producte vist per fomentar productes complementaris
        if rviews and isinstance(rviews[-1], dict):
            last_idx = id_to_idx.get(rviews[-1].get("product_id"))
            if last_idx is not None:
                scores[last_idx] -= 0.4

        top10 = np.argpartition(-scores, 10)[:10]
        top10 = top10[np.argsort(-scores[top10])]
        return [item_ids[i] for i in top10]


def main():
    print(f"Inicialitzant model per a {TEAM_ID}...")
    model = FastNumpyTwoTower(text_dim=32)
    model.refresh_from_db()
    print(f"Catàleg carregat: {len(model.item_ids)} productes ({model.active_mask.sum()} actius).")

    def background_refresh():
        while True:
            time.sleep(60)
            try:
                model.refresh_from_db()
                print(f"[Refresh OK] Catàleg i feedback actualitzats ({model.active_mask.sum()} actius).")
            except Exception as e:
                print(f"[Refresh Error] {e}")

    threading.Thread(target=background_refresh, daemon=True).start()

    seen = set()
    last_ts = None

    print("Escoltant shop.requests en temps real...")
    while True:
        t0 = time.time()
        try:
            if last_ts is None:
                where_clause = "ts::DateTime64(3) >= now() - INTERVAL 3 SECOND"
            else:
                where_clause = f"ts::DateTime64(3) >= '{last_ts}'"

            reqs = sql(f"""
                SELECT request_id, ts, user_id, is_new_user, recent_views
                FROM shop.requests
                WHERE {where_clause}
                ORDER BY ts::DateTime64(3) ASC
                LIMIT 100
            """, timeout=5)

            if not reqs.empty:
                reqs = reqs[~reqs["request_id"].isin(seen)]
                if not reqs.empty:
                    batch = []
                    for row in reqs.itertuples():
                        items = model.recommend(row)
                        batch.append({
                            "request_id": row.request_id,
                            "items": items,
                            "model_version": "v2-numpy-two-tower"
                        })

                    res = http.post(
                        f"{API}/v1/tables/submissions",
                        params={"database": TEAM_ID},
                        json=batch,
                        timeout=2.0
                    )
                    res.raise_for_status()

                    seen.update(reqs["request_id"].tolist())
                    if len(seen) > 10000:
                        seen = set(list(seen)[-5000:])
                    last_ts = str(reqs["ts"].max())[:23]
                    print(f"Enviades {len(batch)} recomanacions (últim ts: {last_ts})")

        except Exception as e:
            print(f"[Loop Error] {e}")

        elapsed = time.time() - t0
        time.sleep(max(0.05, 0.3 - elapsed))


if __name__ == "__main__":
    main()