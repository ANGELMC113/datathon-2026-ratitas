"""Two-tower: afinidad usuario-producto aprendida, como feature del LightGBM (rol C).

Idea
----
Dos redes pequeñas convierten a cada producto y a cada comprador en un vector de 64 números; el
coseno entre ambos mide cuánto "encaja" ese producto con lo que el comprador está mirando. El
LightGBM recibe esa afinidad como una feature más (tt_z, tt_pct) y decide cuánto pesa frente a
popularidad, co-visitación, precio...

Torre de producto:  [ID (con dropout)] + [contenido TF-IDF de catalog.py] + [categoría, subcategoría,
                    marca] + [log precio]  -> MLP -> vector normalizado
Torre de usuario:   media de sus vistas recientes (pesos decay**k, los mismos de session.py)
                    + la última vista + dispositivo + página + nº de vistas -> MLP -> vector normalizado

Decisiones
----------
- El ID del producto se aprende, pero con dropout (30 %) y empezando en cero: un producto que el
  modelo no ha visto (los nuevos de la ronda final) tiene ID = 0 y se apoya en su contenido, que
  existe desde el primer segundo.
- Los productos creados después del corte de entrenamiento NO reciben fila de ID: si no, el modelo
  aprendería "este producto nunca se clica" cuando en realidad aún no existía.
- Sin popularidad dentro del modelo: la popularidad cambia cada hora y ya la tiene el LightGBM.
  Aquí solo afinidad, y además se corrige la exposición (logQ): restamos log(veces mostrado en el
  carrusel antiguo) a cada producto al entrenar, para que "se clicó porque se mostraba mucho" no
  cuente como "le gusta".
- Softmax sobre el catálogo entero (~5.000 productos): cada clic compite con todos los productos,
  igual que al recomendar. Las compras pesan el doble que los clics.
- Entrenamiento con torch (solo offline). En vivo todo es numpy: sin torch en el servicio, y
  ~1 ms por petición (5.000 x 64 multiplicaciones).
- Feature normalizada contra el catálogo entero del mismo usuario (z-score y percentil): así no
  depende de cuántos candidatos haya (10 mostrados al entrenar, 200 en vivo) ni de qué versión
  del modelo esté cargada (la de días < 40 o la de todo el histórico).
"""
from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import ProductTable
from .session import Session, make_session
from .state import State

EMB_DIM = 64
HIDDEN = 128
MAX_HIST = 10          # = Settings.max_recent
ID_DIM, CAT_DIM, SUB_DIM, BRAND_DIM, CTX_DIM = 16, 8, 16, 16, 4
N_CTX = 4              # 3 dispositivos / páginas + desconocido (0)


def _codes(arr: np.ndarray, size: int) -> np.ndarray:
    """Código de vocabulario -> fila de embedding (0 = desconocido o valor nuevo)."""
    c = np.asarray(arr, dtype=np.int64) + 1
    c[(c < 1) | (c >= size)] = 0
    return c


def _ctx_code(v: float) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    c = int(v) + 1
    return c if 0 < c < N_CTX else 0


def _content(products: ProductTable, d: int) -> np.ndarray:
    e = products.emb
    if e.ndim != 2 or e.shape[1] == d:
        return np.asarray(e, dtype=np.float32) if e.ndim == 2 else np.zeros((len(products), d), np.float32)
    out = np.zeros((len(products), d), dtype=np.float32)
    k = min(d, e.shape[1])
    out[:, :k] = e[:, :k]
    return out


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------------------------
# Inferencia (numpy): lo que usa features.py offline y en vivo
# ----------------------------------------------------------------------------------------------
class TwoTower:
    def __init__(self, params: dict[str, np.ndarray], meta: dict) -> None:
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        self.meta = meta
        self.version = str(meta.get("version", "tt"))
        self.d_content = int(meta["d_content"])
        self.n_cat = self.p["e_cat.weight"].shape[0]
        self.n_sub = self.p["e_sub.weight"].shape[0]
        self.n_brand = self.p["e_brand.weight"].shape[0]
        self.id_row = {pid: k + 1 for k, pid in enumerate(meta["id_list"])}
        self._cache: tuple[ProductTable, np.ndarray] | None = None

    # -- persistencia ---------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez(buf, __meta__=np.array(json.dumps(self.meta)), **self.p)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)                  # atómico: el servicio nunca lee un fichero a medias

    @classmethod
    def load(cls, path: Path) -> "TwoTower | None":
        if not Path(path).exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as f:
                meta = json.loads(str(f["__meta__"]))
                params = {k: f[k] for k in f.files if k != "__meta__"}
            return cls(params, meta)
        except Exception as exc:           # fichero roto: sin two-tower, las features quedan NaN
            print(f"[tt] no se pudo cargar {path}: {exc!r}")
            return None

    # -- torres ---------------------------------------------------------------------------------
    def item_matrix(self, products: ProductTable) -> np.ndarray:
        """Vector de todos los productos (n, 64). Se calcula una vez por versión del catálogo."""
        cache = self._cache
        if cache is not None and cache[0] is products:
            return cache[1]
        p = self.p
        rows = np.array([self.id_row.get(pid, 0) for pid in products.ids], dtype=np.int64)
        rows[rows >= p["e_id.weight"].shape[0]] = 0
        x = np.concatenate([
            p["e_id.weight"][rows],
            _content(products, self.d_content),
            p["e_cat.weight"][_codes(products.cat, self.n_cat)],
            p["e_sub.weight"][_codes(products.sub, self.n_sub)],
            p["e_brand.weight"][_codes(products.brand, self.n_brand)],
            np.log1p(np.maximum(products.price, 0)).astype(np.float32)[:, None],
        ], axis=1)
        h = np.maximum(x @ p["p1.weight"].T + p["p1.bias"], 0)
        items = _normalize(h @ p["p2.weight"].T + p["p2.bias"]).astype(np.float32)
        self._cache = (products, items)    # asignación atómica: seguro entre hilos
        return items

    def user_vector(self, sess: Session, items: np.ndarray) -> np.ndarray:
        p = self.p
        rec = np.asarray(sess.recent[:MAX_HIST], dtype=np.int64)
        rec_w = np.asarray(sess.recent_w[:MAX_HIST], dtype=np.float32)
        ok = rec < len(items)
        rec, rec_w = rec[ok], rec_w[ok]
        if len(rec):
            e = items[rec]
            mean = (e * rec_w[:, None]).sum(0) / max(float(rec_w.sum()), 1e-6)
            last = e[0]
        else:
            mean = last = np.zeros(items.shape[1], dtype=np.float32)
        x = np.concatenate([mean, last, p["e_dev.weight"][_ctx_code(sess.device)],
                            p["e_page.weight"][_ctx_code(sess.page)],
                            np.array([math.log1p(sess.n_recent)], dtype=np.float32)])
        h = np.maximum(x @ p["u1.weight"].T + p["u1.bias"], 0)
        return _normalize(h @ p["u2.weight"].T + p["u2.bias"]).astype(np.float32)

    def features(self, products: ProductTable, sess: Session, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(tt_z, tt_pct) para cada candidato: afinidad frente al catálogo entero de este usuario."""
        items = self.item_matrix(products)
        s_all = items @ self.user_vector(sess, items)
        cand = np.asarray(cand, dtype=np.int64)
        s = np.full(len(cand), np.nan, dtype=np.float32)
        ok = cand < len(s_all)
        s[ok] = s_all[cand[ok]]
        z = (s - s_all.mean()) / max(float(s_all.std()), 1e-6)
        pct = np.searchsorted(np.sort(s_all), s, side="right") / len(s_all)
        pct[~ok] = np.nan
        return z.astype(np.float32), pct.astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Entrenamiento (torch, solo offline)
# ----------------------------------------------------------------------------------------------
def _rows(state: State, imps: pd.DataFrame, max_recent: int = MAX_HIST, decay: float = 0.7) -> dict:
    """Una fila por impresión con clic. El historial sale de make_session: el mismo que en vivo."""
    products = state.products
    H, W, dev, page, nrec, pos1, pos2, ts, buy = [], [], [], [], [], [], [], [], []
    for r in imps.itertuples(index=False):
        req = r._asdict()
        c = products.index.get(str(req.get("clicked") or ""))
        b = products.index.get(str(req.get("purchased") or ""))
        if c is None and b is None:
            continue
        sess = make_session(state, req, max_recent, decay)
        h = np.full(MAX_HIST, -1, dtype=np.int64)
        w = np.zeros(MAX_HIST, dtype=np.float32)
        m = min(len(sess.recent), MAX_HIST)
        h[:m], w[:m] = sess.recent[:m], sess.recent_w[:m]
        H.append(h); W.append(w)
        dev.append(_ctx_code(sess.device)); page.append(_ctx_code(sess.page))
        nrec.append(math.log1p(sess.n_recent))
        p1 = c if c is not None else b
        pos1.append(p1); pos2.append(b if (b is not None and b != p1) else -1)
        ts.append(sess.ts)
        buy.append(b is not None)
    return {"H": np.array(H, dtype=np.int64).reshape(-1, MAX_HIST),
            "W": np.array(W, dtype=np.float32).reshape(-1, MAX_HIST),
            "dev": np.array(dev, dtype=np.int64), "page": np.array(page, dtype=np.int64),
            "nrec": np.array(nrec, dtype=np.float32), "pos1": np.array(pos1, dtype=np.int64),
            "pos2": np.array(pos2, dtype=np.int64), "ts": np.array(ts, dtype=np.float64),
            "buy": np.array(buy, dtype=bool)}


def train_two_tower(state: State, imps: pd.DataFrame, until_ms: int, *, epochs: int = 15,
                    val_days: float = 3.0, purchase_weight: float = 2.0, id_dropout: float = 0.3,
                    lr: float = 2e-3, batch: int = 512, seed: int = 0, log=print) -> TwoTower:
    """Entrena con las impresiones con clic ANTERIORES a until_ms y devuelve el modelo en numpy.

    Validación: los últimos `val_days` días de esa misma ventana (parada temprana por pérdida)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    products = state.products
    P = len(products)
    until_s = until_ms / 1000.0

    imps = imps[pd.to_numeric(imps["ts_ms"], errors="coerce") < until_ms]
    t = time.time()
    data = _rows(state, imps)
    n = len(data["pos1"])
    if n < 100:
        raise ValueError(f"demasiado pocas impresiones con clic para entrenar el two-tower: {n}")
    val = data["ts"] >= until_s - val_days * 86400
    if val.sum() < 50 or (~val).sum() < 50:
        val = np.arange(n) >= int(0.9 * n)
    log(f"[tt] {n} impresiones con clic ({int(data['buy'].sum())} con compra), "
        f"{int(val.sum())} de validación; preparadas en {time.time() - t:.0f} s")

    # -- entradas fijas de la torre de producto --------------------------------------------------
    created_ok = (products.created <= until_s) | (products.created <= 0)   # existían antes del corte
    id_list = [pid for pid, ok in zip(products.ids, created_ok) if ok]
    row_of = {pid: k + 1 for k, pid in enumerate(id_list)}
    d_content = products.emb.shape[1] if products.emb.ndim == 2 else 1
    n_cat = int(products.cat.max()) + 2
    n_sub = int(products.sub.max()) + 2
    n_brand = int(products.brand.max()) + 2
    T = {
        "id": torch.tensor([row_of.get(pid, 0) for pid in products.ids], dtype=torch.long),
        "content": torch.tensor(_content(products, d_content)),
        "cat": torch.tensor(_codes(products.cat, n_cat)),
        "sub": torch.tensor(_codes(products.sub, n_sub)),
        "brand": torch.tensor(_codes(products.brand, n_brand)),
        "price": torch.tensor(np.log1p(np.maximum(products.price, 0)).astype(np.float32)),
    }
    # corrección de exposición: log(1 + veces mostrado en el carrusel antiguo antes del corte)
    shown = np.zeros(P, dtype=np.float32)
    car = state.artifacts.carousel.shown_log
    m = min(P, len(car))
    shown[:m] = np.nan_to_num(car[:m])
    logq = torch.tensor(shown)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e_id = nn.Embedding(len(id_list) + 1, ID_DIM)
            nn.init.zeros_(self.e_id.weight)
            self.e_cat = nn.Embedding(n_cat, CAT_DIM)
            self.e_sub = nn.Embedding(n_sub, SUB_DIM)
            self.e_brand = nn.Embedding(n_brand, BRAND_DIM)
            self.p1 = nn.Linear(ID_DIM + d_content + CAT_DIM + SUB_DIM + BRAND_DIM + 1, HIDDEN)
            self.p2 = nn.Linear(HIDDEN, EMB_DIM)
            self.e_dev = nn.Embedding(N_CTX, CTX_DIM)
            self.e_page = nn.Embedding(N_CTX, CTX_DIM)
            self.u1 = nn.Linear(2 * EMB_DIM + 2 * CTX_DIM + 1, HIDDEN)
            self.u2 = nn.Linear(HIDDEN, EMB_DIM)
            self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        def items(self) -> "torch.Tensor":
            ids = T["id"]
            if self.training and id_dropout > 0:
                ids = ids * (torch.rand(ids.shape) > id_dropout)
            x = torch.cat([self.e_id(ids), T["content"], self.e_cat(T["cat"]), self.e_sub(T["sub"]),
                           self.e_brand(T["brand"]), T["price"][:, None]], 1)
            return F.normalize(self.p2(F.relu(self.p1(x))), dim=-1)

        def user(self, items, H, W, dev, page, nrec) -> "torch.Tensor":
            mask = (H >= 0).float()
            e = items[H.clamp(min=0)] * mask[..., None]
            w = W * mask
            mean = (e * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-6)
            x = torch.cat([mean, e[:, 0], self.e_dev(dev), self.e_page(page), nrec[:, None]], 1)
            return F.normalize(self.u2(F.relu(self.u1(x))), dim=-1)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    D = {k: torch.tensor(data[k]) for k in ("H", "W", "dev", "page", "nrec", "pos1", "pos2")}
    wrow = torch.tensor(np.where(data["buy"], purchase_weight, 1.0).astype(np.float32))
    tr_idx = torch.tensor(np.flatnonzero(~val))
    va_idx = torch.tensor(np.flatnonzero(val))

    def batch_loss(items, idx, correct=True):
        u = net.user(items, D["H"][idx], D["W"][idx], D["dev"][idx], D["page"][idx], D["nrec"][idx])
        logits = net.log_scale.exp().clamp(max=50) * (u @ items.T)
        if correct:
            logits = logits - logq
        p1, p2 = D["pos1"][idx], D["pos2"][idx]
        lp = logits.gather(1, p1[:, None])[:, 0]
        lp2 = logits.gather(1, p2.clamp(min=0)[:, None])[:, 0]
        lp = torch.where(p2 >= 0, torch.logaddexp(lp, lp2), lp)
        per = torch.logsumexp(logits, 1) - lp
        return (per * wrow[idx]).sum() / wrow[idx].sum(), logits

    best, best_state, stall = math.inf, None, 0
    for epoch in range(epochs):
        net.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot, t = 0.0, time.time()
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss, _ = batch_loss(net.items(), idx)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            items = net.items()
            vl, recall = 0.0, 0.0
            for i in range(0, len(va_idx), 2048):
                idx = va_idx[i:i + 2048]
                l, logits = batch_loss(items, idx)
                vl += l.item() * len(idx)
                raw = logits + logq                                   # sin corrección, como en vivo
                top = raw.topk(50, 1).indices
                recall += (top == D["pos1"][idx][:, None]).any(1).float().sum().item()
            vl /= max(len(va_idx), 1)
            recall /= max(len(va_idx), 1)
        log(f"[tt] época {epoch + 1:2d}  train {tot / max(len(perm), 1):.4f}  val {vl:.4f}  "
            f"recall@50 {recall:.3f}  ({time.time() - t:.0f} s)")
        if vl < best - 1e-4:
            best, stall = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_recall, best_epoch = recall, epoch + 1
        else:
            stall += 1
            if stall >= 2:
                break
    net.load_state_dict(best_state)
    params = {k: v.cpu().numpy() for k, v in net.state_dict().items() if k != "log_scale"}
    meta = {"version": f"tt-{time.strftime('%d%H%M')}", "until_ms": int(until_ms), "d_content": int(d_content),
            "id_list": id_list, "epochs": best_epoch, "val_loss": best, "val_recall50": best_recall,
            "n_train": int((~val).sum()), "n_val": int(val.sum()),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    log(f"[tt] listo: época {best_epoch}, val loss {best:.4f}, recall@50 {best_recall:.3f}")
    return TwoTower(params, meta)
