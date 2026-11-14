import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURACIÓ I CONNEXIÓ A RAWTREE
# ==========================================
for env_path in [Path(".env"), Path("../.env")]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        break

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
API_KEY = os.environ.get("RAWTREE_API_KEY")

if not API_KEY:
    print("ALERTA: No s'ha trobat la RAWTREE_API_KEY. El codi fallarà en fer les consultes.")

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"

def sql(query_str: str) -> list[dict]:
    """Executa una consulta SQL a RawTree i retorna una llista de diccionaris."""
    res = http.post(f"{API}/v1/query", json={"sql": query_str}, timeout=30)
    res.raise_for_status()
    return res.json().get("data", [])


# ==========================================
# 1. EXTRACCIÓ I PREPARACIÓ DE DADES
# ==========================================
class DataExtractor:
    def __init__(self):
        self.item_to_cat_idx = {}
        self.user_history_cats = {}
        # Hi ha 10 categories en total segons el README
        self.categories = [] 
        
    def carregar_dades_inicials(self):
        print("1. Descarregant catàleg per mapar productes a categories...")
        catalog = sql("SELECT product_id::String AS product_id, category::String AS category FROM shop.catalog")
        
        # Creem un índex únic per a cada categoria (de 0 a 9)
        cats_uniques = list(set(row["category"] for row in catalog if row.get("category")))
        self.categories = sorted(cats_uniques)
        cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        self.item_to_cat_idx = {
            row["product_id"]: cat_to_idx[row["category"]] 
            for row in catalog if row.get("category")
        }

        print("2. Descarregant historial de compres dels usuaris...")
        historial_cru = sql("""
            SELECT 
                user_id::String AS user_id, 
                groupArray(product_id::String) AS past_purchases 
            FROM shop.history_events 
            WHERE event_type = 'purchase' 
            GROUP BY user_id
        """)
        
        # Precalculem el perfil històric de categories per a cada usuari
        for fila in historial_cru:
            user = fila["user_id"]
            purchases = fila.get("past_purchases", [])
            
            cat_counts = np.zeros(10, dtype=np.float32)
            for pid in purchases:
                if pid in self.item_to_cat_idx:
                    cat_counts[self.item_to_cat_idx[pid]] += 1.0
            
            # Normalitzem perquè sumi 1
            if cat_counts.sum() > 0:
                cat_counts /= cat_counts.sum()
                
            self.user_history_cats[user] = cat_counts
        
        print("Dades carregades amb èxit!")

    def processar_peticio(self, row_request: dict) -> torch.Tensor:
        """Converteix una petició de shop.requests en un vector de 28 dimensions."""
        
        # A. Variables directes (1 dimensió)
        feat_is_new = 1.0 if row_request.get("is_new_user") else 0.0
        
        # B. Context (6 dimensions: 3 device + 3 page)
        ctx = row_request.get("context", {})
        dev = ctx.get("device", "")
        feat_dev = [1.0 if dev == "web" else 0.0, 1.0 if dev == "ios" else 0.0, 1.0 if dev == "android" else 0.0]
        
        pag = ctx.get("page", "")
        feat_pag = [1.0 if pag == "home" else 0.0, 1.0 if pag == "product" else 0.0, 1.0 if pag == "category" else 0.0]
        
        # C. Curt termini: Sessió actual (11 dimensions: 1 count + 10 categories)
        rviews = row_request.get("recent_views", [])
        feat_n_views = min(len(rviews) / 10.0, 1.0)
        
        recent_cat_counts = np.zeros(10, dtype=np.float32)
        for v in rviews:
            pid = v.get("product_id")
            if pid in self.item_to_cat_idx:
                recent_cat_counts[self.item_to_cat_idx[pid]] += 1.0
                
        if recent_cat_counts.sum() > 0:
            recent_cat_counts /= recent_cat_counts.sum()
            
        # D. Llarg termini: Historial de compres pre-calculat (10 dimensions)
        user_id = row_request.get("user_id", "")
        hist_cat_counts = self.user_history_cats.get(user_id, np.zeros(10, dtype=np.float32))
        
        # Ensamblar-ho tot junt: 1 + 3 + 3 + 1 + 10 + 10 = 28 dimensions
        features_list = [feat_is_new] + feat_dev + feat_pag + [feat_n_views] + recent_cat_counts.tolist() + hist_cat_counts.tolist()
        
        return torch.tensor(features_list, dtype=torch.float32)


# ==========================================
# 2. XARXA NEURONAL (USER ENCODER)
# ==========================================
class UserEncoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=256, output_dim=128):
        super().__init__()
        # Rep el vector de 28 dimensions creat pel DataExtractor
        self.capa_oculta = nn.Linear(input_dim, hidden_dim)
        self.activacio = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Evita que la xarxa memoritzi (overfitting)
        self.capa_sortida = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x pot ser un sol usuari [28] o un batch d'usuaris [Batch, 28]
        x = self.capa_oculta(x)
        x = self.activacio(x)
        x = self.dropout(x)
        embedding = self.capa_sortida(x)
        
        # Normalització L2: El vector final de 128 dimensions apuntarà a una esfera de radi 1
        return F.normalize(embedding, p=2, dim=-1)


# ==========================================
# PROVA DE FUNCIONAMENT
# ==========================================
if __name__ == "__main__":
    # 1. Inicialitzem l'extractor i descarreguem la info de RawTree
    extractor = DataExtractor()
    extractor.carregar_dades_inicials()
    
    # 2. Inicialitzem la xarxa neuronal
    model_usuari = UserEncoder(input_dim=28, hidden_dim=256, output_dim=128)
    
    # 3. Simulem l'arribada d'una petició en viu (agafem la més recent de shop.requests)
    print("\nLlegint una petició en viu...")
    peticions = sql("SELECT * FROM shop.requests ORDER BY ts::DateTime64(3) DESC LIMIT 1")
    
    if peticions:
        peticio_prova = peticions[0]
        
        # Passem la petició pel nostre extractor de característiques
        vector_entrada = extractor.processar_peticio(peticio_prova)
        print(f"Vector d'entrada creat. Dimensions: {vector_entrada.shape}")
        
        # Passem el vector per la xarxa per obtenir l'embedding final
        # Afegim .unsqueeze(0) per dir-li a PyTorch que és un 'batch' d'1 sol usuari
        amb_batch = vector_entrada.unsqueeze(0) 
        
        # Mode avaluació per desactivar el dropout en predicció
        model_usuari.eval() 
        with torch.no_grad():
            embedding_final = model_usuari(amb_batch)
            
        print(f"Embedding final creat! Dimensions: {embedding_final.shape}")
        print(f"Primers 5 valors de l'embedding: {embedding_final[0][:5].numpy()}")
    else:
        print("No hi ha peticions a la base de dades actualment.")