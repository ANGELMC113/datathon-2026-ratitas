# Recomendador v2 · ratitas de biblio

Implementación de la propuesta: **candidatos baratos → LightGBM → rerank con Thompson sampling**, con un servicio en vivo que no se cae y un entrenamiento offline sin fuga de futuro. No toca ningún fichero existente del repo: todo es nuevo.

## Qué hay

```
reco/                 paquete compartido (mismo código offline y en vivo)
  config.py           .env, rutas y parámetros (plazos, cupos, intervalos de refresco)
  client.py           cliente RawTree con sesión por hilo y reintentos (red, 429, 5xx)
  sql.py              TODAS las consultas SQL
  catalog.py          catálogo en memoria (stock, precios, productos nuevos) + embeddings TF-IDF/SVD
  artifacts.py        co-visitación, vecinos de contenido, carrusel corregido por posición, popularidad
  session.py          lo que sabemos del comprador en esta petición
  candidates.py       ~200 candidatos (rol B)
  features.py         32 features por par petición-producto (rol B)
  model.py            LightGBM / heurística, recarga en caliente (rol C)
  rerank.py           filtro de stock + Thompson sampling en huecos 8-10 (rol D)
  recommender.py      une las piezas + funciones de CONTRACTS.md
  live.py             estado en vivo refrescado en segundo plano (rol A)
  offline.py          descargas, filas de entrenamiento y evaluación
  serving.py          piezas del servicio: estadísticas, registro, inserción con respaldo, variantes A/B
serve_v2.py           bucle en vivo
train.py              artefactos + entrenamiento + validación temporal
monitor.py            resumen del feedback y semáforo (para la ronda final a oscuras)
ab_report.py          conversión oficial de cada versión en ventanas consecutivas (API del leaderboard)
auto_retrain.py       reentrenamiento periódico con puerta de calidad
check_sql.py          prueba cada consulta contra RawTree
run_serve.sh          reinicio automático si el proceso muere
tests/                pruebas sin red (mundo sintético + cliente falso)
```

## Cómo se usa (en este orden)

```bash
pip install -r requirements-v2.txt
python -m pytest -q tests/test_reco.py     # 17 pruebas, ~1 min, sin red

python check_sql.py                        # 1. ¿funcionan las consultas contra RawTree?
python check_sql.py --heavy                #    (también co-visitación y carrusel sobre 1 día)
python train.py --build --version v2-lgbm  # 2. descarga, artefactos, entrena, valida y guarda
python serve_v2.py --dry-run --max-seconds 120   # 3. prueba en vivo SIN insertar
./run_serve.sh                             # 4. en marcha (para el baseline/serve.py antes)
python monitor.py --minutes 15 --watch 60  # 5. vigilar
python train.py --with-feedback --version v3-feedback   # 6. reentrenar con lo aprendido del stream
```

### Ajuste de hiperparámetros

```bash
python train.py --tune 12 --version v4-tuned --model-dir artifacts/model_d
```

Prueba 12 combinaciones (la primera es la de siempre) y elige la de menor logloss en validación. La guarda en `artifacts/offline/best_params.json` y los entrenamientos siguientes (también `auto_retrain.py`) la usan. Tabla completa en `artifacts/offline/tuning.csv`. Como siempre, solo se queda si gana en el A/B.

### Leer el A/B con la métrica oficial

```bash
python ab_report.py --length 15 --windows 4
```

Conversión esperada de cada versión en 4 ventanas consecutivas de 15 min y cuántas veces fue la mejor. Una variante gana si va por delante en casi todas las ventanas. En la ronda final el leaderboard está oculto: solo sirve en la práctica.

### Test A/B en vivo

```bash
python train.py --build --version v3-trans --model-dir artifacts/model_b   # candidato, sin tocar producción
./run_serve.sh --ab-model-dir artifacts/model_b --ab-fraction 0.5
```

La mitad de las peticiones (por hash del `request_id`, siempre la misma mitad) va al modelo B con su propio `model_version`. El leaderboard (vista Beta, desplegando el equipo) muestra la conversión de cada versión en la misma ventana. Para promover al ganador: `cp artifacts/model_b/* artifacts/model/` (recarga en caliente).

### Reentrenamiento automático

```bash
python auto_retrain.py --every-hours 3 --freeze-at "2026-11-15 08:30"
```

Cada 3 h reentrena con el feedback propio en un proceso de baja prioridad. Reserva la última hora de feedback, compara en ella el modelo nuevo con el desplegado y **solo despliega si el nuevo es mejor** (logloss menor y AUC no peor). Desde `--freeze-at` ya no toca nada, para que la versión final esté estable antes de las 10:00. Historial de decisiones: `logs/retrain_history.csv`. `--dry-run` entrena y compara sin desplegar.

`train.py` escribe `artifacts/live.pkl` y `artifacts/model/`. El servicio recarga el modelo solo en cuanto cambia `artifacts/model/model.json`, sin parar. Cada modelo queda copiado en `artifacts/models/<version>/`: para volver atrás, copia esa carpeta sobre `artifacts/model/`.

Si `artifacts/live.pkl` no existe, el servicio arranca en **modo degradado** (popularidad y tendencias, sin co-visitación): responde igualmente.

## Decisiones y por qué

**Fiabilidad (10 puntos)**
- Nunca respondemos algo inválido: se valida cada lista (10 distintos, activos, con stock). Si falla un producto, se *repara* con nuestra lista de emergencia (best-sellers recientes con stock) en vez de tirar la lista entera.
- Si el modelo lanza una excepción o el lote pasa de 0,6 s, esas peticiones van con la lista de emergencia: nunca dejamos que caiga el fallback de la organización.
- La inserción va en otro hilo. Si falla, se reintenta solo con las respuestas que aún llegan a tiempo; las peticiones no se pierden por marcarlas como vistas antes de insertar (fallo del starter).
- Lectura de peticiones filtrada por tiempo (últimos 20 s) en lugar de ordenar la tabla entera; tras 3 fallos seguidos pasa 60 s a la consulta del starter.
- El stock se mira cada 2 s y cada 0,5 s alrededor de los múltiplos de 5 minutos, que es cuando llegan los cambios.
- Reloj: se mide la diferencia con el servidor (`now64()`) para calcular la edad real de cada petición.

**Conversión (50 puntos)**
- Candidatos: vistas recientes, co-visitación vista→vista, vista→compra y compra→compra (en SQL dentro de RawTree), vecinos de contenido, productos nuevos parecidos, más vendidos de las categorías de la sesión, tendencias de la última hora y populares.
- Features relativas (cuotas por 10.000, ratios, similitudes): misma escala con 40 o 60 días de histórico y en vivo. Ningún ID es feature.
- El rendimiento histórico de cada producto en el carrusel se corrige por posición (compras reales / compras esperadas por los huecos donde salió).
- Ya no se ofrece lo que el usuario acaba de comprar (el reto dice que no lo quiere otra vez).

**Sin fuga de futuro**
- Filas de entrenamiento de los días 41-50 con artefactos de los días 1-40; validación en 51-60 con artefactos de 1-50. La popularidad de cada fila es la de los días anteriores a su impresión.
- Se guardan todas las impresiones con compra y una muestra del resto; el peso 1000/‰ compensa el muestreo, así la probabilidad sigue calibrada.

**Aprender del stream (jurado)**
- Al momento: popularidad 24 h / 7 días y tendencias de la última hora se recalculan en vivo.
- Con cada respuesta: el bandit de los huecos 8-10 aprende del feedback (con olvido del 10 % por hora).
- Cada pocas horas: `train.py --with-feedback` añade nuestras propias impresiones (logs + `team_NN.feedback`).

## Lo que NO he podido probar (hazlo tú antes de desplegar)

1. **Las consultas SQL contra RawTree real**: mi entorno no tiene red. Por eso existe `check_sql.py`; si alguna falla, el error dice cuál y suele ser un cast.
2. **LightGBM**: no estaba instalado en mi entorno; el mismo flujo está probado con `HistGradientBoosting` de scikit-learn (se usa solo si falta lightgbm). La parte LightGBM sigue la API estándar de lightgbm 4.
3. **Tiempos reales**: en sintético, 3-5 ms por petición con 5.000 productos. Con RawTree lo que pesa es la red: mira la edad al insertar que imprime el servicio cada 30 s y la latencia de `monitor.py`.

## Cómo leer las métricas de `train.py`

- `auc`, `logloss`: calidad de la probabilidad en lo mostrado.
- `mrr_in_shown_model` frente a `mrr_in_shown_old_order`: si ordenamos lo mostrado mejor que el sistema antiguo.
- Tabla de políticas: ¿está lo comprado en nuestro top 10? **Está sesgada hacia los best-sellers**, porque el sistema antiguo casi solo mostraba eso; úsala para comparar versiones nuestras entre sí, no contra el baseline. La comparación con el baseline la da el leaderboard.
- `recall@N`: qué parte de las compras entra en los N candidatos (para elegir N).

## Relación con CONTRACTS.md

`reco/recommender.py` expone `get_candidates`, `build_features`, `score` y `rerank`. Única diferencia: reciben la petición completa como `context` (resuelve el punto abierto sobre la firma de `recommend()`), porque las vistas recientes hacen falta en todas las etapas.
