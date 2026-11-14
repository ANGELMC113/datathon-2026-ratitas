# Contratos entre piezas

No se cambian sin avisar al equipo.

| Función | Dueño | Entrada | Salida |
| --- | --- | --- | --- |
| `get_candidates(user_id, context)` | B (Angel y Martí) | usuario y contexto de la petición | `list[product_id]`, unos 200 |
| `build_features(user_id, candidates)` | B (Angel y Martí) | candidatos | `DataFrame`, una fila por par usuario-producto |
| `score(features)` | C (Angel y Martí) | el DataFrame anterior | array de probabilidades de compra, mismo orden |
| `rerank(ranked, user_id)` | D (Inés) | `product_id` ordenados por score | `list` de 10: sin inactivos ni agotados, con exploración |

## Por decidir en equipo

- [ ] Firma de `recommend()`: en `starter/recommender.py` es `recommend(recent_views, best, unavailable)`. ¿Pasa a recibir la petición entera (`user_id`, `is_new_user`, `context`, `recent_views`)?
- [ ] Presupuesto de tiempo de cada función dentro de los 3 s.
- [ ] Columnas de `build_features()` acordadas entre B y C.

## Convenciones

- Tablas en `team_NN.scratch` con prefijo por dueño: `feat_*` (B), `model_*` (C), `emb_*` (D).
- `model_version` = `vN-descripcion` (p. ej. `v2-covis-trend`); cada despliegue, una línea en `CHANGELOG.md`.
