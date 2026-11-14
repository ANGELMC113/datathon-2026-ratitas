# offline/ · Rol C (Angel y Martí)

Evaluación offline con un split temporal fijo: entrenar con los días 1–50 de `shop.history_sessions` y validar con los días 51–60.

- Punto de partida: `starter/offline_eval.ipynb` (incluye un estimador IPS con las propensities).
- Una única función `evaluate(recommender)` que usa todo el equipo, para comparar versiones de forma justa.
- Que importe las funciones de `recsys/`, para probar offline el mismo código que corre en vivo.
- Ninguna feature puede usar datos posteriores a la impresión que se evalúa.
