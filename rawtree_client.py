"""Cliente RawTree compartido. Dueño: rol A (Shiqi). Lo usa todo el equipo.

Interfaz acordada (no cambiar sin avisar):
    query(sql: str) -> list[dict]          # filas de un SELECT
    insert(table: str, rows: list[dict])   # inserta en la base del equipo (TEAM_ID)

Punto de partida: query() y push() de starter/recommender.py.

TODO (paso 2 de la guía):
- Leer TEAM_ID, RAWTREE_API_KEY y RAWTREE_API_URL de .env / entorno (nunca la key en el código).
- Timeout HTTP corto (el starter usa 10 s: ¿cuánto te puedes permitir dentro de 3 s?).
- Reintentar solo los errores que tiene sentido repetir (red, 5xx, 429), con espera creciente.
- Mensajes de error con el código HTTP y el principio de la respuesta.
"""
