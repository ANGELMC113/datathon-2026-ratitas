# tests/ · Rol A

Pruebas de que el servicio sigue respondiendo bien pase lo que pase (pasos 5 y 6 de la guía):

- [ ] La validación devuelve siempre 10 IDs distintos y activos (duplicados, inactivos, lista vacía).
- [ ] Con un modelo que tarda 5 s, todas las respuestas salen a tiempo con fallback.
- [ ] Con un modelo que lanza una excepción, el bucle sigue vivo.
