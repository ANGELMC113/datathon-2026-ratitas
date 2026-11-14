# scripts/ · Exploración y monitorización

- Resumen de `feedback` (rol A, paso 7 de la guía): parte de las consultas 5 y 6 de `starter/queries.sql` (conversión por hora y `served_by`; motivo de fallback y latencia p50/p95).
- Ignorar la fila `request_id = '__init__'`.
- Semáforo de la ronda final: verde si más del 90 % de las respuestas de los últimos 5 minutos son válidas y a tiempo.
