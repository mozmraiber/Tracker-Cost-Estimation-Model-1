EXPORT DATA OPTIONS(
  uri='gs://httparchive-export/exp8/exp_*.parquet',
  format='PARQUET',
  overwrite=true
) AS
SELECT
  *
FROM (

SELECT
  req.url,
  req.type AS resource_type,
  CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64) AS transfer_bytes,
  CAST(JSON_VALUE(req.payload, '$._objectSize') AS INT64) AS compressed_bytes,
  JSON_VALUE(req.payload, '$._contentType') AS content_type,
  JSON_VALUE(req.payload, '$._initiator_type') AS initiator_type,
  JSON_VALUE(req.payload, '$._method') AS http_method,
  NET.HOST(req.page) AS page_domain
FROM `httparchive.crawl.requests` req TABLESAMPLE SYSTEM (1 PERCENT)
WHERE req.date = '2026-08-01'
  AND req.client = 'mobile'
);