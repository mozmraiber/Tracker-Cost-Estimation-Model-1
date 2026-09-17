-- Large-scale per-request features for multi-target models
-- 1% sample = ~3.5M rows (~500MB CSV, exportable as single file)
--
-- Includes ALL targets (transfer_bytes, load_ms, ttfb_ms, download_ms, uncompressed_bytes)
-- and ALL features from 05_per_request_full.sql
--
-- Run in BigQuery console, then Save Results → CSV
-- Or via CLI:
--   bq query --use_legacy_sql=false --destination_table=PROJECT.DATASET.per_request_1pct < sql/06_large_scale.sql
--
-- Estimated cost: ~$5-10

SELECT
  NET.HOST(req.url) AS tracker_domain,

  -- URL features (available at block time)
  REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)') AS url_path,
  ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)'), '/'), '/')) - 1 AS path_depth,
  LOWER(REGEXP_EXTRACT(req.url, r'\.([a-zA-Z0-9]+)(?:\?|#|$)')) AS file_extension,
  REGEXP_CONTAINS(req.url, r'\?') AS has_query_params,
  LENGTH(req.url) AS url_length,
  ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'\?(.*)$'), ''), '&')) AS num_query_params,

  -- Resource type (available at block time)
  req.type AS resource_type,

  -- Request metadata (available at block time)
  JSON_VALUE(req.payload, '$._initiator_type') AS initiator_type,
  JSON_VALUE(req.payload, '$._priority') AS chrome_priority,
  JSON_VALUE(req.payload, '$._method') AS http_method,
  JSON_VALUE(req.payload, '$._protocol') AS http_version,
  CAST(JSON_VALUE(req.payload, '$._is_secure') AS INT64) AS is_https,
  req.index AS waterfall_index,

  -- Page context (available at block time)
  NET.HOST(req.page) AS page_domain,

  -- TARGETS (NOT available at block time — these are what we predict)
  CAST(JSON_VALUE(req.payload, '$._bytesIn') AS INT64) AS transfer_bytes,
  CAST(JSON_VALUE(req.payload, '$._objectSizeUncompressed') AS INT64) AS uncompressed_bytes,
  CAST(JSON_VALUE(req.payload, '$._load_ms') AS INT64) AS load_ms,
  CAST(JSON_VALUE(req.payload, '$._ttfb_ms') AS INT64) AS ttfb_ms,
  CAST(JSON_VALUE(req.payload, '$._download_ms') AS INT64) AS download_ms,
  JSON_VALUE(req.payload, '$._contentType') AS content_type,
  JSON_VALUE(req.payload, '$._contentEncoding') AS content_encoding,

FROM `httparchive.crawl.requests` req
WHERE req.date = '2026-08-01'
  AND req.client = 'mobile'
  AND req.is_root_page = TRUE
  AND NET.HOST(req.url) IN (
    SELECT domain
    FROM `httparchive.almanac.third_parties`
    WHERE date = '2025-07-01'
      AND category IN ('ad', 'analytics', 'social', 'tag-manager', 'consent-provider')
  )
  -- 1% sample (~3.5M rows)
  AND MOD(ABS(FARM_FINGERPRINT(CONCAT(req.page, req.url))), 100) = 0
