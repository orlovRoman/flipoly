BEGIN;
UPDATE exposure_reservations e
SET released_at = now()
FROM execution_requests r
WHERE e.request_id = r.id
  AND e.released_at IS NULL
  AND r.state IN (
      'FILLED',
      'PARTIALLY_FILLED_FINAL',
      'REJECTED',
      'EXPIRED',
      'CANCELED'
  );
COMMIT;
