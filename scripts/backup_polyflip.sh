#!/bin/bash
# backup_polyflip.sh
# Делает pg_dump базы polyflip и сохраняет в /home/orlovrp/backups/
# Хранит последние 7 дней (удаляет старше 7 дней)

set -euo pipefail

BACKUP_DIR="/home/orlov/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M)
FILENAME="${BACKUP_DIR}/polyflip_${TIMESTAMP}.sql.gz"

mkdir -p "${BACKUP_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting backup -> ${FILENAME}"

docker exec -e PGPASSWORD=secret polyflip_db pg_dump \
    -U polyflip \
    -d polyflip \
    | gzip -9 > "${FILENAME}"

SIZE=$(du -sh "${FILENAME}" | cut -f1)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: ${FILENAME} (${SIZE})"

# Удаляем бэкапы старше 7 дней
find "${BACKUP_DIR}" -name "polyflip_*.sql.gz" -mtime +7 -delete
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Old backups cleaned (>7 days)"
