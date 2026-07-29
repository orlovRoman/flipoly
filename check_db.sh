#!/bin/bash
# check_db.sh - find DB credentials and size
echo "=== Env vars in polyflip_api ==="
docker exec polyflip_api env | grep -iE 'postgres|db_url|database|pguser|pgpassword|pghost' | head -20
echo ""
echo "=== DB container env ==="
docker inspect polyflip_db --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -iE 'postgres|user|pass|db'
