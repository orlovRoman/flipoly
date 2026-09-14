#!/bin/bash
# Этап 3 master chain: funnel(done separately) -> snapsdec -> flipslice -> full.
# Runs sequentially (2GB server). Logs to /tmp/exp_chain.log
{
echo "CHAIN-START $(date -u +%H:%M:%S)"
echo '=== allow-list check ==='
if grep -i -E '(INSERT|UPDATE |DELETE |DROP |ALTER |CREATE |TRUNCATE|GRANT )' /tmp/exp_snapsdec.sh /tmp/exp_flipslice.sh /tmp/exp_full.sh; then
  echo 'FORBIDDEN-SQL-FOUND'; exit 1
fi
echo 'ALLOW-LIST-OK'
bash /tmp/exp_snapsdec.sh
bash /tmp/exp_flipslice.sh
bash /tmp/exp_full.sh
echo "CHAIN-ALLDONE $(date -u +%H:%M:%S)"
} 2>&1
