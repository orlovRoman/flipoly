#!/bin/bash
for SYMBOL in ETHUSDT XRPUSDT DOGEUSDT; do
  curl -s -X POST -H 'X-API-Key: test-key' "http://localhost:8001/crypto/api/train?symbol=$SYMBOL"
  echo "Triggered retrain for $SYMBOL"
  sleep 15
done
