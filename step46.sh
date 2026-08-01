cd /home/orlovrp/flipoly
API_KEY=$(sudo docker compose exec -T api printenv API_KEY < /dev/null | tr -d '\r\n')

curl -sS -X POST "http://127.0.0.1:8001/api/execution/requests/1b62298f-ce15-4d70-8165-6f3858bd2fd4/resolve-review" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" --data '{"action": "MARK_FAILED_NO_FILL","operator": "orlovrp","note": "No provider order, no fills; Polymarket minimum-order validation rejection"}'

curl -sS -X POST "http://127.0.0.1:8001/api/execution/requests/ca9d260c-47fb-4e21-bb9f-aa33cbe818d7/resolve-review" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" --data '{"action": "MARK_FAILED_NO_FILL","operator": "orlovrp","note": "No provider order, no fills; Polymarket minimum-order validation rejection"}'

