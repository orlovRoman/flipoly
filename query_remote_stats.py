import urllib.request
import urllib.error
import json

api_key = "test-key"
base_urls = [
    "http://89.223.120.30:8001",
    "http://89.223.120.30:8000",
]

for base_url in base_urls:
    url = f"{base_url}/api/backtest/dataset_stats"
    print(f"Querying {url}...")
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req) as f:
            res = json.loads(f.read().decode('utf-8'))
            print("SUCCESS!")
            print(json.dumps(res, indent=2))
            break
    except urllib.error.HTTPError as e:
        print("HTTP ERROR", e.code)
        try:
            print(e.read().decode('utf-8'))
        except:
            pass
    except Exception as e:
        print("ERROR", e)
