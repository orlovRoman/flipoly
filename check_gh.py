import urllib.request
import json
req = urllib.request.Request('https://api.github.com/repos/orlovRoman/flipoly/actions/runs?branch=main&per_page=1', headers={'User-Agent': 'Mozilla'})
res = json.loads(urllib.request.urlopen(req).read())['workflow_runs'][0]
print(f"Status: {res['status']}, Conclusion: {res['conclusion']}, URL: {res['html_url']}")
