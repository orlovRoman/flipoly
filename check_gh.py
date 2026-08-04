import urllib.request, json
req = urllib.request.Request('https://api.github.com/repos/orlovRoman/flipoly/actions/runs/30810882660/jobs')
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    for job in data.get('jobs', []):
        if job.get('conclusion') == 'failure':
            print(f"Failed job: {job['name']}, ID: {job['id']}")
            # We can't fetch step logs easily without auth, but we can just run the test locally using poetry on the server.
