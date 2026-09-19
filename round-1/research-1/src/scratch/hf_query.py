import json, urllib.request, urllib.parse, sys, time

def api(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-audit/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                return {"error": str(e)}
            time.sleep(2)

def show(models, wanted=None):
    for m in models:
        if "error" in m:
            print("ERR", m["error"]); continue
        rid = m.get("id")
        if wanted and not any(w.lower() in rid.lower() for w in wanted):
            continue
        tags = m.get("tags", [])
        params = [t for t in tags if t.replace(".", "", 1).isdigit() or (t[0].isdigit() and "B" in t)][:3]
        print(f"{rid} | dl={m.get('downloads')} | likes={m.get('likes')} | params={params} | license={m.get('license')} | lastMod={str(m.get('lastModified'))[:10]} | gated={m.get('gated')}")

query = sys.argv[1] if len(sys.argv) > 1 else "Qwen3"
limit = sys.argv[2] if len(sys.argv) > 2 else "60"
url = "https://huggingface.co/api/models?search=" + urllib.parse.quote(query) + "&sort=downloads&limit=" + limit
print("=== search:", query, "===")
data = api(url)
if isinstance(data, list):
    for m in data:
        rid = m.get("id")
        tags = m.get("tags", [])
        params = [t for t in tags if t.replace(".", "", 1).isdigit() or (t[0].isdigit() and "B" in t)][:3]
        print(f"{rid} | dl={m.get('downloads')} | likes={m.get('likes')} | params={params} | license={m.get('license')} | lastMod={str(m.get('lastModified'))[:10]} | gated={m.get('gated')}")
else:
    print("API error:", data)