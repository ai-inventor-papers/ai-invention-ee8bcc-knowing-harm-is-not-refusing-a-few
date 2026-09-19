import json, urllib.request, urllib.parse, time

def api(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-audit/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "retries exhausted"}

repos = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-1.7B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-3-1b",
    "google/gemma-3-1b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

out = {}
for rid in repos:
    m = api("https://huggingface.co/api/models/" + rid)
    if "error" in m:
        out[rid] = {"error": m["error"]}
        print(f"{rid} | ERROR {m['error']}")
        continue
    tags = m.get("tags", [])
    params = [t for t in tags if t.replace(".", "", 1).isdigit() or (t[0].isdigit() and "B" in t and len(t) < 6)]
    sf = [s for s in m.get("siblings", []) if s.get("rfilename", "").endswith(".safetensors")]
    total_bytes = 0
    largest = ""
    # try size from blobs info via tree API
    tree = api("https://huggingface.co/api/models/" + rid + "/tree/main?recursive=true&expand=false")
    if isinstance(tree, list):
        for item in tree:
            if item.get("path", "").endswith(".safetensors") and item.get("size"):
                total_bytes += item["size"]
                if item["size"] > 1_000_000:
                    largest = item["path"]
    out[rid] = {
        "id": rid,
        "downloads": m.get("downloads"),
        "likes": m.get("likes"),
        "license": m.get("license"),
        "gated": m.get("gated"),
        "tags": tags[:6],
        "params_tags": params,
        "safetensors_n": len(sf),
        "sf_total_bytes_est": total_bytes,
        "lastModified": str(m.get("lastModified"))[:10],
        "pipeline_tag": m.get("pipeline_tag"),
    }
    print(json.dumps(out[rid]))

with open("scratch/hf_repo_meta.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved scratch/hf_repo_meta.json")