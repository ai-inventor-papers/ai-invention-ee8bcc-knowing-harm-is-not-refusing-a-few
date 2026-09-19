#!/usr/bin/env python3
"""Fetch key pages and save markdown text to scratch/pages/ for quote verification."""
import subprocess, sys, os, json

SKILL_DIR = "/ai-inventor/.claude/skills/aii-web-tools"
PY = SKILL_DIR + "/../.ability_client_venv/bin/python"
FETCH = [PY, SKILL_DIR + "/scripts/aii_fast_web_fetch.py", "fetch"]
OUT = "/ai-inventor/aii_data/runs/run_hhQTjY478FHc/3_invention_loop/iter_1/gen_art/gen_art_research_1/scratch/pages"
os.makedirs(OUT, exist_ok=True)

PAGES = {
 "basu_abs": "https://arxiv.org/abs/2603.18353",
 "llorente_harm_abs": "https://arxiv.org/abs/2604.18901",
 "latentbiopsy_abs": "https://arxiv.org/abs/2603.27412",
 "jiang_abs": "https://arxiv.org/abs/2606.08044",
 "ras_abs": "https://arxiv.org/abs/2606.25750",
 "hildebrant_abs": "https://arxiv.org/abs/2501.08145",
 "refusal_tokens_abs": "https://arxiv.org/abs/2412.06748",
 "khatri_abs": "https://arxiv.org/abs/2609.19472",
 "refusal_before_decoding_abs": "https://arxiv.org/abs/2605.28553",
 "joad_abs": "https://arxiv.org/abs/2602.02132",
 "hurtado_abs": "https://arxiv.org/abs/2607.01854",
 "arditi_abs": "https://arxiv.org/abs/2406.11717",
 "arditi_html": "https://arxiv.org/html/2406.11717v3",
 "ams_zenodo": "https://zenodo.org/records/19501951",
 "ams_blog": "https://opensource.googleblog.com/2026/04/introducing-ams-activation-based-model-scanner-for-open-weight-llm-safety-verification.html",
 "qwen3_readme": "https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md",
}

meta = {}
for name, url in PAGES.items():
    r = subprocess.run(FETCH + ["--url", url, "--max-chars", "200000"], capture_output=True, text=True, timeout=180)
    text = r.stdout
    # strip the leading "URL: ... / Type: ... / Length: ..." header lines
    lines = text.split("\n")
    body = []
    started = False
    for ln in lines:
        if ln.startswith("--- Content ---"):
            started = True
            continue
        if started:
            body.append(ln)
    bodytext = "\n".join(body).strip()
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        f.write(bodytext)
    meta[name] = {"url": url, "chars": len(bodytext), "ok": len(bodytext) > 300}
    print(name, "->", len(bodytext), "chars", "OK" if len(bodytext) > 300 else "SHORT")

with open(os.path.join(OUT, "_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("done")