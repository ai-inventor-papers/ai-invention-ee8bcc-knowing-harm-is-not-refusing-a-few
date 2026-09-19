#!/usr/bin/env python3
"""03_verify_zoo.py — verify every model-zoo candidate on the HF Hub API and
emit sources/zoo_manifest.json.

Primary ids come from the artifact plan; EVERY id is verified at execution
time via GET /api/models/<id>. Slots whose primary id 404s are substituted
with the first verified ungated community id returned by one HF search
(recorded in metadata_substitution_note); gated primaries are kept and a
fallback mirror is recorded. No weights are downloaded.

Zero LLM API calls. Results cached in sources/hf_api_cache.json.
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path

import requests
from loguru import logger

WORKSPACE = Path(__file__).resolve().parents[1]
CACHE_FILE = WORKSPACE / "sources" / "hf_api_cache.json"
OUT = WORKSPACE / "sources" / "zoo_manifest.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

HF_API = "https://huggingface.co/api"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aii-zoo-verifier/1.0"})
if HF_TOKEN:
    SESSION.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})


def _cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save(c: dict) -> None:
    CACHE_FILE.write_text(json.dumps(c, indent=1))


def api_get(path: str, retries: int = 3) -> dict | None:
    """Cached GET with retry+backoff; None on non-200."""
    c = _cache()
    if path in c:
        return c[path]
    url = f"{HF_API}/{path}"
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                c[path] = data
                _save(c)
                return data
            if resp.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            if resp.status_code == 404:
                c[path] = None
                _save(c)
                return None
            time.sleep(2**attempt)
        except requests.RequestException:
            time.sleep(2**attempt)
    return None


def model_info(model_id: str) -> dict | None:
    return api_get(f"models/{model_id}")


def detect_template(raw: str) -> str:
    if "<|im_start|>" in raw or "im_start" in raw:
        return "chatml"
    if "start_header_id" in raw or "<|start_header_id" in raw:
        return "llama3"
    if "start_of_turn" in raw or "<start_of_turn>" in raw:
        return "gemma"
    if "[INST]" in raw:
        return "mistral"
    return "custom"


def verify_one(model_id: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    row = {"hf_id": model_id, "verified": False, "verified_at": now}
    info = model_info(model_id)
    if info is None:
        row["verification_method"] = "hf_api_GET 404 (id does not exist on the Hub)"
        return row
    row["verified"] = True
    row["verification_method"] = f"hf_api_GET /api/models/{model_id} HTTP 200"
    row["sha"] = info.get("sha", "")
    row["gated"] = bool(info.get("gated", False))
    row["downloads"] = info.get("downloads", 0)
    row["likes"] = info.get("likes", 0)
    lic = [t for t in info.get("tags", []) if t.startswith("license:")]
    row["license"] = lic[0].split("license:", 1)[1] if lic else "unknown"
    row["license_tags"] = lic
    st = info.get("safetensors") or {}
    row["params_b"] = st.get("total")
    cfg = info.get("config") or {}
    if row["params_b"] is None and isinstance(cfg, dict):
        row["params_b"] = cfg.get("num_parameters")
    tmpl = cfg.get("chat_template") if isinstance(cfg, dict) else None
    row["template"] = detect_template(tmpl) if isinstance(tmpl, str) else "unknown"
    row["fp16_gb"] = round(row["params_b"] * 2 / 1e9, 1) if isinstance(row["params_b"], (int, float)) else None
    return row


QUANT_SKIP = ["gguf", "ggml", "gptq", "awq", "mlx", "onnx", "upload", "fp8", "int8",
              "int4", "bnb", "quantized", "safetensors-upload"]


def search_ungated_candidates(query: str, base_token: str) -> list[str]:
    """Ordered list of ungated-looking candidate ids from one HF search."""
    q = requests.utils.quote(query)
    data = api_get(f"models?search={q}&limit=25")
    if not isinstance(data, list):
        return []
    out = []
    for m in data:
        mid = m.get("id", "")
        if m.get("gated"):  # search results occasionally carry the flag
            continue
        if base_token.lower() in mid.lower():
            out.append(mid)
    return out


def find_substitute(query: str, base_token: str, require_full_precision: bool) -> str:
    """First verified ungated id from the search list (optionally non-quantized).

    Substitutes must expose safetensors weights (params observable) so the
    experiment can instantiate them with transformers/torch.
    """
    for mid in search_ungated_candidates(query, base_token):
        if require_full_precision and any(tok in mid.lower() for tok in QUANT_SKIP):
            continue
        info = model_info(mid)
        if info is None:
            continue
        if bool(info.get("gated", False)):
            continue
        if require_full_precision:
            st = info.get("safetensors") or {}
            if not st.get("total"):
                continue  # no safetensors weights (e.g. tflite/flax-only repo)
        return mid
    return "UNVERIFIED"


# slot: (slot_id, primary, family, size_b, klass, template_expect, sub_query, sub_token)
SLOTS = [
    ("q3_060_base", "Qwen/Qwen3-0.6B-Base", "qwen3", 0.6, "base", "qwen3", "Qwen3-0.6B", "qwen3-0.6b"),
    ("q3_060_instruct", "Qwen/Qwen3-0.6B-Instruct", "qwen3", 0.6, "instruct", "qwen3", "Qwen3-0.6B-Instruct", "qwen3-0.6b-instruct"),
    ("q3_060_ablit", "huihui-ai/Qwen3-0.6B-abliterated", "qwen3", 0.6, "abliterated", "qwen3", "Qwen3-0.6B abliterated", "abliterated"),
    ("q3_170_base", "Qwen/Qwen3-1.7B-Base", "qwen3", 1.7, "base", "qwen3", "Qwen3-1.7B", "qwen3-1.7b"),
    ("q3_170_instruct", "Qwen/Qwen3-1.7B-Instruct", "qwen3", 1.7, "instruct", "qwen3", "Qwen3-1.7B-Instruct", "qwen3-1.7b-instruct"),
    ("q3_170_ablit", "huihui-ai/Qwen3-1.7B-abliterated", "qwen3", 1.7, "abliterated", "qwen3", "Qwen3-1.7B abliterated", "abliterated"),
    ("q3_4b_base", "Qwen/Qwen3-4B-Base", "qwen3", 4.0, "base", "qwen3", "Qwen3-4B", "qwen3-4b"),
    ("q3_4b_instruct", "Qwen/Qwen3-4B-Instruct-2507", "qwen3", 4.0, "instruct", "qwen3", "Qwen3-4B-Instruct", "qwen3-4b-instruct"),
    ("q3_4b_ablit", "huihui-ai/Qwen3-4B-abliterated", "qwen3", 4.0, "abliterated", "qwen3", "Qwen3-4B abliterated", "abliterated"),
    ("llm_1b_base", "meta-llama/Llama-3.2-1B", "llama3.2", 1.0, "base", "llama3", "Llama-3.2-1B", "llama-3.2-1b"),
    ("llm_1b_instruct", "meta-llama/Llama-3.2-1B-Instruct", "llama3.2", 1.0, "instruct", "llama3", "Llama-3.2-1B-Instruct", "llama-3.2-1b-instruct"),
    ("llm_1b_ablit", "huihui-ai/Llama-3.2-1B-Instruct-abliterated", "llama3.2", 1.0, "abliterated", "llama3", "Llama-3.2-1B abliterated", "abliterated"),
    ("llm_3b_base", "meta-llama/Llama-3.2-3B", "llama3.2", 3.0, "base", "llama3", "Llama-3.2-3B", "llama-3.2-3b"),
    ("llm_3b_instruct", "meta-llama/Llama-3.2-3B-Instruct", "llama3.2", 3.0, "instruct", "llama3", "Llama-3.2-3B-Instruct", "llama-3.2-3b-instruct"),
    ("llm_3b_ablit", "huihui-ai/Llama-3.2-3B-Instruct-abliterated", "llama3.2", 3.0, "abliterated", "llama3", "Llama-3.2-3B abliterated", "abliterated"),
    ("gemma_2b_it", "google/gemma-2-2b-it", "gemma2", 2.0, "instruct", "gemma", "Gemma-2-2b-it", "gemma-2-2b"),
    ("gemma_1b_it", "google/gemma-3-1b-it", "gemma3", 1.0, "instruct", "gemma", "Gemma-3-1b-it", "gemma-3-1b"),
    ("gemma_4b_it", "google/gemma-3-4b-it", "gemma3", 4.0, "instruct", "gemma", "Gemma-3-4b-it", "gemma-3-4b"),
    ("phi_35", "microsoft/Phi-3.5-mini-instruct", "phi3", 3.8, "instruct", "phi", "Phi-3.5-mini-instruct", "phi-3.5-mini-instruct"),
    ("phi_4mini", "microsoft/Phi-4-mini-instruct","phi4", 3.8, "instruct", "phi", "Phi-4-mini-instruct", "phi-4-mini-instruct"),
    ("ministral", "mistralai/Ministral-3B-instruct-2410", "ministral", 3.0, "instruct", "mistral", "Ministral-3B-instruct", "ministral-3b"),
    ("smollm2", "allenai/SmolLM2-1.7B-Instruct", "smollm2", 1.7, "instruct", "chatml", "SmolLM2-1.7B-Instruct", "smollm2-1.7b"),
]


@logger.catch(reraise=True)
def main() -> None:
    rows: list[dict] = []
    subst_log: list[dict] = []
    for slot_id, primary, family, size, klass, _tmpl, sub_q, sub_tok in SLOTS:
        row = verify_one(primary)
        row["slot"] = slot_id
        row["family"] = family
        row["size"] = size
        row["class"] = klass
        row["substitution_note"] = ""
        row["fallback_mirror"] = ""
        if not row["verified"]:
            sub = find_substitute(sub_q, sub_tok, require_full_precision=True)
            if sub == "UNVERIFIED":
                row["substitution_note"] = f"primary {primary} 404s on the Hub; no verified full-precision ungated substitute found"
                subst_log.append({"slot": slot_id, "primary": primary, "status": "404", "substitute": None})
                logger.warning(f"{slot_id}: {primary} 404, no substitute")
                continue  # do not emit non-existent models
            sub_row = verify_one(sub)
            sub_row["slot"] = slot_id
            sub_row["family"] = family
            sub_row["size"] = size
            sub_row["class"] = klass
            sub_row["substitution_note"] = f"primary {primary} 404s on the Hub; substituted with verified ungated {sub}"
            subst_log.append({"slot": slot_id, "primary": primary, "status": "404", "substitute": sub})
            rows.append(sub_row)
            logger.info(f"{slot_id}: {primary} 404 -> substitute {sub}")
        else:
            if row["gated"]:
                mirror = find_substitute(sub_q, sub_tok, require_full_precision=False)
                row["fallback_mirror"] = mirror
                logger.info(f"{slot_id}: gated {primary} -> mirror {mirror}")
            rows.append(row)
            logger.info(f"{slot_id}: {primary} verified params={row['params_b']} "
                        f"gated={row['gated']} lic={row['license']}")

    # random-control row (config-only; no weights stored)
    rows.append({
        "input": "CONFIG_ONLY:Qwen/Qwen3-0.6B",
        "output": "random-control",
        "hf_id": "CONFIG_ONLY:Qwen/Qwen3-0.6B",
        "slot": "random-control",
        "family": "qwen3",
        "size": 0.6,
        "class": "random-control",
        "template": "n/a(config only)",
        "params_b": 0.6e9,
        "fp16_gb": 1.2,
        "gated": False,
        "license": "n/a",
        "seed": 42,
        "config_source": "Qwen/Qwen3-0.6B",
        "verified": True,
        "verification_method": "config-only row; weights instantiated by the experiment with torch.manual_seed(42); no weights stored in this bundle",
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "substitution_note": "",
        "fallback_mirror": "",
    })

    OUT.write_text(json.dumps({"rows": rows, "substitution_log": subst_log}, indent=1))
    logger.info(f"zoo manifest written: {len(rows)} model rows (incl. 1 control) -> {OUT}")


if __name__ == "__main__":
    main()