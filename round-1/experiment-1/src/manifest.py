"""Model zoo manifest — verified at runtime by config.json download probes.

The HF API "gated" flag proved unreliable from this network, so verification =
a real hf_hub_download probe per model id (token=False).

Class labels follow the verified HF layout (checked 2026-09-19 by the dataset
and research steps against the live API + downloaded configs):
- Qwen/Qwen3-{0.6B,1.7B,4B}       = the CONVERSATIONAL (instruct) checkpoints
  (their tokenizers ship the ChatML chat_template and they emit thinking
  blocks); class = aligned.
- Qwen/Qwen3-{0.6B,1.7B,4B}-Base   = the true base checkpoints; class = base.
- mlabonne/Qwen3-{0.6B,1.7B,4B}-abliterated  = community abliterated (ungated);
  class = abliterated.  (huihui-ai abliterated repos are gated from this
  network -> mlabonne substitutued; both are one-direction orthogonalization
  recipes on the instruct models.)
- Llama-3.2-1B: base = NousResearch mirror; aligned = unsloth mirror;
  abliterated = mylesgoose (meta-llama repos are gated).
- Gemma-3-1B: aligned = unsloth/gemma-3-1b-it; abliterated = lunahr;
  base = unsloth/gemma-3-1b (google repos are gated; slot dropped if the
  mirror is unavailable).
- Bonus aligned singletons (from the dataset step's verified manifest,
  ungated): HuggingFaceTB/SmolLM2-1.7B-Instruct, microsoft/Phi-3.5-mini-instruct.
  These are NOT triplets; their native templates are reproduced as fixed
  family strings (smollm2 ChatML, phi3) so positions stay comparable.

Final zoo target = 17 slots + random-weights control.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

ZOO_SLOTS: dict[str, dict[str, Any]] = {
    # ---- Qwen3 triplets (base / aligned / abliterated) x 3 sizes
    "qwen3_0.6b_base": {"id": "Qwen/Qwen3-0.6B-Base", "family": "qwen3", "cls": "base", "size": "0.6B"},
    "qwen3_0.6b_aligned": {"id": "Qwen/Qwen3-0.6B", "family": "qwen3", "cls": "aligned", "size": "0.6B"},
    "qwen3_0.6b_abliterated": {"id": "mlabonne/Qwen3-0.6B-abliterated", "family": "qwen3", "cls": "abliterated", "size": "0.6B"},
    "qwen3_1.7b_base": {"id": "Qwen/Qwen3-1.7B-Base", "family": "qwen3", "cls": "base", "size": "1.7B"},
    "qwen3_1.7b_aligned": {"id": "Qwen/Qwen3-1.7B", "family": "qwen3", "cls": "aligned", "size": "1.7B"},
    "qwen3_1.7b_abliterated": {"id": "mlabonne/Qwen3-1.7B-abliterated", "family": "qwen3", "cls": "abliterated", "size": "1.7B"},
    "qwen3_4b_base": {"id": "Qwen/Qwen3-4B-Base", "family": "qwen3", "cls": "base", "size": "4B"},
    "qwen3_4b_aligned": {"id": "Qwen/Qwen3-4B", "family": "qwen3", "cls": "aligned", "size": "4B"},
    "qwen3_4b_abliterated": {"id": "mlabonne/Qwen3-4B-abliterated", "family": "qwen3", "cls": "abliterated", "size": "4B"},
    # ---- Llama-3.2-1B set
    "llama_1b_base": {"id": "NousResearch/Llama-3.2-1B", "family": "llama3", "cls": "base", "size": "1B"},
    "llama_1b_instruct": {"id": "unsloth/Llama-3.2-1B-Instruct", "family": "llama3", "cls": "aligned", "size": "1B"},
    "llama_1b_abliterated": {"id": "mylesgoose/Llama-3.2-1B-Instruct-abliterated", "family": "llama3", "cls": "abliterated", "size": "1B"},
    # ---- Gemma-3-1B set
    "gemma_1b_base": {"id": "unsloth/gemma-3-1b", "family": "gemma", "cls": "base", "size": "1B"},
    "gemma_1b_it": {"id": "unsloth/gemma-3-1b-it", "family": "gemma", "cls": "aligned", "size": "1B"},
    "gemma_1b_abliterated": {"id": "lunahr/gemma-3-1b-it-abliterated", "family": "gemma", "cls": "abliterated", "size": "1B"},
    # ---- Bonus cross-family aligned singletons (native template via family string)
    "smollm2_1.7b_instruct": {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "family": "smollm2", "cls": "aligned", "size": "1.7B"},
    "phi3_3.8b_mini": {"id": "microsoft/Phi-3.5-mini-instruct", "family": "phi3", "cls": "aligned", "size": "3.8B"},
}


def _probe(model_id: str) -> dict[str, Any]:
    """Probe by downloading config.json with token=False."""
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model_id, "config.json", token=False)
        return {"id": model_id, "ok": True, "error": ""}
    except Exception as exc:
        return {"id": model_id, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:100]}"}


def short_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)


def verify_manifest(max_workers: int = 4) -> dict[str, Any]:
    """Re-probe every slot; drop failures with notes."""
    ids = list({s["id"] for s in ZOO_SLOTS.values()})
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_probe, cid): cid for cid in ids}
        for fut in as_completed(futs):
            res = fut.result()
            results[res["id"]] = res
    slots: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for k, s in ZOO_SLOTS.items():
        mid = s["id"]
        info = results.get(mid, {})
        if info.get("ok"):
            slots[k] = {**s, "status": "ok", "short_id": short_id(mid), "notes": ""}
            logger.info(f"slot {k} -> {mid}")
        else:
            notes.append(f"{k}: {mid} not downloadable ({info.get('error')}) -> slot dropped")
            slots[k] = {**s, "status": "dropped", "notes": notes[-1]}
    return {"slots": slots, "notes": notes, "verified_at": "2026-09-19",
            "method": "config.json download probe (token=False)"}


def save_manifest(path: Path) -> dict[str, Any]:
    manifest = verify_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"manifest saved: {path}")
    return manifest


if __name__ == "__main__":
    import sys
    from loguru import logger as _lg
    _lg.remove()
    _lg.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    m = verify_manifest()
    print(json.dumps({k: v["id"] for k, v in m["slots"].items()}, indent=1))
    print("NOTES:")
    for n in m["notes"]:
        print(" -", n)
    sys.exit(0)