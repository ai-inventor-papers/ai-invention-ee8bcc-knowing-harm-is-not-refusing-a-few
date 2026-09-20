"""Iteration-2 model slots: the 7 never-run manifest checkpoints.

Primary IDs come from the iteration-1 dataset bundle's model_zoo_manifest
(verified live 2026-09-19).  Gated primaries fall back to full-precision
mirrors recorded by the iteration-2 probe (2026-09-19 evening): the GGUF
mirrors in the dataset manifest are NOT usable by the fp16 pipeline, so the
mirror table below holds fp16-safe repos only.  Every substitution is LOGGED at
runtime by resolve_never_run_slots(); nothing is substituted silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

# 7 never-run slots from the dataset manifest's zoo (iter_1 did not touch them).
# mirrors: fp16 full-precision fallbacks for gated primaries (verified alive
# 2026-09-19 by config.json config probes).  Order matters: id first, then
# mirrors, first config-ok repo wins.
NEVER_RUN_SLOTS: dict[str, dict[str, Any]] = {
    "llama_3b_base": {
        "id": "meta-llama/Llama-3.2-3B", "family": "llama3", "cls": "base", "size": "3B",
        "mirrors": ["unsloth/Llama-3.2-3B"],
        "note": "primary gated; NousResearch/Llama-3.2-3B mirror 404s (probed 2026-09-19); unsloth/Llama-3.2-3B is the fp16 base fallback",
    },
    "llama_3b_instruct": {
        "id": "meta-llama/Llama-3.2-3B-Instruct", "family": "llama3", "cls": "aligned", "size": "3B",
        "mirrors": ["unsloth/Llama-3.2-3B-Instruct"],
        "note": "primary gated; unsloth/Llama-3.2-3B-Instruct fp16 fallback",
    },
    "llama_3b_abliterated": {
        "id": "huihui-ai/Llama-3.2-3B-Instruct-abliterated", "family": "llama3", "cls": "abliterated", "size": "3B",
        "mirrors": [],
        "note": "ungated primary (verified 2026-09-19)",
    },
    "gemma_2b_it": {
        "id": "google/gemma-2-2b-it", "family": "gemma", "cls": "aligned", "size": "2B",
        "mirrors": ["unsloth/gemma-2-2b-it"],
        "note": "primary gated; unsloth/gemma-2-2b-it fp16 fallback (uses the shared gemma template)",
    },
    "gemma_4b_it": {
        "id": "google/gemma-3-4b-it", "family": "gemma", "cls": "aligned", "size": "4B",
        "mirrors": ["unsloth/gemma-3-4b-it"],
        "note": "primary gated; unsloth/gemma-3-4b-it fp16 fallback; V=262144 (adaptive CONT_SCORE_CHUNK applies)",
    },
    "phi4_mini": {
        "id": "microsoft/Phi-4-mini-instruct", "family": "phi4", "cls": "aligned", "size": "3.8B",
        "mirrors": [],
        "note": "ungated primary (verified 2026-09-19); phi4 ChatML template identical to phi3",
    },
    "ministral_3b": {
        "id": "ministral/Ministral-3b-instruct", "family": "ministral", "cls": "aligned", "size": "3B",
        "mirrors": [],
        "note": "ungated primary (verified 2026-09-19); ministral template: no_gen == gen, first_gen_pos == decision_pos",
    },
}

# Weight-source fallbacks for the iteration-1 in-zoo models that are either 404
# on the Hub today or were evicted from the shared HF cache.  The DEP manifest
# logs the verified substitutes for the two official Qwen3-0.6B/1.7B-Instruct
# repos; iteration-1's final manifest used the official (non -Instruct-suffixed)
# ids which re-verified live on 2026-09-19.
PARTB_WEIGHT_SUBSTITUTES: dict[str, list[str]] = {
    "Qwen/Qwen3-0.6B-Instruct": ["rd211/Qwen3-0.6B-Instruct"],
    "Qwen/Qwen3-1.7B-Instruct": ["rd211/Qwen3-1.7B-Instruct"],
}


def short_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)


def _config_ok(model_id: str) -> bool:
    """True if the repo is loadable anonymously: config.json AND at least one
    tokenizer file (tokenizer.json / tokenizer.model) must be either cached or
    downloadable with token=False.  Cache hits count (the file may live in the
    shared HF cache even for gated repos); a gated repo whose tokenizer files
    are NOT cached is rejected here so the slot falls back to its mirror."""
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model_id, "config.json", token=False)
    except Exception:
        return False
    for tf in ("tokenizer.json", "tokenizer.model"):
        try:
            hf_hub_download(model_id, tf, token=False)
            return True
        except Exception:
            continue
    return False


def _cache_has(model_id: str) -> bool:
    """True if any HF cache dir under HF_HOME already holds the snapshot."""
    import os
    org, _, name = model_id.partition("/")
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    d = hub / f"models--{org}--{name}" / "snapshots"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def resolve_never_run_slots() -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Probe each slot's primary then mirrors; return {slot_key: entry} with the
    resolved `id`, plus the substitution log."""
    resolved: dict[str, dict[str, Any]] = {}
    log: list[str] = []
    for key, slot in NEVER_RUN_SLOTS.items():
        cands = [slot["id"]] + list(slot.get("mirrors", []))
        chosen = None
        for c in cands:
            if _config_ok(c):
                chosen = c
                break
        if chosen is None:
            log.append(f"slot {key}: NO downloadable candidate in {cands} -> DROPPED")
            continue
        entry = dict(slot)
        if chosen != slot["id"]:
            entry["resolved_id"] = chosen
            entry["substitution"] = f"{slot['id']} unavailable -> {chosen}"
            log.append(entry["substitution"])
            logger.warning(f"slot {key}: {entry['substitution']}")
        else:
            entry["resolved_id"] = chosen
            entry["substitution"] = None
            logger.info(f"slot {key}: {chosen} (primary ok)")
        resolved[key] = entry
    return resolved, log


def resolve_partb_weights(model_id: str) -> tuple[str, str]:
    """PART B weight resolution for one in-zoo model.

    Order: (1) shared-pool HF cache of the primary id, (2) primary id download,
    (3) verified substitute (rd211 ...) for the official -Instruct ids.
    Returns (load_id, source) where source in {'cache','hub','substitute-hub'}.
    """
    if _cache_has(model_id):
        return model_id, "cache"
    if _config_ok(model_id):
        return model_id, "hub"
    for sub in PARTB_WEIGHT_SUBSTITUTES.get(model_id, []):
        if _config_ok(sub):
            logger.warning(f"PART B substitute: {model_id} -> {sub}")
            return sub, "substitute-hub"
    return model_id, "not-available"


def resolve_partb_source_log(model_id: str, store: dict[str, Any], source: str, used_id: str) -> None:
    """Record which weight source produced each number (plan step 4)."""
    store.setdefault("weight_source", {})[model_id] = {
        "used_id": used_id, "source": source,
        "note": ("substituted weights: cross-verification marked 'N/A - substituted weights'"
                 if source == "substitute-hub" else "weights identical to iteration-1 id"),
    }


if __name__ == "__main__":
    import sys
    from loguru import logger as _lg
    _lg.remove()
    _lg.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    res, log = resolve_never_run_slots()
    print(json.dumps({k: (v["resolved_id"], v["substitution"]) for k, v in res.items()}, indent=1))
    for line in log:
        print("NOTE:", line)
    print("PARTB probes:")
    for mid in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B-Base", "mlabonne/Qwen3-0.6B-abliterated"]:
        print(" ", mid, "->", resolve_partb_weights(mid))
    sys.exit(0)