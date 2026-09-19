#!/usr/bin/env python3
"""Post-run behavioral review for Qwen3-family models.

Writes review spot-check bundles (STRIPPED final answers, not deliberations)
for every qwen3 model so the executor can manually label 4 random outputs per
model and compare against the keyword heuristic (plan testing item 5).

Usage: .venv/bin/python review_behavior.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os

WS = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(WS / "hf_cache"))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

from corpus_dataset import load_corpus_with_fallback  # noqa: E402
from behavior import run_behavior, write_spotcheck_review  # noqa: E402
from engine import load_model  # noqa: E402
from manifest import ZOO_SLOTS  # noqa: E402


def main() -> None:
    corpus = load_corpus_with_fallback()
    ids = sorted({s["id"] for s in ZOO_SLOTS.values() if s["family"] == "qwen3"},
                 key=lambda m: (ZOO_SLOTS[next(k for k, v in ZOO_SLOTS.items() if v["id"] == m)]["size"], m))
    out_dir = WS / "caches" / "spotcheck_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        fam = ZOO_SLOTS[next(k for k, v in ZOO_SLOTS.items() if v["id"] == mid)]["family"]
        cls = ZOO_SLOTS[next(k for k, v in ZOO_SLOTS.items() if v["id"] == mid)]["cls"]
        logger.info(f"=== review decode {mid} ({cls}) ===")
        h = load_model(mid, fam, cls)
        try:
            res = run_behavior(h, corpus["B1_HARM"], corpus["B1_BEN"], max_new=400)
        finally:
            h.free()
        write_spotcheck_review(out_dir, mid, res)
        with open(out_dir / f"{mid.replace('/', '__')}.summary.json", "w") as f:
            json.dump({"id": mid, "cls": cls,
                       "refusal_rate": res["refusal_rate"],
                       "benign_refusal_rate": res["benign_refusal_rate"],
                       "n_refused": res["n_refused"], "labels_harm": res["labels_harm"],
                       "n_deliberation_only": res["n_deliberation_only"]}, f, indent=1)
        logger.info(f"rate={res['refusal_rate']:.2f} ben={res['benign_refusal_rate']:.2f} "
                    f"delib_only={res['n_deliberation_only']}")
    print(f"review bundles written to {out_dir}")


if __name__ == "__main__":
    main()