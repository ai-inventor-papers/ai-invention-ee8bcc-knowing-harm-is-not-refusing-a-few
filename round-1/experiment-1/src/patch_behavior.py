#!/usr/bin/env python3
"""Re-run behavioral ground truth (decode + classify) for every checkpointed
model, patching behavior/b2 fields in place.

v3 (2026-09-19, executor review): the cached behavior fields mixed runs at
different decode lengths (0.6B checkpoints patched at max_new=160, 1.7B written
by the mid-stage run at max_new=400).  A direct replication showed the stale
benign rates (0.40/0.50) do not reproduce with the current strip_v2+extended
vocab code (0.10/0.00).  This pass re-decodes EVERY checkpoint with the SAME
config the pipeline uses (max_new=400, greedy, strip_v2, extended vocab) so the
behavioral ground truth is one homogeneous set across the zoo."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from loguru import logger

from behavior import run_behavior, write_spotcheck_review
from corpus import B1_BEN, B1_HARM, B2_BEN, B2_HARM
from engine import load_model

WS = Path(__file__).resolve().parent


@logger.catch(reraise=True)
def main() -> None:
    import sys
    only = set(sys.argv[1:])  # optional: explicit model ids to patch (else all)
    cps = sorted((WS / "caches" / "per_model").glob("*.json"))
    if only:
        cps = [cp for cp in cps if json.loads(cp.read_text())["meta"]["id"] in only]
    logger.info(f"patching behavior for {len(cps)} models (only={sorted(only) or 'all'})")
    for cp in cps:
        j = json.loads(cp.read_text())
        mid = j["meta"]["id"]
        logger.info(f"=== re-decode {mid}")
        h = load_model(mid, j["meta"]["family"], j["meta"]["cls"])
        try:
            beh = run_behavior(h, B1_HARM, B1_BEN, max_new=400, seed=0)
            beh2 = run_behavior(h, B2_HARM, B2_BEN, max_new=400, seed=0)
        finally:
            h.free()
        del h
        gc.collect()
        j["behavior"] = {
            "refusal_rate": beh["refusal_rate"],
            "benign_refusal_rate": beh["benign_refusal_rate"],
            "n_harm": beh["n_harm"], "n_ben": beh["n_ben"],
            "n_refused": beh["n_refused"],
            "exemplars": beh["exemplars"],
            "labels_harm": beh["labels_harm"],
            "n_think_blocks_stripped": beh["n_think_blocks_stripped"],
            "n_deliberation_only": beh["n_deliberation_only"],
        }
        j["b2_reserved"] = {"refusal_rate": beh2["refusal_rate"], "n": beh2["n_harm"],
                            "n_think_blocks_stripped": beh2["n_think_blocks_stripped"]}
        cp.write_text(json.dumps(j, indent=1))
        write_spotcheck_review(WS / "caches" / "spotcheck", mid, beh)
        logger.info(f"patched {mid} rate={beh['refusal_rate']:.2f}")
    logger.info("behavior patch pass done")


if __name__ == "__main__":
    main()