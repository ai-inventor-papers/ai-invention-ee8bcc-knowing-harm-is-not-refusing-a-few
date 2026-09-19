#!/usr/bin/env python3
"""Preview HF datasets locally (bypassing the broken ability-server preview endpoint).

Usage:
    python scripts/00_preview.py DATASET_ID [--split SPLIT] [--config CONFIG] [--rows N]

Calls the skill's core_preview_dataset function directly in-process.
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

WORKSPACE = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path("/ai-inventor/.claude/skills/aii-hf-datasets/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))

import aii_hf_preview_datasets as preview_mod  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs/run.log"), rotation="30 MB", level="DEBUG")


@logger.catch(reraise=True)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()

    preview_mod.init_preview_dataset()
    result = preview_mod.core_preview_dataset(
        dataset_id=args.dataset_id, config=args.config, split=args.split, num_rows=args.rows
    )
    if not result.get("success"):
        logger.error(f"Preview failed for {args.dataset_id}: {result.get('sample_error') or result.get('error')}")
        sys.exit(1)

    out = {
        "dataset_id": result["dataset_id"],
        "downloads": result.get("downloads"),
        "likes": result.get("likes"),
        "tags": result.get("tags"),
        "configs": result.get("configs"),
        "split": result.get("split"),
        "columns": result.get("columns"),
        "num_sample_rows": result.get("num_sample_rows"),
        "sample_rows": result.get("sample_rows"),
    }
    print(json.dumps(out, indent=1, default=str))
    logger.info(f"Preview OK: {args.dataset_id} split={args.split}")


if __name__ == "__main__":
    main()