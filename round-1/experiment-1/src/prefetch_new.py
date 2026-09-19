"""Prefetch the 5 new model repos (weights + tokenizer) into hf_cache.

NOTE: HF_HOME must be set BEFORE importing huggingface_hub (the hub computes
its cache root at import time); the earlier version imported first and cached
to /root/.cache by mistake."""
import os
from pathlib import Path

WS = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(WS / "hf_cache"))

from huggingface_hub import snapshot_download  # noqa: E402

IDS = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base",
       "HuggingFaceTB/SmolLM2-1.7B-Instruct", "microsoft/Phi-3.5-mini-instruct"]

if __name__ == "__main__":
    for mid in IDS:
        try:
            p = snapshot_download(mid, token=False,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])
            print("OK", mid, p, flush=True)
        except Exception as exc:  # noqa: BLE001 - report-and-continue for prefetch
            print("FAIL", mid, type(exc).__name__, str(exc)[:120], flush=True)