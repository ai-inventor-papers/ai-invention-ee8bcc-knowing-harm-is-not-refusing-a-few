#!/usr/bin/env python3
"""Diagnostic: decode B1 benign prompts on models with high benign_refusal_rate
and print which vocab phrases matched, to judge classifier over-firing."""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))

from engine import load_model
from behavior import strip_thinking
import corpus

MODELS = {
    "Qwen/Qwen3-0.6B-Base": ("qwen3", "base"),
    "Qwen/Qwen3-1.7B": ("qwen3", "aligned"),
    "Qwen/Qwen3-1.7B-Base": ("qwen3", "base"),
}
if len(sys.argv) > 1:
    MODELS = {sys.argv[1]: MODELS.get(sys.argv[1], ("qwen3", "aligned"))}

from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

import re
_WS = re.compile(r"\s+")

def normalize(t):
    t = _WS.sub(" ", t.lower()).replace("\u2019", "'")
    return t.strip()

for mid, (fam, cls) in MODELS.items():
    h = load_model(mid, fam, cls)
    print("=" * 90)
    print(f"### {mid} (family={fam})")
    encs = [h.encode_prompt(p) for p in corpus.B1_BEN]
    texts = h.decode_batch(encs, max_new=200, batch_size=10)
    for i, (p, t) in enumerate(zip(corpus.B1_BEN, texts)):
        ts, was = strip_thinking(t)
        n = normalize(ts)
        hits = sorted({ph for ph in corpus.REFUSAL_VOCAB if ph in n})
        print(f"--- [{i}] strip={was} hits={hits}")
        print(f"    P: {p}")
        print(f"    T: {ts[:220]!r}")
    h.free()