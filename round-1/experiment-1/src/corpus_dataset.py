"""Canonical probe corpus: reads the standardized prompt bundle produced by the
dataset step of this run (gen_art_dataset_1/data_out/data_out.json, bundle
"safety_screen_bundle_v1") and re-exports the corpus API used by the pipeline:

    A12_HARM, A12_BEN   -- 12 harmful + 12 benign probe prompts (paired)
    A4_IDX,  A8_IDX     -- nested subsets of the 12 harmful (encoded by the
                           bundle's metadata_subsets: p4/p8/p12)
    ANCHOR_PAIRS        -- 6 (benign_i, harmful_i) interpolation pairs
    B1_HARM, B1_BEN     -- 40 harmful + 10 benign behavior ground truth
    B2_HARM, B2_BEN     -- 20 harmful + 5 benign RESERVED (iteration 2 only)

If the bundle is missing or fails validation, CorpusValidationError is raised;
the pipeline then falls back to the embedded corpus in corpus.py (logged).

All prompts come from real, openly licensed sources (BeaverTails, HarmBench,
hh-rlhf, Dolly, alpaca-cleaned, ...) -- zero synthetic generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

# Candidate bundle locations: workspace copy first, then the sibling dataset
# workspace (read-only shared pool), then a shallow glob of the run root.
_BUNDLE_CANDIDATES = [
    Path(__file__).resolve().parent / "corpus" / "data_out.json",
    Path("/ai-inventor/aii_data/runs/run_hhQTjY478FHc/3_invention_loop/iter_1/gen_art/gen_art_dataset_1") / "data_out" / "data_out.json",
]


class CorpusValidationError(Exception):
    """Raised when the dataset bundle is present but malformed."""


def _find_bundle() -> Path | None:
    for p in _BUNDLE_CANDIDATES:
        if p.is_file():
            return p
    # shallow glob of the run root as a last resort
    root = Path("/ai-inventor/aii_data/runs/run_hhQTjY478FHc")
    hits = sorted(root.glob("**/data_out/data_out.json"))
    for h in hits:
        if h.is_file():
            return h
    return None


def _load_examples(ds: dict) -> list[dict]:
    return ds.get("examples", [])


def _validate_probe(examples: list[dict]) -> None:
    if len(examples) != 24:
        raise CorpusValidationError(f"probe set has {len(examples)} examples, expected 24")
    harms = [e for e in examples if e.get("metadata_polarity") == "harmful"]
    bens = [e for e in examples if e.get("metadata_polarity") == "benign"]
    if len(harms) != 12 or len(bens) != 12:
        raise CorpusValidationError(f"probe polarity split {len(harms)}/{len(bens)}, expected 12/12")
    # pairing: harmful[i] and benign[i] must share metadata_pair_id
    ids_h = [e.get("metadata_pair_id") for e in harms]
    ids_b = [e.get("metadata_pair_id") for e in bens]
    if ids_h != ids_b:
        raise CorpusValidationError(f"probe pairing mismatch: {ids_h} vs {ids_b}")
    # subset annotations: A4 in A8 in A12 on the harmful side
    subs = [e.get("metadata_subsets") for e in harms]
    a4 = [i for i, s in enumerate(subs) if s and "p4" in s]
    a8 = [i for i, s in enumerate(subs) if s and "p8" in s]
    if not (set(a4) <= set(a8) and len(a4) == 4 and len(a8) == 8):
        raise CorpusValidationError(f"bad subset nesting: A4={a4} A8={a8}")


def _validate_split(examples: list[dict], n_harm: int, n_ben: int, name: str) -> None:
    if len(examples) != n_harm + n_ben:
        raise CorpusValidationError(f"{name} has {len(examples)} examples, expected {n_harm + n_ben}")
    nh = sum(1 for e in examples if e.get("metadata_polarity") == "harmful")
    nb = sum(1 for e in examples if e.get("metadata_polarity") == "benign")
    if nh != n_harm or nb != n_ben:
        raise CorpusValidationError(f"{name} polarity split {nh}/{nb}, expected {n_harm}/{n_ben}")


def _check_lengths(examples: list[dict], name: str, max_words: int = 120) -> None:
    for e in examples:
        nw = e.get("metadata_len_words")
        inp = e.get("input", "")
        if nw is not None and int(nw) > max_words:
            raise CorpusValidationError(f"{name} prompt > {max_words} words: {inp[:80]!r} ({nw} words)")
        if not inp or not inp.strip():
            raise CorpusValidationError(f"{name} has an empty prompt")


def load_dataset_corpus() -> dict:
    """Load + validate the bundle; return the corpus dict or raise."""
    bundle = _find_bundle()
    if bundle is None:
        raise CorpusValidationError("no dataset bundle found in shared pool")
    try:
        raw = json.loads(bundle.read_text())
    except json.JSONDecodeError as exc:
        raise CorpusValidationError(f"bundle not valid JSON: {exc}") from exc
    if "datasets" not in raw or not isinstance(raw["datasets"], list) or len(raw["datasets"]) != 5:
        raise CorpusValidationError(f"bundle has no 5-dataset structure (got {len(raw.get('datasets', []))})")
    probe, b1, b2, anchors, zoo = raw["datasets"]

    pe = _load_examples(probe)
    _validate_probe(pe)
    _check_lengths(pe, "probe")
    _validate_split(_load_examples(b1), 40, 10, "B1")
    _check_lengths(_load_examples(b1), "B1")
    _validate_split(_load_examples(b2), 20, 5, "B2")
    _check_lengths(_load_examples(b2), "B2")
    for e in _load_examples(anchors):
        if not e.get("input", "").strip():
            raise CorpusValidationError("anchor pool has an empty prompt")
    if len(_load_examples(zoo)) < 10:
        raise CorpusValidationError(f"zoo manifest has only {len(_load_examples(zoo))} rows")

    harms = [e for e in pe if e.get("metadata_polarity") == "harmful"]
    bens = [e for e in pe if e.get("metadata_polarity") == "benign"]
    subs = [e.get("metadata_subsets") for e in harms]
    A4_IDX = sorted(i for i, s in enumerate(subs) if s and "p4" in s)
    A8_IDX = sorted(i for i, s in enumerate(subs) if s and "p8" in s)

    def split(lst: list[dict], polarity: str) -> list[str]:
        return [e["input"] for e in lst if e.get("metadata_polarity") == polarity]

    b1_ex, b2_ex = _load_examples(b1), _load_examples(b2)
    b1_harm = split(b1_ex, "harmful")
    b1_ben = split(b1_ex, "benign")
    corpus = {
        "A12_HARM": [e["input"] for e in harms],
        "A12_BEN": [e["input"] for e in bens],
        "A4_IDX": A4_IDX,
        "A8_IDX": A8_IDX,
        # 6 interpolation anchors = the first 6 probe pairs (matches the bundle's
        # anchor pool anchor_01..06 which are exactly these harmful prompts)
        "ANCHOR_PAIRS": [(bens[i]["input"], harms[i]["input"]) for i in range(6)],
        "B1_HARM": b1_harm,
        "B1_BEN": b1_ben,
        "B2_HARM": split(b2_ex, "harmful"),
        "B2_BEN": split(b2_ex, "benign"),
        "CORPUS_SOURCE": {
            "kind": "dataset_bundle",
            "bundle_id": raw.get("metadata", {}).get("bundle_id"),
            "path": str(bundle),
            "version": raw.get("metadata", {}).get("version"),
            "n_prompts": {
                "probe": len(pe), "b1_harm": len(b1_harm),
                "b1_ben": len(b1_ben),
                "b2": len(b2_ex),
                "anchors_pool": len(_load_examples(anchors)),
            },
            "validated": True,
        },
    }
    return corpus


def _extract(corpus: dict):
    return (
        corpus["A12_HARM"], corpus["A12_BEN"], corpus["A4_IDX"], corpus["A8_IDX"],
        corpus["ANCHOR_PAIRS"], corpus["B1_HARM"], corpus["B1_BEN"],
        corpus["B2_HARM"], corpus["B2_BEN"], corpus["CORPUS_SOURCE"],
    )


def load_corpus_with_fallback() -> dict:
    """Try the dataset bundle; fall back to the embedded corpus.py sets."""
    try:
        corpus = load_dataset_corpus()
        src = corpus["CORPUS_SOURCE"]
        logger.info(
            f"corpus source: dataset bundle {src.get('bundle_id')} "
            f"({src.get('path')}) | probe={src['n_prompts']['probe']} "
            f"b1={src['n_prompts']['b1_harm']}+{src['n_prompts']['b1_ben']} "
            f"b2={src['n_prompts']['b2']} anchors_pool={src['n_prompts']['anchors_pool']}"
        )
        return corpus
    except CorpusValidationError as exc:
        import corpus as _emb
        logger.warning(f"dataset bundle unusable ({exc}); falling back to embedded corpus.py")
        return {
            "A12_HARM": _emb.A12_HARM, "A12_BEN": _emb.A12_BEN,
            "A4_IDX": _emb.A4_IDX, "A8_IDX": _emb.A8_IDX,
            "ANCHOR_PAIRS": _emb.ANCHOR_PAIRS,
            "B1_HARM": _emb.B1_HARM, "B1_BEN": _emb.B1_BEN,
            "B2_HARM": _emb.B2_HARM, "B2_BEN": _emb.B2_BEN,
            "CORPUS_SOURCE": {
                "kind": "embedded", "bundle_id": None, "path": "corpus.py",
                "n_prompts": {"probe": 24, "b1_harm": len(_emb.B1_HARM),
                              "b1_ben": len(_emb.B1_BEN), "b2": len(_emb.B2_HARM) + len(_emb.B2_BEN),
                              "anchors_pool": 6},
                "validated": True, "fallback_reason": str(exc),
            },
        }


if __name__ == "__main__":
    from loguru import logger as _lg
    import sys
    _lg.remove()
    _lg.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    c = load_corpus_with_fallback()
    print(json.dumps(c["CORPUS_SOURCE"], indent=1))
    print("A4:", c["A4_IDX"], "A8:", c["A8_IDX"])
    print("A12_HARM[0]:", c["A12_HARM"][0])
    print("anchors[0]:", c["ANCHOR_PAIRS"][0])