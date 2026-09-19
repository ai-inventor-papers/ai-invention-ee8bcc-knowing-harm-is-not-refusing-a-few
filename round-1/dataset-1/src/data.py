#!/usr/bin/env python3
"""data.py — assemble the canonical safety-screening evidence bundle.

Loads sources/selection.json (probe/B1/B2/anchor rows) + sources/zoo_manifest.json
(model zoo) + sources/provenance.json, and writes the five grouped datasets in the
repo-standard exp_sel_data_out schema:

  1. probe_corpus          12 harmful + 12 matched benign (nested p4/p8/p12)
  2. behavioral_split_b1   40 harmful + 10 benign   (iteration-1 screen set)
  3. reserved_split_b2     20 harmful + 5 benign    (RESERVED for iteration 2)
  4. interpolation_anchors 52 topic+length-matched (harmful, benign) pairs
  5. model_zoo_manifest    verified HF model registry + random-weights control

Writes data_out/data_out.json (canonical), data_out/full_data_out.json,
data_out/mini_data_out.json, data_out/preview_data_out.json,
data_out/zoo_manifest.csv, README.md. Zero LLM API calls.
"""

import csv
import datetime
import json
import sys
from pathlib import Path

from loguru import logger

WORKSPACE = Path(__file__).resolve().parent
SOURCES = WORKSPACE / "sources"
DATA_OUT = WORKSPACE / "data_out"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

SEED = 20260919

RESERVED_CONTRACT = (
    "B2 (metadata_fold='b2_reserved', metadata_reserved='true') is the iteration-2 "
    "confirmation evidence. Iteration-1 screen artifacts MUST NOT load, sample, "
    "or reference rows with metadata_fold='b2_reserved'. Only iteration-2 "
    "confirmation artifacts may consume them."
)

DS_PROBE = "probe_corpus"
DS_B1 = "behavioral_split_b1"
DS_B2 = "reserved_split_b2"
DS_ANCHOR = "interpolation_anchors"
DS_ZOO = "model_zoo_manifest"

LICENSE_NOTE = {
    "PKU-Alignment/BeaverTails": "cc-by-nc-4.0 (observed card tag)",
    "databricks/databricks-dolly-15k": "cc-by-sa-3.0 (observed card tag)",
    "tatsu-lab/alpaca_eval": "cc-by-nc-4.0 (observed card tag)",
    "yahma/alpaca-cleaned": "cc-by-4.0 (observed card tag)",
    "Anthropic/hh-rlhf": "mit (observed card tag; metadata-only use)",
    "fka/prompts.chat": "cc0-1.0 (observed card tag)",
    "verazuo/jailbreak_llms": "MIT (GitHub repo LICENSE file)",
    "mlabonne/harmful_behaviors": "no license tag observed (recorded as unknown)",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def review_note_for(text: str, review: list) -> str:
    for rv in review:
        if rv.get("new_text") == text and rv.get("old_text"):
            flags = ", ".join(rv.get("flags", []))
            return (f"quality-gate auto-swap; replaced previous pick "
                    f"'{rv['old_text'][:60]}' ({flags})")
    return ""


def row_prompt(h: dict, fold: str, pair_id: str, subsets: str, review: list,
               len_ratio, topic, reserved: bool = False) -> dict:
    r = {
        "input": h["text"],
        "output": "",
        "metadata_fold": fold,
        "metadata_polarity": "harmful",
        "metadata_category": h["category"],
        "metadata_len_chars": h["len_chars"],
        "metadata_len_words": h["len_words"],
        "metadata_source_dataset": h["src"],
        "metadata_source_row": h["src_row"],
    }
    if pair_id:
        r["metadata_pair_id"] = pair_id
    if subsets:
        r["metadata_subsets"] = subsets
    if len_ratio is not None:
        r["metadata_len_ratio"] = len_ratio
    if topic:
        r["metadata_topic"] = topic
    if reserved:
        r["metadata_reserved"] = "true"
    note = review_note_for(h["text"], review)
    if note:
        r["metadata_review_note"] = note
    return r


def row_benign(b: dict, fold: str, pair_id: str, len_ratio, topic,
               reserved: bool = False) -> dict:
    r = {
        "input": b["text"],
        "output": "",
        "metadata_fold": fold,
        "metadata_polarity": "benign",
        "metadata_len_chars": b["len_chars"],
        "metadata_len_words": b["len_words"],
        "metadata_source_dataset": b["src"],
        "metadata_source_row": b["src_row"],
    }
    if pair_id:
        r["metadata_pair_id"] = pair_id
    if len_ratio is not None:
        r["metadata_len_ratio"] = len_ratio
    if topic:
        r["metadata_topic"] = topic
    if reserved:
        r["metadata_reserved"] = "true"
    return r


def build_probe(sel: dict) -> list[dict]:
    pairs = sel["probe_pairs"]
    p4_idx = set(sel["p4_pair_indices"])
    p8_idx = set(sel["p8_pair_indices"])
    rows = []
    for i, pr in enumerate(pairs, start=1):
        if (i - 1) in p4_idx:
            subsets = "p4;p8;p12"
        elif (i - 1) in p8_idx:
            subsets = "p8;p12"
        else:
            subsets = "p12"
        pair_id = f"probe_{i:02d}"
        ratio = pr["benign"].get("len_ratio")
        topic = pr["harmful"]["category"]
        rows.append(row_prompt(pr["harmful"], "probe", pair_id, subsets,
                               sel["review"], ratio, topic))
        rows.append(row_benign(pr["benign"], "probe", pair_id, ratio, topic))
    return rows


def build_b1(sel: dict) -> list[dict]:
    rows = []
    for i, h in enumerate(sel["b1_harmful"], start=1):
        rows.append(row_prompt(h, "b1", f"b1_{i:02d}", "", sel["review"], None, h["category"]))
    for i, b in enumerate(sel["b1_benign"], start=1):
        rows.append(row_benign(b, "b1", f"b1c_{i:02d}", None, None))
    return rows


def build_b2(sel: dict) -> list[dict]:
    rows = []
    for i, h in enumerate(sel["b2_harmful"], start=1):
        rows.append(row_prompt(h, "b2_reserved", f"b2_{i:02d}", "", sel["review"],
                               None, h["category"], reserved=True))
    for i, b in enumerate(sel["b2_benign"], start=1):
        rows.append(row_benign(b, "b2_reserved", f"b2c_{i:02d}", None, None, reserved=True))
    return rows


def build_anchors(sel: dict) -> list[dict]:
    rows = []
    for i, a in enumerate(sel["anchors"], start=1):
        h, b = a["harmful"], a["benign"]
        ratio = b.get("len_ratio")
        rows.append({
            "input": h["text"],
            "output": b["text"],
            "metadata_fold": "interp_anchor",
            "metadata_pair_id": f"anchor_{i:02d}",
            "metadata_topic": h["category"],
            "metadata_len_ratio": ratio,
            "metadata_len_chars": h["len_chars"],
            "metadata_len_words": h["len_words"],
            "metadata_source_dataset": h["src"],
            "metadata_source_row": h["src_row"],
            "metadata_anchor_source": b["src"],
        })
    return rows


def build_zoo(zoo: dict) -> list[dict]:
    rows = []
    for m in zoo["rows"]:
        r = {
            "input": m["hf_id"],
            "output": m["class"],
            "metadata_hf_id": m["hf_id"],
            "metadata_family": m["family"],
            "metadata_size": m["size"],
            "metadata_class": m["class"],
            "metadata_fp16_gb": m.get("fp16_gb"),
            "metadata_params_b": m.get("params_b"),
            "metadata_verified": "true" if m.get("verified") else "false",
            "metadata_verification_method": m.get("verification_method", ""),
            "metadata_verified_at": m.get("verified_at", ""),
            "metadata_gated": "true" if m.get("gated") else "false",
            "metadata_license": m.get("license", "unknown"),
        }
        if m.get("template"):
            r["metadata_template"] = m["template"]
        if m.get("fallback_mirror"):
            r["metadata_fallback_mirror"] = m["fallback_mirror"]
        if m.get("substitution_note"):
            r["metadata_substitution_note"] = m["substitution_note"]
        if m.get("seed") is not None:
            r["metadata_seed"] = m["seed"]
        if m.get("config_source"):
            r["metadata_config_source"] = m["config_source"]
        rows.append(r)
    return rows


def make_mini_preview(bundle: dict, n: int) -> dict:
    out = {"metadata": bundle["metadata"], "datasets": []}
    for ds in bundle["datasets"]:
        exs = []
        for e in ds["examples"][:n]:
            ee = {k: (str(v)[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                  for k, v in e.items()}
            exs.append(ee)
        out["datasets"].append({"dataset": ds["dataset"], "examples": exs})
    return out


def write_zoo_csv(zoo: dict) -> None:
    cols = ["slot", "hf_id", "family", "size", "class", "params_b", "fp16_gb",
            "gated", "license", "verified", "verification_method", "verified_at",
            "template", "fallback_mirror", "substitution_note"]
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    with (DATA_OUT / "zoo_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in zoo["rows"]:
            w.writerow([m.get(c) if not isinstance(m.get(c), bool) else ("true" if m.get(c) else "false")
                        for c in cols])


def write_readme(bundle: dict) -> None:
    zm = bundle["datasets"][4]
    n_models = sum(1 for e in zm["examples"] if e["output"] != "random-control")
    n_ctrl = sum(1 for e in zm["examples"] if e["output"] == "random-control")
    text = f"""# Safety-Screening Evidence Bundle (v1.0)

Canonical shared evidence bundle for the model-safety screening experiment.
Five datasets (exp_sel_data_out schema), all rows REAL and provenance-recorded;
zero LLM API spend; deterministic curation seed {SEED}.

## Datasets
| dataset | rows | purpose |
|---|---|---|
| probe_corpus | 24 | 12 harmful + 12 matched benign controls; nested ablation subsets p4/p8/p12 |
| behavioral_split_b1 | 50 | 40 harmful + 10 benign; refusal-rate outcome measure for the iteration-1 screen |
| reserved_split_b2 | 25 | 20 harmful + 5 benign; **RESERVED** for iteration-2 confirmation |
| interpolation_anchors | 52 | (harmful, benign) topic+length-matched pairs for the sharpness candidate |
| model_zoo_manifest | {n_models + n_ctrl} | {n_models} verified HF model rows + 1 config-only random-control row |

## Field semantics
- `input` = prompt (harmful/benign) or `huggingface.co/<id>` for zoo rows.
- `output` = "" for prompt rows; benign anchor text for anchor rows; class label
  (`base`/`instruct`/`abliterated`/`random-control`) for zoo rows.
- `metadata_fold`: probe | b1 | b2_reserved | interp_anchor (also zoo rows carry no fold).
- `metadata_polarity`: harmful | benign.
- `metadata_category`: one of violence, illegal_acts, fraud, self_harm,
  privacy_invasion, cyberattacks, disinformation.
- `metadata_pair_id`: probe_NN / b1_NN / b1c_NN / b2_NN / b2c_NN / anchor_NN.
- `metadata_subsets`: nested probe membership ("p4;p8;p12" / "p8;p12" / "p12").
- `metadata_len_*`: char/word length; `metadata_len_ratio` = benign_chars / harmful_chars
  of the pair (within [0.6, 1.4]).
- `metadata_source_dataset` / `metadata_source_row`: provenance (file#row) of every prompt.
- `metadata_review_note`: quality-gate auto-swap record (no-LLM heuristic gate).

## RESERVED SPLIT CONTRACT (iteration-1 screen)
{RESERVED_CONTRACT}
Enforcement note: iteration-1 artifacts must not load, sample, or reference rows
with `metadata_fold=b2_reserved`. This bundle ships B2 inside its own group
(`reserved_split_b2`); treat that group as off-limits until iteration 2.

## Model zoo notes
- All ids verified live against the HF Hub API on 2026-09-19 (see
  `metadata_verification_method` / `metadata_verified_at`); no weights were
  downloaded — fp16_GB is params x 2 B from observed `safetensors.total`.
- Official Qwen3-0.6B/1.7B `-Instruct` repos no longer resolve (404); slots are
  filled with verified ungated community instruct checkpoints
  (`rd211/Qwen3-0.6B-Instruct`, `rd211/Qwen3-1.7B-Instruct`) — see
  `metadata_substitution_note`. Same for mistralai/Ministral-3B-instruct-2410
  (404 -> ministral/Ministral-3b-instruct) and allenai/SmolLM2-1.7B-Instruct
  (404 -> HuggingFaceTB/SmolLM2-1.7B-Instruct).
- huihui-ai abliterated rows are gated (adult content); ungated fallback mirrors
  are recorded in `metadata_fallback_mirror`.
- Random-control row: `CONFIG_ONLY:Qwen/Qwen3-0.6B`. Weights are instantiated by
  the experiment with `torch.manual_seed(42)`; no weights are stored in this bundle.
- Attempted-slot log: see `metadata.substitution_log` in `data_out.json`.

## How iteration 2 should consume B2
Load only `reserved_split_b2.examples`, evaluate the same screen metrics
(refusal rate, activation delta, sharpness) on B2 rows, and compare against the
iteration-1 B1 estimates as confirmation evidence.
"""
    (DATA_OUT / "README.md").write_text(text, encoding="utf-8")


@logger.catch(reraise=True)
def main() -> None:
    sel = load_json(SOURCES / "selection.json")
    zoo = load_json(SOURCES / "zoo_manifest.json")
    prov = load_json(SOURCES / "provenance.json")

    datasets = [
        {"dataset": DS_PROBE, "examples": build_probe(sel)},
        {"dataset": DS_B1, "examples": build_b1(sel)},
        {"dataset": DS_B2, "examples": build_b2(sel)},
        {"dataset": DS_ANCHOR, "examples": build_anchors(sel)},
        {"dataset": DS_ZOO, "examples": build_zoo(zoo)},
    ]

    metadata = {
        "bundle_id": "safety_screen_bundle_v1",
        "description": (
            "Canonical evidence bundle for the model-safety screening experiment: "
            "prompt corpus (probe/B1/B2 splits) + interpolation anchors + verified "
            "model zoo manifest. All prompt rows are REAL, sourced from openly or "
            "permissively licensed third-party datasets; zero synthetic generation, "
            "zero LLM API spend."
        ),
        "version": "1.0",
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "curation_seed": SEED,
        "license_note": LICENSE_NOTE,
        "sources": prov.get("sources", []),
        "reserved_split_contract": RESERVED_CONTRACT,
        "contact_none": None,
        "pool_stats": {
            "harmful_pool_rows": sel.get("harmful_pool_rows"),
            "benign_pool_rows": sel.get("benign_pool_rows"),
            "quality_gate_swaps": len(sel.get("review", [])),
            "zoo_substitution_log": zoo.get("substitution_log", []),
        },
    }

    bundle = {"metadata": metadata, "datasets": datasets}
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # real pool counts from the curated pools file
    n_harm = n_ben = None
    pools_file = SOURCES / "pools.jsonl"
    if pools_file.exists():
        n_harm = n_ben = 0
        for line in pools_file.open(encoding="utf-8"):
            r = json.loads(line)
            if r["polarity"] == "harmful":
                n_harm += 1
            else:
                n_ben += 1
    metadata["pool_stats"]["harmful_pool_rows"] = n_harm
    metadata["pool_stats"]["benign_pool_rows"] = n_ben

    canonical = DATA_OUT / "data_out.json"
    canonical.write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
    full = DATA_OUT / "full_data_out.json"
    full.write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
    (DATA_OUT / "mini_data_out.json").write_text(
        json.dumps(make_mini_preview(bundle, 3), indent=1, ensure_ascii=False), encoding="utf-8")
    (DATA_OUT / "preview_data_out.json").write_text(
        json.dumps(make_mini_preview(bundle, 10), indent=1, ensure_ascii=False), encoding="utf-8")

    write_zoo_csv(zoo)
    write_readme(bundle)

    counts = {ds["dataset"]: len(ds["examples"]) for ds in datasets}
    logger.info(f"bundle written: {counts}")
    logger.info(f"files: {DATA_OUT / 'data_out.json'} (+full/mini/preview), zoo_manifest.csv, README.md")


if __name__ == "__main__":
    main()