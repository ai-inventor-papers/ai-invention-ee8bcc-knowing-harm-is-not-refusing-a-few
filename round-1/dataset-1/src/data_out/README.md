# Safety-Screening Evidence Bundle (v1.0)

Canonical shared evidence bundle for the model-safety screening experiment.
Five datasets (exp_sel_data_out schema), all rows REAL and provenance-recorded;
zero LLM API spend; deterministic curation seed 20260919.

## Datasets
| dataset | rows | purpose |
|---|---|---|
| probe_corpus | 24 | 12 harmful + 12 matched benign controls; nested ablation subsets p4/p8/p12 |
| behavioral_split_b1 | 50 | 40 harmful + 10 benign; refusal-rate outcome measure for the iteration-1 screen |
| reserved_split_b2 | 25 | 20 harmful + 5 benign; **RESERVED** for iteration-2 confirmation |
| interpolation_anchors | 52 | (harmful, benign) topic+length-matched pairs for the sharpness candidate |
| model_zoo_manifest | 23 | 22 verified HF model rows + 1 config-only random-control row |

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
B2 (metadata_fold='b2_reserved', metadata_reserved='true') is the iteration-2 confirmation evidence. Iteration-1 screen artifacts MUST NOT load, sample, or reference rows with metadata_fold='b2_reserved'. Only iteration-2 confirmation artifacts may consume them.
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
