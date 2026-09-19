# d09: sources metadata (passages extracted programmatically by build_out.py)
SOURCES_META = [
 {"index": 1, "page": "ams_blog", "url": "https://opensource.googleblog.com/2026/04/introducing-ams-activation-based-model-scanner-for-open-weight-llm-safety-verification.html",
  "title": "Introducing AMS: Activation-based model scanner for open-weight LLM safety verification (Google Open Source Blog)",
  "authors": ["Glen Messenger"], "year": 2026,
  "summary": "Primary announcement of the AMS activation scanner (Glen Messenger, Google, 2026-04-27). Documents the sigma-separation design, the 14-model-configuration validation set, per-class sigma values, and the cited 2025 study of 8000+ safety-modified repositories. Grep for '0.546|correlat|Spearman' returned 0 matches - the r = -0.546 number is NOT on this page.",
  "passages": [
   {"start": "Instruction-tuned models develop internal", "end": "linear probes on intermediate-layer hidden states."},
   {"start": "In our validation across 14 model configurations", "end": "less than 5% separation drift"}]},
 {"index": 2, "page": "ams_zenodo", "url": "https://zenodo.org/records/19501951",
  "title": "AMS: Detecting Unsafe and Tampered Language Models via Activation Analysis (Zenodo)",
  "authors": ["Glen Messenger"], "year": 2026,
  "summary": "Zenodo record v1 (2026-04-10) of the AMS paper. Abstract confirms contrastive prompt pairs + direction vector analysis, model-level verification (not prompt-level classification), validation across 14 model configurations, 3 families, 3 quantization levels, per-class sigma values, the DarkIdol counter-example, and 10-40 s GPU scans. The attached ams_paper.pdf returned HTTP 403 to every fetch attempt (Zenodo bot-block); the r = -0.546 value could not be located and is marked UNVERIFIED.",
  "passages": [
   {"start": "Safe models exhibit strong class separation (4-8", "end": "direction vector analysis, AMS performs model-level verification rather than prompt-level classification."},
   {"start": "We validate AMS across 14 model configurations", "end": "instruction-tuned, base, abliterated, uncensored)."}]},
 {"index": 3, "page": "ams_github_readme", "url": "https://github.com/GoogleCloudPlatform/activation-model-scanner/",
  "title": "GoogleCloudPlatform/activation-model-scanner (GitHub README)",
  "authors": None, "year": 2026,
  "summary": "Official AMS code repo README: tiered scanning (Tier 1 with PASS >3.5 / WARNING 2.0-3.5 / CRITICAL <2.0 sigma thresholds; Tier 2 identity verification with cosine similarity >0.7), concepts (harmful_content, injection_resistance, refusal_capability, truthfulness), AMS builds on AASE methodology, GPU-first (10-40 s on A100/L4) with slower CPU fallback, and gated vs ungated usage examples.",
  "passages": [
   {"start": "Thresholds: PASS (>3.5", "end": "CRITICAL (<2.0"},
   {"start": "built on [AASE (Activation-based AI Safety Enforcement)", "end": "Activation Fingerprinting technique."},
   {"start": "| Instruction-tuned (Llama, Gemma, Qwen) | 4.7", "end": "PASS |"}]}
]