# d09b: sources metadata, part 2 (index 4-9)
SOURCES_META2 = [
 {"index": 4, "page": "aase_pub", "url": "https://research.google/pubs/aase-activation-based-ai-safety-enforcement-via-lightweight-probes/",
  "title": "AASE: Activation-Based AI Safety Enforcement via Lightweight Probes (research.google/pubs)",
  "authors": ["Glen Messenger"], "year": 2026,
  "summary": "Methodological foundation of AMS. Abstract grep: paired contrastive probes; 7 models / 3 families; Gemma-2-9B AUC 1.00 with 7.2 sigma; Gemma-2-2B 4.3 sigma; AAG AUC >=0.88 on InjecAgent; APC 0.97-1.00 AUC; survives INT4; 9x faster than Llama Guard 3 (33 ms vs 306 ms) with TPR 88% vs 50%.",
  "passages": [
   {"start": "Validation across 7 models from 3 architecture families shows strong class separation", "end": "three enterprise policies."},
   {"start": "AASE is 9", "end": "88% vs 50%"}]},
 {"index": 5, "page": "arditi_abs", "url": "https://arxiv.org/abs/2406.11717",
  "title": "Refusal in Language Models Is Mediated by a Single Direction (arXiv abstract)",
  "authors": ["Andy Arditi", "Oscar Obeso", "Aaquib Syed", "Daniel Paleka", "Nina Panickssery", "Wes Gurnee", "Neel Nanda"], "year": 2024,
  "summary": "Arditi et al. 2024. Abstract confirms: refusal mediated by a one-dimensional subspace across 13 popular open-source chat models up to 72B; erasing the direction prevents refusal, adding it elicits refusal on harmless instructions; white-box jailbreak; adversarial-suffix analysis. Submitted 2024-06-17, v3 2024-10-30; NeurIPS 2024 proceedings version exists.",
  "passages": [
   {"start": "In this work, we show that refusal is mediated by a one-dimensional subspace", "end": "parameters in size."}]},
 {"index": 6, "page": "arditi_html", "url": "https://arxiv.org/html/2406.11717v3",
  "title": "Refusal in Language Models Is Mediated by a Single Direction (arXiv HTML v3 - method detail)",
  "authors": ["Andy Arditi", "Oscar Obeso", "Aaquib Syed", "Daniel Paleka", "Nina Panickssery", "Wes Gurnee", "Neel Nanda"], "year": 2024,
  "summary": "Full method text. Key verbatim facts: 128 train + 32 validation per class; direction = DIFFERENCE-IN-MEANS (grep 'PCA|principal component' = 0 matches - no PCA in the paper); post-instruction token positions (i*=-1 mostly); per-model layer choices in Table 5; directional ablation x' = x - rhat*rhat^T*x; weight orthogonalization W' = W - rhat*rhat^T*W on every matrix writing to the residual stream plus output biases; greedy decoding 512 tokens; orthogonalized Llama-2 7B ASR 22.6 (79.9 without system prompt).",
  "passages": [
   {"start": "Each dataset consists of train and validation splits of 128 and 32 samples", "end": "respectively."},
   {"start": "we can take each matrix", "end": "orthogonalize its column vectors with respect to"},
   {"start": "the matrices that write to the residual stream are: the embedding matrix", "end": "output biases, with respect to the direction"},
   {"start": "we always use greedy decoding and a maximum generation length of 512 tokens", "end": "Mazeika"}]},
 {"index": 7, "page": None, "url": "https://github.com/andyrdt/refusal_direction",
  "title": "andyrdt/refusal_direction (official code)",
  "authors": None, "year": 2024,
  "summary": "Official repo 'Code and results accompanying the paper' (443 stars per GitHub API, 2026-09-19). The canonical refusal-direction + abliteration codebase.",
  "passages": []},
 {"index": 8, "page": None, "url": "https://github.com/Sumandora/remove-refusals-with-transformers",
  "title": "Sumandora/remove-refusals-with-transformers",
  "authors": None, "year": 2024,
  "summary": "Community implementation (2202 stars, not archived, per GitHub API 2026-09-19): 'Implements harmful/harmless refusal removal using pure HF Transformers'. The widely cited 'abliteration' recipe implementation.",
  "passages": []},
 {"index": 9, "page": "khatri_abs", "url": "https://arxiv.org/abs/2609.19472",
  "title": "Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models",
  "authors": ["Alizishaan Khatri", "Chiquita Prabhu", "Omkar Neogi"], "year": 2026,
  "summary": "Khatri et al., submitted 2026-09-16, IEEE DSN-W 2026 pp. 48-52. LLaMA-3.1-8B; MLP probes (12.6M params) on activations; F1 99/83/84 on WildJailbreak, BeaverTails, AEGIS 2.0. Per-prompt harm detection; NO abliteration or 0.98 claims in the abstract (mischaracterization flagged). Reproducibility study NOT_FOUND.",
  "passages": [
   {"start": "We extract activations from LLaMA-3.1-8B", "end": "cutting latency and compute costs."}]}
]