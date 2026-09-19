# d01: verification table rows (short file to avoid corruption)
V1 = {
 "claim_id": "V1",
 "claim_text": "AMS-style activation scanner (Messenger 2026): existence; sigma computation (contrastive prompt pairs, probe, separation in standard deviations); 14-configuration validation set; best r = -0.546 correlation with behavioral compliance",
 "status": "VERIFIED primary (existence, sigma design, 14-config validation); r = -0.546 UNVERIFIED (absent from all accessible texts; paper PDF blocked by Zenodo HTTP 403)",
 "primary_source": "Zenodo record 19501951 (Messenger 2026-04-10); Google Open Source Blog (2026-04-27); GitHub GoogleCloudPlatform/activation-model-scanner",
 "url": "https://zenodo.org/records/19501951",
 "quote": "\"Safe models exhibit strong class separation (4-8\u03c3) between harmful and benign content; models with removed or degraded safety training show collapsed separation (<2\u03c3). Using contrastive prompt pairs and direction vector analysis, AMS performs model-level verification rather than prompt-level classification.\"",
 "numbers": "14 configurations; 3 families; 3 quant levels; instruction-tuned 3.8-8.4 sigma; uncensored 1.1-1.3; abliterated 3.33 (WARNING); base 0.69; quant drift <5%; 10-40 s/GPU; thresholds PASS >3.5 / WARNING 2.0-3.5 / CRITICAL <2.0; r = -0.546 NOT FOUND (blog grep 0 matches)",
 "implication": "Drop r = -0.546 as a comparison target; use the experiment's own within-zoo scanner-vs-behavior rho instead. AMS is a per-model structure check needing labeled contrast pairs; the hypothesis's label-free few-prompt per-model refusal-rate prediction is not covered by it."
}
V2 = {
 "claim_id": "V2",
 "claim_text": "Arditi et al. 2024 'Refusal in Language Models Is Mediated by a Single Direction' (expected arXiv 2406.11717)",
 "status": "VERIFIED primary (arXiv 2406.11717 v3 2024-10-30; NeurIPS 2024 proceedings version exists)",
 "primary_source": "arXiv 2406.11717 (abstract + HTML v3 full text)",
 "url": "https://arxiv.org/abs/2406.11717",
 "quote": "\"Each dataset consists of train and validation splits of 128 and 32 samples, respectively.\" and \"we can take each matrix that writes to the residual stream, and orthogonalize its column vectors with respect to\"",
 "numbers": "13 models 1.8-72B; harmful 128+32 (AdvBench, MaliciousInstruct, TDC2023, HarmBench), harmless 128+32 (Alpaca) - NOT the guessed 74/54; direction = DIFFERENCE-IN-MEANS (grep 'PCA|principal component' = 0 matches in the full HTML; the hypothesis's 'PCA top component' is wrong); readout at post-instruction positions (mostly i*=-1); Llama-2 7B l*/L = 14/32, Llama-3 8B i*=-5 l*=12/32, Gemma 7B 14/28, Qwen 1.8B 15/24; W' = W - rhat*rhat^T*W over embedding, positional embedding, attention-out, MLP-out matrices and output biases; greedy decoding 512 tokens; orthogonalized Llama-2 7B ASR 22.6 (79.9 without system prompt)",
 "implication": "B1 baseline spec complete (diff-of-means, single-vector selection via bypass/induce/kl filters with layer < 0.8L, weight orthogonalization). Do NOT report PCA or variance share. Code: github.com/andyrdt/refusal_direction (official) and Sumandora/remove-refusals-with-transformers (community, 2202 stars)."
}