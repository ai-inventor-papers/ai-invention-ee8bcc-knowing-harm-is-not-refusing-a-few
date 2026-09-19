# d08: numbers to reproduce
NUMBERS_TO_REPRODUCE = [
 {"number": "scanner-vs-compliance best r = -0.546 and the 14-configuration validation set",
  "status": "UNVERIFIED for the r value (paper PDF 403; absent from blog/README/Zenodo abstract); VERIFIED for the 14-config set and sigma classes (3.8-8.4 / 1.1-1.3 / 3.33 / 0.69; quant drift <5%)",
  "action": "drop the r target; report within-zoo scanner-vs-behavior rho"},
 {"number": "Basu knowledge-action: 98.2% AUROC vs 45.1% output sensitivity, 53-pp gap, 3,695 significant SAE features, 400 vignettes (144 hazards, 256 benign), Qwen2.5-7B-Instruct + Steerling-8B, arXiv 2603.18353",
  "status": "VERIFIED primary",
  "action": "cite as the knowledge-vs-behavior anchor for the knowledge_control baseline"},
 {"number": "Arditi: contrast-pair counts 128+32 train/val per class (NOT 74/54); readout at post-instruction positions (i*=-1 mostly); NO PCA; layer e.g. Llama-2 7B 14/32; orthogonalization W' = W - rhat^T*rhat*W over embedding/pos-emb/attn-out/MLP-out/biases; code repos; orthogonalized Llama-2 7B ASR 22.6 (79.9 no-system-prompt)",
  "status": "VERIFIED primary",
  "action": "implement B1 exactly; report without PCA/variance-share"},
 {"number": "flat ~0.98 probes across the zoo (hypothesis claim)",
  "status": "to-be-tested; anchors: Llorente-Saguer 0.982 +-0.003 (12 models) and LatentBiopsy >=0.937 <=0.015 (6 Qwen variants); Basu 98.2% single-model",
  "action": "test with latent_probe + knowledge_control; compare to 0.982 and 0.937 anchors"},
 {"number": "additional surfaced numbers: RAS calibrated 0-100, separates aligned/uncensored/abliterated, tracks ASR (no effect size in abstract); Hurtado abliteration audit AUROC 0.95 (z-sum) vs 0.84/0.90, balanced accuracy 0.89 FPR 0.11 missing 4/57; Refusal Before Decoding: decodability at every block + 72% search-time cut; AMS thresholds PASS >3.5 / WARNING 2.0-3.5 / CRITICAL <2.0",
  "status": "VERIFIED primary (abstract-level)",
  "action": "use as comparison/scrutiny targets where applicable"}
]