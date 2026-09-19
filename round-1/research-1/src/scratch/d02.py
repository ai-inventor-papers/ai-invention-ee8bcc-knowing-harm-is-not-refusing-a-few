# d02: verification rows V3, V4
V3 = {
 "claim_id": "V3",
 "claim_text": "Khatri 2026 latent safety probes (per-prompt classification, insensitive to abliteration, flat around 0.98) plus a claimed reproducibility study",
 "status": "VERIFIED primary (paper exists); MIS-CHARACTERIZED in hypothesis (no abliteration or 0.98 claims in it); reproducibility study NOT_FOUND",
 "primary_source": "arXiv 2609.19472 (Khatri, Prabhu, Neogi; IEEE DSN-W 2026 pp. 48-52)",
 "url": "https://arxiv.org/abs/2609.19472",
 "quote": "\"We extract activations from LLaMA-3.1-8B and train lightweight MLP classifier probes (12.6M parameters) to detect harmful prompts.\"",
 "numbers": "12.6M-param MLP probes on LLaMA-3.1-8B; F1 99/83/84 on WildJailbreak, BeaverTails, AEGIS 2.0; NO abliteration experiments in the abstract; the flat ~0.98-across-abliteration claim belongs to Llorente-Saguer 2604.18901 (+-0.003 AUROC) and 2603.27412/LatentBiopsy (<=0.015)",
 "implication": "Re-anchor the flat-probe expectation to Llorente-Saguer and LatentBiopsy; cite Khatri only for 'harm decodable from latent states and usable as guardrail'; do not cite a reproducibility study that was not found."
}
V4 = {
 "claim_id": "V4",
 "claim_text": "Knowledge-action gap (hypothesis attributes it to Llorente-Saguer 2026)",
 "status": "VERIFIED primary - the numbers belong to Basu et al. 2603.18353 (2026-03-18); Llorente-Saguer is REAL but his papers are 2604.18901 (supervised harm-directions) and 2603.27412 (LatentBiopsy), not the knowledge-action work",
 "primary_source": "Basu, Patel, Sheth, Muralidharan, Elamaran, Kinra, Morgan, Batniji - 'Interpretability without actionability', arXiv 2603.18353",
 "url": "https://arxiv.org/abs/2603.18353",
 "quote": "\"Linear probes discriminated hazardous from benign cases with 98.2% AUROC, yet the model's output sensitivity was only 45.1%, a 53-percentage-point knowledge-action gap.\"",
 "numbers": "400 physician-adjudicated vignettes (144 hazards, 256 benign); 98.2% AUROC / 45.1% sensitivity / 53-pp gap; Steerling-8B corrected 20% of missed hazards but disrupted 53% of correct detections (p=0.84); SAE steering zero effect despite 3,695 significant features; TSV high-strength steering corrected 24%, disrupted 6%, left 76% uncorrected; Qwen 2.5 7B Instruct + Steerling-8B; code github.com/sanjaybasu/interpretability-triage",
 "implication": "B5 knowledge-side control anchor. The 98.2/45.1 result is ONE domain (clinical triage) and ONE family (Qwen2.5); zoo-wide flatness is a hypothesis claim to test. A dental-scenario knowledge-action paper exists (2601.12974, title-level via arXiv API)."
}