# d04: verification rows V8, V9, V10
V8 = {
 "claim_id": "V8-EXTRA",
 "claim_text": "Flat ~0.98 probes across base/aligned/abliterated + extension of the knowledge-action gap beyond the clinical domain",
 "status": "PARTIAL: flat-probe-across-abliteration VERIFIED via Llorente-Saguer (0.982 +-0.003) and LatentBiopsy (>=0.937, <=0.015); Basu beyond clinical: only a dental-scenario title found (2601.12974, arXiv API title-level)",
 "primary_source": "arXiv 2604.18901 (Llorente-Saguer) and arXiv 2603.27412 (LatentBiopsy)",
 "url": "https://arxiv.org/abs/2604.18901",
 "quote": "\"A direction fitted from 100 labelled examples per class via Soft-AUC optimisation reaches mean effective AUROC 0.982 and TPR@1%FPR 0.797\" and \"matches its instruction-tuned counterpart within \u00b10.003 AUROC in abliterated variants from which the refusal mechanism has been removed\"",
 "numbers": "12 models / 4 families (Qwen2.5, Qwen3.5, Llama-3.2, Gemma-3) / 3 alignment variants / 0.5-1.3B plus 9B extension; LatentBiopsy: 200 safe prompts, AUROC >=0.937, abliteration gap <=0.015, sigma_theta 0.03 rad vs 0.27 rad normative",
 "implication": "The flat-probe expectation now has real anchors (0.982/0.003; 0.937/0.015), both per-prompt and Qwen-heavy; the per-model behavioral-refusal-rate prediction from few prompts remains untested. Treat 0.98-flatness as the claim to test; Basu 98.2% is the single-model anchor."
}
V9 = {
 "claim_id": "V9-EXTRA (nearest prior art: RAS)",
 "claim_text": "A per-model refusal-alignment safety score with behavioral correlation already exists (RAS / SafeVec, 2026)",
 "status": "VERIFIED primary",
 "primary_source": "arXiv 2606.25750 (Huang, Chen, Yu, Lee; NYCU / Hon Hai), 2026-06-24",
 "url": "https://arxiv.org/abs/2606.25750",
 "quote": "\"Given a safety-aligned reference model, SafeVec extracts refusal directions and scores a target model according to how strongly its hidden states align with these directions under unsafe and jailbreak prompts. The resulting metric, RAS (Refusal Alignment Score), maps representation-level refusal alignment to a calibrated 0-100 safety score.\"",
 "numbers": "calibrated 0-100 RAS; requires a reference aligned model + three LABELED prompt sets (safe / unsafe / jailbreak); separates aligned/uncensored/abliterated on Llama, Gemma, Qwen; tracks output-level attack success rate; substantially faster than judge-based evaluation",
 "implication": "RAS occupies the label-rich per-model refusal-alignment-score lane. The hypothesis's distinct slot is label-FREE (or 0-few-prompt) per-MODEL behavioral refusal-RATE prediction. The experiment must position against RAS; label-freedom is the differentiator."
}
V10 = {
 "claim_id": "V10-EXTRA (contests to single-direction framing)",
 "claim_text": "Single-direction assumptions are contested (refusal is multi-directional; directions are protocol-dependent)",
 "status": "VERIFIED primary",
 "primary_source": "Joad et al. 2602.02132 (EMNLP 2026 main track) and Llorente-Saguer 2604.18901",
 "url": "https://arxiv.org/abs/2602.02132",
 "quote": "\"refusal behaviors correspond to geometrically distinct directions in activation space\" and \"different directions primarily affect not whether the model refuses, but how it refuses.\"",
 "numbers": "Joad: refusal categories map to distinct refusal directions; Llorente-Saguer: max-pool vs last-token pooling protocols recover directions 73 degrees apart; Hurtado abliteration audit (2607.01854): z-sum AUROC 0.95 vs 0.84/0.90 single signals on a 273-checkpoint registry, balanced accuracy 0.89 (FPR 0.11), missing 4 of 57",
 "implication": "Frame the index as reading a PROTOCOL-SPECIFIC direction (first PC of harm decision states), not 'the' refusal direction; report sensitivity to pooling and hook position; cite Joad and Llorente-Saguer on protocol dependence; Hurtado's audit is an alternate per-model signal to compare against."
}