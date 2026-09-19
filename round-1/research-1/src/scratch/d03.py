# d03: verification rows V5, V6, V7
V5 = {
 "claim_id": "V5",
 "claim_text": "Hildebrant 2025 nonlinear perspectives (six models, descriptive, no scalar metric, no abliterated class)",
 "status": "VERIFIED primary",
 "primary_source": "arXiv 2501.08145 (Hildebrandt, Maier, Krauss, Schilling - FAU Erlangen), 2025-01-14",
 "url": "https://arxiv.org/abs/2501.08145",
 "quote": "\"This paper investigates refusal behavior across six LLMs from three architectural families.\" and \"Our results reveal that refusal mechanisms exhibit nonlinear, multidimensional characteristics that vary by model architecture and layer.\"",
 "numbers": "six LLMs, three families; PCA, t-SNE, UMAP; no scalar score, no abliterated class - matches the hypothesis characterization",
 "implication": "Motivates the index's per-layer profile and max-over-layers design; descriptive, not a quantitative baseline."
}
V6 = {
 "claim_id": "V6",
 "claim_text": "Jiang 2026 'When behavioral safety evaluation fails' / dissociated models",
 "status": "VERIFIED primary (exact title: 'When Behavioral Safety Evaluation Fails: A Representation-Level Perspective')",
 "primary_source": "arXiv 2606.08044 (Jiang, Gjolbye, Zhang, Koyejo); v1 2026-06-06, v2 2026-08-04",
 "url": "https://arxiv.org/abs/2606.08044",
 "quote": "\"Every static audit we run gives the dissociated model the same verdict as its base, since its refusals match the base, jailbreaks show no consistent signature, and a strong fixed probe on clean activations cannot tell it from the base.\"",
 "numbers": "dissociated models from Gemma 2 2B, Llama 3.2 3B, Qwen 2.5 3B; LVS 2.5-3.1x higher at targeted mid layer; bounded latent attack 54-86% compliance vs 3-48% bases; random perturbations <=12%; harmful fine-tuning reaches high compliance in 5 gradient steps vs 10-25 for bases",
 "implication": "Scopes out adversarially dissociated models: static audits are evadable even with fixed probes; the index's action-side readout (refusal-vs-compliance logits at the first generated position) is complementary and should be tested on dissociated-style constructions."
}
V7 = {
 "claim_id": "V7",
 "claim_text": "Refusal tokens (2024) / template-level per-model refusal-rate control",
 "status": "VERIFIED primary (paper exists); it serves a DIFFERENT purpose (training-time calibration tokens), not per-model refusal detection",
 "primary_source": "Jain et al., 'Refusal Tokens: A Simple Way to Calibrate Refusals in Large Language Models', arXiv 2412.06748 (Dec 2024)",
 "url": "https://arxiv.org/abs/2412.06748",
 "quote": "\"we propose refusal tokens, one such token for each refusal category or a single refusal token, which are prepended to the model's responses during training\"",
 "numbers": "none required; repo neelsjain/refusal-tokens",
 "implication": "The experiment's refusal-template log-prob baseline is its own design (fixed phrase set + first-generated-position log-prob); cite 2412.06748 as mechanism evidence; the Zou-2023-style keyword list source must be verified during the experiment."
}