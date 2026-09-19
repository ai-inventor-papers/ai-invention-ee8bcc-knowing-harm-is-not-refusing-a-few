# d09c: sources metadata, part 3 (index 10-15)
SOURCES_META3 = [
 {"index": 10, "page": "basu_abs", "url": "https://arxiv.org/abs/2603.18353",
  "title": "Interpretability without actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations",
  "authors": ["Sanjay Basu", "Sadiq Y. Patel", "Parth Sheth", "Bhairavi Muralidharan", "Namrata Elamaran", "Aakriti Kinra", "John Morgan", "Rajaie Batniji"], "year": 2026,
  "summary": "Primary source of the 98.2/45.1/53-pp knowledge-action gap: 400 physician-adjudicated vignettes (144 hazards, 256 benign); Steerling-8B and TSV results; SAE feature steering zero effect despite 3,695 significant features; Qwen 2.5 7B Instruct + Steerling-8B; code github.com/sanjaybasu/interpretability-triage.",
  "passages": [
   {"start": "Linear probes discriminated hazardous from benign cases with 98.2% AUROC", "end": "knowledge-action gap."},
   {"start": "SAE feature steering produced zero effect despite 3,695 significant features", "end": "TSV steering at high strength"},
   {"start": "Concept bottleneck steering corrected 20% of missed hazards", "end": "random perturbation (p=0.84)."}]},
 {"index": 11, "page": "llorente_harm_abs", "url": "https://arxiv.org/abs/2604.18901",
  "title": "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams",
  "authors": ["Isaac Llorente-Saguer"], "year": 2026,
  "summary": "Llorente-Saguer 2026 (single author). 12 models / 4 families / 3 alignment variants; Soft-AUC direction from 100 labels per class; mean effective AUROC 0.982, TPR@1%FPR 0.797; abliterated variants within +-0.003 AUROC (flat-probe anchor); pooling protocols 73 degrees apart; code github.com/isaac-6/harm-directions.",
  "passages": [
   {"start": "A direction fitted from 100 labelled examples per class via Soft-AUC optimisation", "end": "has been removed"},
   {"start": "matches its instruction-tuned counterpart within", "end": "has been removed"}]},
 {"index": 12, "page": "latentbiopsy_abs", "url": "https://arxiv.org/abs/2603.27412",
  "title": "The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams (LatentBiopsy)",
  "authors": ["Isaac Llorente-Saguer"], "year": 2026,
  "summary": "LatentBiopsy: training-free harmful-prompt detector; 200 safe normative prompts; leading PC + radial deviation angle theta (Gaussian NLL anomaly score); AUROC >=0.937 (harm-vs-normative) and 1.000 (XSTest) across six Qwen2.5/Qwen3.5 triplets (base/instruct/abliterated); abliteration gap <=0.015; sigma_theta 0.03 rad vs 0.27 rad normative.",
  "passages": [
   {"start": "Given 200 safe normative prompts, LatentBiopsy computes the leading principal component", "end": "regardless of orientation."},
   {"start": "both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts", "end": "generative refusal mechanism."}]},
 {"index": 13, "page": "hurtado_abs", "url": "https://arxiv.org/abs/2607.01854",
  "title": "Has This Checkpoint Been Abliterated? A Two-Signal Audit and Its Failure Map",
  "authors": ["Gabriel Hurtado"], "year": 2026,
  "summary": "Per-model abliteration audit: reference-anchored activation refusal-gap + weight-recovery energy; z-sum AUROC 0.95 vs 0.84/0.90 single signals on a 273-checkpoint registry (Qwen, DeepSeek-distilled Qwen, Llama, Gemma); balanced accuracy 0.89 (FPR 0.11), missing 4 of 57; failure map (spoofed reference; white-box owner).",
  "passages": [
   {"start": "their z-sum separates 57 public abliterations from 37 benign fine-tunes", "end": "missing only 4 of 57."}]},
 {"index": 14, "page": "ras_abs", "url": "https://arxiv.org/abs/2606.25750",
  "title": "RAS: Measuring LLM Safety Through Refusal Alignment",
  "authors": ["Chang-Chieh Huang", "Yan-Lun Chen", "Chia-Mu Yu", "Wei-Bin Lee"], "year": 2026,
  "summary": "DIRECT NEAR-PRIOR: SafeVec white-box procedure; extracts refusal directions from a safety-aligned REFERENCE model; RAS = calibrated 0-100 safety score; separates aligned/uncensored/abliterated on Llama, Gemma, Qwen; tracks output-level attack success rate; needs three LABELED prompt sets + reference model.",
  "passages": [
   {"start": "maps representation-level refusal alignment to a calibrated 0-100 safety score", "end": "safety score"},
   {"start": "RAS separates aligned models from uncensored and abliterated variants", "end": "judge-based evaluation."}]},
 {"index": 15, "page": "ras_html", "url": "https://arxiv.org/abs/2606.25750",
  "title": "RAS method detail (arXiv HTML v1)",
  "authors": ["Chang-Chieh Huang", "Yan-Lun Chen", "Chia-Mu Yu", "Wei-Bin Lee"], "year": 2026,
  "summary": "Method detail from the HTML version: the evaluator is given three prompt sets (safe S, unafe U, jailbreq J) fomatted with model-specific chat templates; the calibration set includes aligned, uncensored, and abliterated variants.",
  "passages": [
   {"start": "The evaluator is given three prompt sets", "end": "role-playing instructions."}]}
]