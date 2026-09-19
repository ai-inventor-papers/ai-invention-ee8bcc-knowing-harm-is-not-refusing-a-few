# d07: saturation data
SATURATION = {
 "search_log": [
  {"date": "2026-09-19", "engine": "scholarly (openalex)", "query": "activation scanner open-weight model safety sigma", "top_hits": "no relevant hits"},
  {"date": "2026-09-19", "engine": "general (exa)", "query": "activation-based verification open-weight models scanner safety", "top_hits": "Google Open Source Blog AMS announcement; GitHub GoogleCloudPlatform/activation-model-scanner; Zenodo AMS paper"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"activation scanner\"", "top_hits": "no relevant hits (AMS is not on arXiv)"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"refusal direction\" AND all:\"score\"", "top_hits": "2606.25750 RAS; 2603.27412 LatentBiopsy; 2605.17413 Ablating Safety; 2608.18093 Abliteration Mitigation via Refusal Aliases (total 8)"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"compliance\" AND all:\"activation\" AND all:\"refusal\"", "top_hits": "2607.14147 Prefill Jailbreak; 2605.26772 Beyond a Single Direction; 2602.02132 There Is More to Refusal; 2502.09755 Jailbreak Initializations as Extractors of Compliance Directions (total 31)"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"refusal rate\" AND all:\"activation\"", "top_hits": "2510.08646 Activation Energy (over-refusal); 2606.26161 Refusal Lives Downstream of Persona; 2601.08489 Surgical Refusal Ablation; 2609.05794 Bait-and-Recover (total 23)"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"knowledge-action gap\"", "top_hits": "2601.12974 Bridging the Knowledge-Action Gap (dental clinical); 2603.18353 Basu; 2601.07972 Knowing But Not Doing (total 6)"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"refusal\" AND all:\"abliteration\"", "top_hits": "2607.17427 Abliteration Is Not a Scalpel; 2603.22061 Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration; 2606.05396 Willing but Unable; 2512.13655 Comparative Analysis of Abliteration Methods (total 24)"},
  {"date": "2026-09-19", "engine": "arXiv API (all:)", "query": "all:\"label-free\" AND all:\"safety\" AND all:\"language model\"", "top_hits": "2608.17202 Fool's Gold; 2606.15980 Do Activation Monitors Survive Model Updates; 2605.00326 (total 8)"},
  {"date": "2026-09-19", "engine": "general (exa)", "query": "few prompts safety score refusal activation metric", "top_hits": "RAS pdf (2606.25750); Refusal Before Decoding (2605.28553)"},
  {"date": "2026-09-19", "engine": "general (exa)", "query": "refusal-action index OR refusal action index language model", "top_hits": "no exact-phrase match; only the Arditi NeurIPS proceedings pdf -> the exact term 'refusal-action index' is not occupied as a named metric"},
  {"date": "2026-09-19", "engine": "general (marginalia)", "query": "predicting refusal rate from activations model score", "top_hits": "no direct match; adjacent: Meta Muse Spark safety report (refusal rates), transformer-circuits J-lens"},
  {"date": "2026-09-19", "engine": "general (exa)", "query": "behavior-calibrated activation metric safety open-weight", "top_hits": "no direct match (only 'Behaviorally Calibrated RL' for hallucination, 2512.19920)"}
 ],
 "closest_prior_art": [
  {"paper": "RAS / SafeVec (Huang, Chen, Yu, Lee, 2606.25750)", "url": "https://arxiv.org/abs/2606.25750",
   "overlap": "per-model refusal-alignment safety SCORE (calibrated 0-100) with behavioral correlation across Llama/Gemma/Qwen; needs a reference aligned model + three LABELED prompt sets (safe/unsafe/jailbreak) - not label-free, not few-prompt; the hypothesis's distinct slot is label-free few-prompt per-model refusal-RATE prediction"},
  {"paper": "LatentBiopsy (Llorente-Saguer, 2603.27412)", "url": "https://arxiv.org/abs/2603.27412",
   "overlap": "training-free per-prompt harmful-prompt detection from 200 SAFE prompts (leading PC + angular deviation, Gaussian NLL anomaly score); flat across abliteration (<=0.015); per-PROMPT detector, no per-model behavioral refusal-rate prediction, no action-side readout, Qwen-only families"},
  {"paper": "AMS (Messenger, 2026)", "url": "https://zenodo.org/records/19501951",
   "overlap": "per-model sigma separation score as safety-structure check; 14-config validation; needs labeled contrast prompt pairs + supervised probes; no published behavior correlation (r = -0.546 unverified)"},
  {"paper": "Abliterated-checkpoint audit (Hurtado, 2607.01854)", "url": "https://arxiv.org/abs/2607.01854",
   "overlap": "per-model abliteration DETECTION via two internal signals (reference-anchored activation refusal-gap + weight-recovery energy), AUROC 0.95 on 273 checkpoints; abliteration-specific triage; needs an attested reference and base weights; no refusal-rate prediction"},
  {"paper": "Refusal Before Decoding (Collu et al., 2605.28553)", "url": "https://arxiv.org/abs/2605.28553",
   "overlap": "refusal linearly decodable from intermediate activations at every block before the final layer (readout-side evidence); used for per-prompt jailbreak search fitness (Mechanistic AutoDAN), not a per-model safety metric"},
  {"paper": "Harmful intent supervised directions (Llorente-Saguer, 2604.18901)", "url": "https://arxiv.org/abs/2604.18901",
   "overlap": "supervised harm-direction probe AUROC 0.982, flat +-0.003 across abliteration, 12 models / 4 families / 3 alignment variants; per-prompt classification with 100 labels/class; no per-model behavior prediction; no label-free protocol"},
  {"paper": "There Is More to Refusal than a Single Direction (Joad et al., 2602.02132)", "url": "https://arxiv.org/abs/2602.02132",
   "overlap": "EMNLP-2026 counterpoint: refusal is multi-directional; steering any direction changes HOW the model refuses, not WHETHER (shared one-dimensional control knob); risk to the index's first-PC framing - cite and address with pooling sensitivity"}
 ],
 "novelty_verdict": {
  "date": "2026-09-19",
  "sub_claims": {
   "label-free few-prompt predictor of behavioral refusal rate": "THIN/OPEN - no paper found that predicts a per-MODEL behavioral refusal RATE from 0-few prompts without harmful labels. Nearest: RAS (labeled + reference model), LatentBiopsy (label-free but per-prompt detection). Dense competitor activity since Mar-Jun 2026; open but narrow; must be positioned against both.",
   "action-side readout metric (refusal-vs-compliance logits at first generated position)": "OPEN as a named metric - the exact phrase 'refusal action index' is not occupied. The readout mechanism is not novel (Tuned Lens 2303.08112; Refusal Before Decoding 2605.28553); the novelty is the per-model behavior-calibrated scoring use.",
   "knowledge-action tri-partition of base/aligned/abliterated": "OCCUPIED as phenomenology - Llorente-Saguer (+-0.003) and LatentBiopsy (<=0.015) already report probes persisting across abliteration; Basu 98.2/45.1 is the metric anchor; 'Willing but Unable' (2606.05396) and 'Abliteration Is Not a Scalpel' (2607.17427) cover capability/refusal separation. What remains open: using the tri-partition to PREDICT refusal rates per model (an index), not merely describing probes.",
   "abliteration as a synthetic knowledge-action gap": "OCCUPIED as a phenomenon (probes flat at 0.982 / 0.937 while behavior collapses, per Llorente-Saguer / LatentBiopsy). Re-framing it as a calibration/validation device for a new metric is legitimate, but the phenomenon itself is documented.",
   "whole claim (few-prompt label-free refusal-action index beating well-implemented baselines)": "OPEN but dense - requires the ICML-2026-MI-workshop bar-ii standard (clear practical benefit over well-implemented baselines); comparisons must include RAS, LatentBiopsy, AMS, probes, template log-prob; the r = -0.546 target is UNVERIFIED and must be replaced with in-zoo rho."
  },
  "hypothesis_risk": "HIGH for the 'beating a scanner at r = -0.546' framing (number unverified; RAS/LatentBiopsy are close); MODERATE overall - the label-free, few-prompt, per-model behavioral-predictor slot is genuinely thin as of 2026-09-19, but the window is closing fast and the knowledge-action tri-partition is already documented by Llorente-Saguer."
 }
}