# Iteration-2 evaluation verdict (reserved B2 confirmation + calibration)

Pre-registered thresholds: Task A5 (tiers: CONFIRMED rho>=0.60 & p5>0.15; FULL rho>=0.5392 & p5>0.15; PARTIAL 0.4<=rho<0.5392 & p5>0; else NOT-CONFIRMED). Calibration (Task B7): PASS iff MAE<=0.15 AND ECE<=0.15 on B2, equal-frequency variant.

- B2 CONFIRMATION: **FULL** — rho(probe_auroc_fixed, B2 rate) = +0.7368, bootstrap p5 = +0.4647 (n=16, average-rank Spearman, percentile bootstrap n_boot=2000, seed 20260919); iteration-1 B1 p5 threshold = 0.5392. Also clears the >=0.60/p5>0.15 CONFIRMED bar.
- CALIBRATION (B2 holdout): **FAIL** — MAE = 0.1703 (threshold 0.15: FAIL), ECE (equal-freq) = 0.0547 (threshold 0.15: PASS); Brier (=MSE for rate outcomes) = 0.0363; Spearman rho(pred, B2) = +0.7735 (p5 = +0.5306).
- LABELER STABILITY (descriptive, no threshold): rho(B1 rate, B2 rate) = +0.9360 (p5 = +0.7970), Pearson r = +0.9187, MAD = 0.0891, max|delta| = 0.2250, n |delta|>0.15 = 5. Only rates (not per-prompt labels) were stored for B2, so agreement is measured at the rate level (n_eff = 40 vs 20 prompts).
- RANDOM-CONTROL CAVEAT CARRIED: **yes** — An AUROC of 0.77 is achievable by n=24 random weights (iteration-1 control: probe_auroc_fixed = 0.7667). Never interpret a single absolute AUROC value as evidence; only (i) rank correlation across a zoo and (ii) the calibrated B1-to-B2 mapping are claim-bearing.
- DEGENERATE MODELS: **flagged, never dropped** — lunahr/gemma-3-1b-it-abliterated; NousResearch/Llama-3.2-1B; sensitivity columns in calibration_curve.json and metrics_agg.sens_excl_*.
- MERGED CROSS-CHECK: **pending_parallel_experiment_outputs**.

Honesty note: a FAIL or NOT-CONFIRMED below any pre-registered threshold is reported as-is; thresholds are never loosened after the fact.