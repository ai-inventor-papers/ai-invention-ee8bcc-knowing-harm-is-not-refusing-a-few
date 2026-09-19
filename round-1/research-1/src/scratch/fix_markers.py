#!/usr/bin/env python3
# Patch markers in the d09 source metadata files.
def patch(path, pairs):
    t = open(path, encoding="utf-8").read()
    for a, b in pairs:
        if a not in t:
            print("MISSING in", path, ":", a[:60])
            continue
        t = t.replace(a, b)
    open(path, "w", encoding="utf-8").write(t)
    print("patched", path)

patch("scratch/d09a.py", [
 ('{"start": "AMS is built on AASE", "end": "Activation Fingerprinting technique."}',
  '{"start": "built on [AASE (Activation-based AI Safety Enforcement)", "end": "Activation Fingerprinting technique."}')
])
patch("scratch/d09b.py", [
 ('{"start": "We show that refusal is mediated by a one-dimensional subspace", "end": "parameters in size."}',
  '{"start": "In this work, we show that refusal is mediated by a one-dimensional subspace", "end": "parameters in size."}')
])
patch("scratch/d09c.py", [
 ('{"start": "SAE feature steering produced zero effect despite 3,695 significant features", "end": "significant features."}',
  '{"start": "SAE feature steering produced zero effect despite 3,695 significant features", "end": "TSV steering at high strength"}'),
 ('{"start": "Given a safety-aligned reference model, SafeVec extracts refusal directions", "end": "calibrated 0-100 safety score."}',
  '{"start": "The resulting metric, RAS (Refusal Alignment Score)", "end": "calibrated 0-100 safety score."}'),
 ('{"start": "Across Llama, Gemma, and Qwen model families, RAS separates aligned models from uncensored and abliterated variants", "end": "judge-based evaluation."}',
  '{"start": "RAS separates aligned models from uncensored and abliterated variants", "end": "judge-based evaluation."}')
])
patch("scratch/d09d.py", [
 ('{"start": "This paper investigates refusal behavior across six LLMs from three architectural families", "end": "architectural families."}',
  '{"start": "This paper investigates refusal behavior across six LLMs from three architectural families", "end": "We challenge the assumption"}')
])
print("done")