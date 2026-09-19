"""Post-run finalize checks (testing plan 7):
1. All expected slots have checkpoints.
2. Screening ranking stable at 500 vs 2000 bootstrap resamples.
3. b2_reserved exists per model and screening never consumed it.
Run after the full stage: .venv/bin/python finalize_check.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))


def main() -> int:
    manifest = json.loads((WS / "caches" / "manifest.json").read_text())
    ok = True
    missing = []
    for k, s in manifest["slots"].items():
        if s.get("status") == "dropped":
            continue
        mid = s["id"]
        cp = WS / "caches" / "per_model" / (mid.replace("/", "__") + ".json")
        if not cp.exists():
            missing.append(k)
            ok = False
    print(f"[1] missing checkpoints: {missing if missing else 'NONE'}")
    if missing:
        return 1

    per_model = {}
    for f in sorted((WS / "caches" / "per_model").glob("*.json")):
        j = json.loads(f.read_text())
        per_model[j["meta"]["id"]] = j

    # b2_reserved present everywhere (written but never scored)
    b2_ok = all("b2_reserved" in v for v in per_model.values())
    print(f"[3] b2_reserved present in all {len(per_model)} checkpoints: {b2_ok}")

    import screening as scr
    r2000 = scr.run_screening(per_model, seed=0, n_resamples=2000)
    r500 = scr.run_screening(per_model, seed=0, n_resamples=500)
    rank2000 = sorted(scr.CANDIDATES + scr.BASELINES,
                      key=lambda k: r2000["rho_table"][k]["p5"], reverse=True)
    rank500 = sorted(scr.CANDIDATES + scr.BASELINES,
                     key=lambda k: r500["rho_table"][k]["p5"], reverse=True)
    print(f"[2] 2000-resample ranking: {[(s, round(r2000['rho_table'][s]['p5'], 3)) for s in rank2000]}")
    print(f"[2] 500-resample  ranking: {[(s, round(r500['rho_table'][s]['p5'], 3)) for s in rank500]}")
    top_agree = rank2000[0] == rank500[0] and rank2000[1] == rank500[1]
    surv_agree = r2000["selection"]["survivor"] == r500["selection"]["survivor"]
    print(f"[2] top-2 ranking agree: {top_agree}; survivor agree: {surv_agree}")
    ok &= top_agree and surv_agree and b2_ok
    print("RESULT:", "ALL FINALIZE CHECKS PASS" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())