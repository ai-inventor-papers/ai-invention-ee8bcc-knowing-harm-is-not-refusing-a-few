#!/usr/bin/env python3
"""Summarize method_out.json: zoo table, rho table, selection, criteria, controls.

Usage: .venv/bin/python view_results.py [path-to-method_out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else WS / "method_out.json"
    d = json.loads(path.read_text())
    md = d["metadata"]
    print("=" * 100)
    print("METHOD_OUT SUMMARY —", md.get("method_name"))
    print("rule:", md.get("rule_verbatim", "")[:120], "...")
    fr = md.get("full_report", {})
    meta = fr.get("meta", {})
    print("\n-- meta --")
    cs = meta.get("corpus_source")
    print("  corpus_source:", cs.get("kind") if isinstance(cs, dict) else cs)
    print("  compute:", meta.get("compute"))

    zoo = fr.get("zoo", [])
    print(f"\n-- zoo ({len(zoo)} models) --")
    print(f"  {'id':52s} {'family':9s} {'cls':11s} {'size':5s} L   D")
    for z in zoo:
        print(f"  {z['id']:52s} {z['family']:9s} {z['cls']:11s} {str(z.get('size')):5s} {z['L']:3d} {z['D']}")

    scr = fr.get("screening", {})
    print("\n-- screening rho table (Spearman vs behavior_refusal_rate) --")
    rt = scr.get("rho_table", {})
    for k, v in sorted(rt.items()):
        print(f"  {k:20s} rho={v.get('rho'):+.3f} boot_p5={v.get('p5'):+.3f} ci=[{v.get('ci95_lo'):+.2f},{v.get('ci95_hi'):+.2f}] n={v.get('n_valid')}")

    sel = scr.get("selection", {})
    print("\n-- selection --")
    for k in ("survivor", "runner_up", "margin_vs_runnerup", "margin_vs_best_baseline", "best_baseline", "tiebreak_used", "informative_null"):
        print(f"  {k}: {sel.get(k)}")

    print("\n-- success criteria --")
    for k, v in scr.get("success_criteria", {}).items():
        if k != "pairs":
            print(f"  {k}: {v}")

    print("\n-- clusters --")
    cl = scr.get("clusters", {})
    print("  silhouette:", cl.get("silhouette"))
    print("  partition_verdict:", cl.get("partition_verdict"))
    print("  centroids:", json.dumps(cl.get("centroids_standardized", {})))

    print("\n-- family confound --")
    fc = scr.get("family_confound", {})
    for k in ("rho_index_vs_family_max", "rho_index_vs_behavior", "rho_behavior_vs_family_max", "flag_family_confound"):
        print(f"  {k}: {fc.get(k)}")

    print("\n-- controls --")
    ctrl = fr.get("controls", {}).get("random_weights", {})
    print("  random_weights:", json.dumps({k: ctrl.get(k) for k in ("index12", "sigma_max", "probe_auroc_fixed", "sharpness", "p_index_perm", "p_auroc_perm", "pass")}))
    print("  seeds:", json.dumps(fr.get("controls", {}).get("seeds", {})))

    rows = d.get("datasets", [{}])[0].get("examples", [])
    print(f"\n-- per-model table ({len(rows)} rows) --")
    cols = ("metadata_id", "metadata_class", "metadata_refusal_rate", "metadata_index12", "metadata_index4",
            "metadata_sharpness", "metadata_notch_pr", "metadata_spectral_effrank", "metadata_sigma_max",
            "metadata_probe_auroc_fixed", "metadata_template_logprob", "metadata_arditi_mag")
    hdr = "  " + " | ".join(f"{c.replace('metadata_', '')[:22]:>22s}" for c in cols)
    print(hdr)
    for r in rows:
        if r.get("metadata_id") is None:
            print("  [verdict] " + str(r.get("output"))[:160])
            continue
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append(f"{v if v is not None else 'nan':>22}")
        print("  " + " | ".join(vals))

    print("\n-- b2_reserved present:", "b2_reserved" in fr, "| iteration2_recipe:", str(fr.get("iteration2_recipe"))[:80])
    print("-- notes count:", len(fr.get("notes_ambiguities", [])))


if __name__ == "__main__":
    main()