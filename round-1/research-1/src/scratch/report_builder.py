#!/usr/bin/env python3
"""Generate research_report.md from research_out.json."""
import json

out = json.load(open("research_out.json", encoding="utf-8"))
A = out["answer"]
L = []

L.append("# Research Report: Prior-Art Verification for a Few-Prompt Refusal-Action Safety Score")
L.append("")
L.append("**Date of all findings:** 2026-09-19. **Method:** web research only (arXiv, Google Open Source Blog, Zenodo, "
         "GitHub, Hugging Face API). Every supporting passage was extracted verbatim from a same-day fetched copy of the "
         "cited page (substring-verified).")
L.append("")
L.append("## 1. Verification table (7 hypothesis claims + 3 extra findings)")
L.append("")
L.append("| Claim | Status | Key numbers | Implication |")
L.append("|---|---|---|---|")
for row in A["verification_table"]:
    L.append(f"| **{row['claim_id']}** | {row['status'][:90]} | {row['numbers'][:140]} | {row['implication'][:140]} |")
L.append("")
L.append("### Detail per claim")
for row in A["verification_table"]:
    L.append(f"### {row['claim_id']} - {row['claim_text']}")
    L.append("")
    L.append(f"- **Status:** {row['status']}")
    L.append(f"- **Primary source:** {row['primary_source']}")
    L.append(f"- **URL:** {row['url']}")
    L.append(f"- **Verbatim quote (seen):** {row['quote']}")
    L.append(f"- **Numbers:** {row['numbers']}")
    L.append(f"- **Implication:** {row['implication']}")
    L.append("")

L.append("## 2. Baseline specs for the experiment")
L.append("")
for name, spec in A["baseline_specs"].items():
    L.append(f"### B-spec: {name}")
    L.append("")
    for k, v in spec.items():
        if isinstance(v, list):
            L.append(f"- **{k}:**")
            for item in v:
                L.append(f"  - {item}")
        else:
            L.append(f"- **{k}:** {v}")
    L.append("")

L.append("## 3. HuggingFace checkpoint audit (live API, 2026-09-19)")
L.append("")
L.append("| repo_id | author | params | gated | size (bytes) | chat_template | notes |")
L.append("|---|---|---|---|---|---|---|")
for c in A["hf_checkpoints"]:
    L.append(f"| {c['repo_id']} | {c['author']} | {c['param_count']} | {c['gated']} | {c['size_bytes']} | {c['chat_template']} | {c['notes']} |")
L.append("")
L.append("**Environment note:** HF_TOKEN is present in the environment (value not recorded); gated repos (meta-llama, "
         "google) additionally require license acceptance on the account, so non-gated unsloth mirrors are the safe "
         "fallback. Recommended primary zoo: Qwen3-0.6B/1.7B/4B (instruct) + -Base variants + huihui-ai/Huihui-Qwen3-4B-"
         "Instruct-2507-abliterated (fp16, 8.0GB) + mlabonne/Qwen3-4B-abliterated (author-dependence check; 16.1GB - verify "
         "file layout). Fallbacks: Llama-3.2 via unsloth mirrors + huihui-ai/Llama-3.2-3B-Instruct-abliterated / "
         "mylesgoose 1B; Qwen2.5-1.5B via Goekdeniz-Guelmez abliterated; Gemma by self-abliteration of unsloth/gemma-2-2b-it "
         "(only a GGUF community abliteration exists).")
L.append("")
L.append("## 4. Saturation search log (all dated 2026-09-19)")
L.append("")
for s in A["saturation"]["search_log"]:
    L.append(f"- **{s['engine']}** - `{s['query']}` -> {s['top_hits']}")
L.append("")
L.append("### Closest prior art")
L.append("")
for p in A["saturation"]["closest_prior_art"]:
    L.append(f"- **{p['paper']}** ({p['url']}): {p['overlap']}")
L.append("")
L.append("### Novelty verdict (2026-09-19)")
for k, v in A["saturation"]["novelty_verdict"]["sub_claims"].items():
    L.append(f"- **{k}:** {v}")
L.append(f"- **Hypothesis risk:** {A['saturation']['novelty_verdict']['hypothesis_risk']}")
L.append("")

L.append("## 5. Numbers the experiment baselines must reproduce")
L.append("")
L.append("| Number | Status | Action |")
L.append("|---|---|---|")
for n in A["numbers_to_reproduce"]:
    L.append(f"| {n['number']} | {n['status']} | {n['action']} |")
L.append("")

L.append("## 6. Risks and unverified claims")
L.append("")
L.append("- **UNVERIFIED:** AMS scanner r = -0.546 (paper PDF HTTP-403; absent from blog, README, and Zenodo abstract). "
         "The comparison target must be an in-zoo scanner-vs-behavior rho.")
L.append("- **MIS-ATTRIBUTIONS in the hypothesis:** (a) Arditi's method is difference-in-means with NO PCA, 128+32 prompts "
         "per class - not 74/54 with PCA; (b) the flat ~0.98 abliteration-insensitive probe belongs to Llorente-Saguer "
         "(0.982 +-0.003; LatentBiopsy <=0.015), not Khatri; (c) the knowledge-action 98.2/45.1 numbers belong to Basu, not "
         "Llorente-Saguer (whose real papers are 2604.18901 and 2603.27412).")
L.append("- **Not found:** Khatri reproducibility study; community-abliterated Qwen3-0.6B/1.7B and safetensors Gemma-2-2B; "
         "the exact term 'refusal-action index' as a published metric (lane not occupied).")
L.append("- **Contests to single-direction framing:** Joad et al. (EMNLP 2026) - refusal is multi-directional; "
         "Llorente-Saguer - pooling protocol changes the recovered direction by 73 degrees. The index must report "
         "hook-position and pooling sensitivity.")
L.append("- **Peer-review status:** mostly preprints (Basu, Llorente-Saguer, Jiang, RAS, Hurtado); Khatri is IEEE DSN-W "
         "2026; Joad is EMNLP 2026; Arditi is NeurIPS 2024; AMS is a Zenodo preprint + Google blog. All were verified at "
         "abstract/full-text level on the access date.")
L.append("")

L.append("## 7. Sources (24, all accessed 2026-09-19)")
for s in out["sources"]:
    L.append(f"{s['index']}. [{s['title']}]({s['url']})")
    for p in s.get("supporting_passages", []):
        L.append(f"   > {p['quote']}")
    L.append("")
L.append("## 8. Follow-up questions for the experiment (10)")
for i, q in enumerate(out["follow_up_questions"], 1):
    L.append(f"{i}. {q}")

open("research_report.md", "w", encoding="utf-8").write("\n".join(L))
print("wrote research_report.md with", len(L), "lines")