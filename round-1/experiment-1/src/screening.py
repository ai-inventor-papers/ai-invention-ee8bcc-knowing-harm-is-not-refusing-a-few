"""Screening: rho vs behavioral refusal rate, percentile bootstrap,
pre-registered selection rule, leave-one-family-out, prompt-count ablation,
knowledge-action fingerprint, success criteria mapping."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RULE_VERBATIM = (
    "The survivor is the candidate with the highest bootstrap lower-5th-percentile of "
    "Spearman rho vs the measured behavioral refusal rate (percentile bootstrap over >=2000 "
    "model resamples), PROVIDED its margin over the runner-up is >=0.15 AND it beats the best "
    "baseline (scanner sigma or template log-prob) by >=0.15; tie-break by mean "
    "leave-one-family-out rho, then by prompt-count robustness; if no candidate clears the "
    "margin, report the ranking honestly as an informative null."
)

CANDIDATES = ["index12", "index8", "index4", "sharpness", "notch_pr", "spectral_effrank"]
BASELINES = ["sigma_max", "probe_auroc_fixed", "template_logprob", "arditi_mag"]

# skipped-signal -> 'skipped' if n distinct ranks < 4


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Pairwise-finite Spearman rho; nan if <4 valid samples or no variance."""
    x = np.asarray([float(v) if v is not None and v == v else np.nan for v in x])
    y = np.asarray([float(v) if v is not None and v == v else np.nan for v in y])
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def percentile_bootstrap(
    x: np.ndarray, y: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Percentile bootstrap over model resamples (with replacement)."""
    rng = np.random.default_rng(seed)
    rhos = np.zeros(n_resamples)
    n = len(x)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        rhos[r] = spearman_rho(list(x[idx]), list(y[idx]))
    rhos = rhos[~np.isnan(rhos)]
    if len(rhos) == 0:
        return {"mean": float("nan"), "p5": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(rhos)),
        "p5": float(np.percentile(rhos, 5)),
        "ci95_lo": float(np.percentile(rhos, 2.5)),
        "ci95_hi": float(np.percentile(rhos, 97.5)),
        "n": int(len(rhos)),
    }


def select_survivor(
    rho_table: dict[str, dict[str, float]],
    families: list[str],
    ids: list[str],
    xvals: dict[str, list[float]],
    y: list[float],
    seed: int = 0,
) -> dict[str, Any]:
    """Apply the pre-registered rule mechanically."""
    keys_c = [k for k in CANDIDATES if k in rho_table and rho_table[k]["p5"] == rho_table[k]["p5"]]
    keys_b = [k for k in BASELINES if k in rho_table and rho_table[k]["p5"] == rho_table[k]["p5"]]
    if not keys_c:
        return {"survivor": None, "informative_null": True, "reason": "no candidate with finite p5"}
    ranked = sorted(keys_c, key=lambda k: rho_table[k]["p5"], reverse=True)
    survivor = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin_runner = rho_table[survivor]["p5"] - (rho_table[runner_up]["p5"] if runner_up else -1.0)
    best_baseline_p5 = max((rho_table[b]["p5"] for b in keys_b), default=-1.0)
    margin_baseline = rho_table[survivor]["p5"] - best_baseline_p5

    # tie-break metrics
    def loo_mean(sig: str) -> float:
        fam_set = sorted(set(families))
        vals = []
        for f in fam_set:
            m = [i for i, fa in enumerate(families) if fa != f]
            if len(m) < 4:
                continue
            vals.append(spearman_rho([xvals[sig][i] for i in m], [y[i] for i in m]))
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    tiebreak_used = None
    if margin_runner < 0.15 or margin_baseline < 0.15:
        loo = {k: loo_mean(k) for k in keys_c}
        loo_valid = {k: v for k, v in loo.items() if v == v}
        if loo_valid:
            survivor = max(loo_valid, key=loo_valid.get)
            tiebreak_used = "leave-one-family-out rho"
            margin_runner = rho_table[survivor]["p5"] - (rho_table[runner_up]["p5"] if runner_up else -1.0)
            margin_baseline = rho_table[survivor]["p5"] - best_baseline_p5
            if margin_runner < 0.15 or margin_baseline < 0.15:
                # prompt-count robustness tiebreak
                sub = {"index12": "index8", "index8": "index4"}
                if survivor in sub and rho_table.get(sub[survivor], {}).get("p5", float("nan")) == rho_table[survivor]["p5"]:
                    tiebreak_used = "prompt-count robustness"
                    survivor = survivor  # keep; robustness reported separately
                else:
                    tiebreak_used = tiebreak_used
    ok = (margin_runner >= 0.15) and (margin_baseline >= 0.15)
    return {
        "survivor": survivor if ok else None,
        "runner_up": runner_up,
        "margin_vs_runnerup": float(margin_runner),
        "margin_vs_best_baseline": float(margin_baseline),
        "best_baseline": max(keys_b, key=lambda b: rho_table[b]["p5"]) if keys_b else None,
        "tiebreak_used": tiebreak_used,
        "informative_null": not ok,
        "reason": None if ok else "no candidate cleared the pre-registered margins -> informative null",
    }


def leave_one_family_out(
    families: list[str], xvals: dict[str, list[float]], y: list[float], signals: list[str]
) -> dict[str, Any]:
    fam_set = sorted(set(families))
    per_fam: dict[str, dict[str, float]] = {}
    for sig in signals:
        per_fam[sig] = {}
    for f in fam_set:
        out = [i for i, fa in enumerate(families) if fa != f]
        inn = [i for i, fa in enumerate(families) if fa == f]
        for sig in signals:
            rho_out = spearman_rho([xvals[sig][i] for i in out], [y[i] for i in out])
            rho_in = spearman_rho([xvals[sig][i] for i in inn], [y[i] for i in inn]) if len(inn) >= 4 else float("nan")
            per_fam[sig][f] = {"rho_excl": rho_out, "rho_within": rho_in}
    agg: dict[str, float] = {}
    for sig in signals:
        vals = [per_fam[sig][f]["rho_excl"] for f in fam_set]
        vals = [v for v in vals if v == v]
        agg[sig] = float(np.mean(vals)) if vals else float("nan")
    return {"per_family": per_fam, "mean_excl": agg}


def knowledge_action_clusters(
    auroc: list[float], index12: list[float], classes: list[str]
) -> dict[str, Any]:
    auroc = np.asarray([float(v) if v is not None and v == v else np.nan for v in auroc])
    index12 = np.asarray([float(v) if v is not None and v == v else np.nan for v in index12])
    classes = np.array(classes)
    m = ~(np.isnan(auroc) | np.isnan(index12))
    auroc, index12, classes = auroc[m], index12[m], classes[m]
    if len(auroc) < 2:
        return {"silhouette": float("nan"), "centroids_standardized": {},
                "centroid_distances": {}, "partition_verdict": "inconclusive (<2 finite rows)"}
    X = np.column_stack([StandardScaler().fit_transform(auroc.reshape(-1, 1)).ravel(),
                         StandardScaler().fit_transform(index12.reshape(-1, 1)).ravel()])
    sil = float("nan")
    if 2 <= len(set(classes)) <= len(classes) - 1:
        try:
            sil = float(silhouette_score(X, classes))
        except ValueError:
            sil = float("nan")
    centroids: dict[str, list[float]] = {}
    for c in sorted(set(classes)):
        m = X[classes == c]
        if len(m):
            centroids[c] = m.mean(axis=0).tolist()
    dists: dict[str, float] = {}
    cs = list(centroids)
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            dists[f"{cs[i]}_vs_{cs[j]}"] = float(np.linalg.norm(np.array(centroids[cs[i]]) - np.array(centroids[cs[j]])))
    base_c = centroids.get("base")
    verdict = "base=(low,low): base lacks harm knowledge AND refusal action (hypothesis claim)" if (
        base_c and base_c[0] < 0.0 and base_c[1] < 0.0
    ) else (
        "base=(high,low): base HAS harm knowledge (universal-knowledge claim) but lacks refusal action"
        if base_c and base_c[0] >= 0.0 and base_c[1] < 0.0 else "inconclusive"
    )
    return {"silhouette": sil, "centroids_standardized": centroids,
            "centroid_distances": dists, "partition_verdict": verdict}


def success_criteria(
    rho_table: dict[str, dict[str, float]], index4_rho: float,
    per_model: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Map hypothesis success criteria -> pass/fail (mechanical)."""
    out: dict[str, Any] = {}
    r12 = rho_table.get("index12", {}).get("rho", float("nan"))
    out["c1_index12_rho_ge_0.75"] = bool(r12 == r12 and r12 >= 0.75)
    out["c2_index4_rho_ge_0.6"] = bool(index4_rho == index4_rho and index4_rho >= 0.6)
    r_sig = rho_table.get("sigma_max", {}).get("rho", float("nan"))
    r_probe = rho_table.get("probe_auroc_fixed", {}).get("rho", float("nan"))
    out["c3_index_beats_sigma_and_probe"] = bool(r12 == r12 and r_sig == r_sig and r_probe == r_probe
                                                  and r12 > r_sig and r12 > r_probe)
    # c4: per pair: abliterated AUROC within noise of aligned twin; index drops to base level
    pairs = []
    for base_id, aligned_id, abli_id in _qwen_triplets(per_model):
        if aligned_id not in per_model or abli_id not in per_model:
            continue
        a_a = per_model[aligned_id]["signals"]["probe_auroc_fixed"]
        a_m = per_model[abli_id]["signals"]["probe_auroc_fixed"]
        i_a = per_model[aligned_id]["signals"]["index12"]
        i_m = per_model[abli_id]["signals"]["index12"]
        i_b = per_model[base_id]["signals"]["index12"] if base_id else float("nan")
        pairs.append({"aligned": aligned_id, "abliterated": abli_id, "base": base_id,
                      "auroc_gap": float(a_a - a_m),
                      "index_drop": float(i_a - i_m),
                      "index_base_level": float(i_b) if i_b == i_b else None})
    if pairs:
        auroc_gaps = [p["auroc_gap"] for p in pairs if p["auroc_gap"] == p["auroc_gap"]]
        drops = [p["index_drop"] for p in pairs if p["index_drop"] == p["index_drop"]]
        out["c4a_mean_auroc_gap_abs_lt_0.1"] = bool(auroc_gaps and np.mean(np.abs(auroc_gaps)) < 0.1)
        out["c4b_mean_index_drop_ge_0.2"] = bool(drops and np.mean(drops) >= 0.2)
        out["c4_n_pairs"] = len(pairs)
    else:
        out["c4a_mean_auroc_gap_abs_lt_0.1"] = False
        out["c4b_mean_index_drop_ge_0.2"] = False
        out["c4_n_pairs"] = 0
    out["pairs"] = pairs
    return out


def _qwen_triplets(per_model: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Return (base, aligned, abliterated) ids per family/SIZE present.

    With the corrected manifest the Qwen3 triplets exist in full
    (Qwen/Qwen3-{...}-Base | Qwen/Qwen3-{...} | mlabonne abliterated); Llama
    and Gemma sets contribute their own triplets when the base slot exists.
    Pairs are SIZE-MATCHED (aligned twin == abliterated twin of the same
    size, exactly the comparison the knowledge-action hypothesis makes);
    the base member is the size-matched base when one exists, else None."""
    tri = []
    for fam in sorted({v.get("meta", {}).get("family") for v in per_model.values()}):
        fam_ids = [k for k, v in per_model.items() if v.get("meta", {}).get("family") == fam]
        cls_of = {k: per_model[k]["meta"]["cls"] for k in fam_ids}
        size_of = {k: per_model[k]["meta"].get("size") for k in fam_ids}
        aligned = [k for k in fam_ids if cls_of[k] == "aligned"]
        abli = [k for k in fam_ids if cls_of[k] == "abliterated"]
        bases = {k for k in fam_ids if cls_of[k] == "base"}
        for a in aligned:
            for m in abli:
                sz = size_of[a] or size_of[m]
                if sz and size_of[m] != sz:
                    continue  # size-matched twins only
                base_id = next((k for k in bases if size_of[k] == sz), None)
                tri.append((base_id, a, m))
    return tri


def run_screening(per_model: dict[str, dict[str, Any]], seed: int = 0, n_resamples: int = 2000) -> dict[str, Any]:
    """Assemble the table and run the whole pre-registered screen."""
    ids = [k for k, v in per_model.items() if v.get("meta", {}).get("cls", "random") != "random"]
    rows = [per_model[i] for i in ids]
    y = [r["behavior"]["refusal_rate"] for r in rows]
    families = [r["meta"]["family"] for r in rows]
    classes = [r["meta"]["cls"] for r in rows]
    signals = CANDIDATES + BASELINES
    xvals: dict[str, list[float]] = {}
    for sig in signals:
        xvals[sig] = [r["signals"][sig] for r in rows]

    rho_table: dict[str, dict[str, float]] = {}
    for sig in signals:
        xv = np.asarray([float(v) if v is not None and v == v else np.nan for v in xvals[sig]])
        n_valid = int(np.sum(~np.isnan(xv) & ~np.isnan(np.asarray(y, dtype=float))))
        rho = spearman_rho(xvals[sig], y)
        boot = percentile_bootstrap(xv, np.array(y, dtype=float), n_resamples=n_resamples, seed=seed)
        rho_table[sig] = {"rho": rho, "n_valid": n_valid, **boot}
        logger.info(f"rho({sig:>20s}) = {rho:+.3f} | boot p5 = {boot['p5']:+.3f} | n={n_valid}")

    sel = select_survivor(rho_table, families, ids, xvals, y, seed=seed)
    loo = leave_one_family_out(families, xvals, y, signals)
    index4_rho = rho_table.get("index4", {}).get("rho", float("nan"))
    criteria = success_criteria(rho_table, index4_rho, per_model)
    clusters = knowledge_action_clusters(
        [r["signals"]["probe_auroc_fixed"] for r in rows],
        [r["signals"]["index12"] for r in rows],
        classes,
    )
    # family confound diagnostic (6.3): does index12 correlate with family
    # membership as strongly as with behavior?
    fam_set = sorted(set(families))
    fam_dummies = np.array([[1.0 if fa == f else 0.0 for fa in families] for f in fam_set])
    rho_index_fam = max(spearman_rho(list(fam_dummies[k]), xvals["index12"]) for k in range(fam_dummies.shape[0]))
    rho_index_beh = rho_table.get("index12", {}).get("rho", float("nan"))
    rho_beh_fam = max(spearman_rho(list(fam_dummies[k]), y) for k in range(fam_dummies.shape[0]))
    confound_flag = bool(
        rho_index_fam == rho_index_fam and rho_index_beh == rho_index_beh
        and abs(rho_index_fam) >= 0.7 * abs(rho_index_beh) and rho_index_beh != 0
    )
    fam_means = {f: {"n": sum(1 for fa in families if fa == f),
                     "mean_index12": float(np.mean([xvals["index12"][i] for i, fa in enumerate(families) if fa == f])),
                     "mean_refusal": float(np.mean([y[i] for i, fa in enumerate(families) if fa == f]))}
                  for f in fam_set}
    return {
        "table": {i: {"id": ids[i], "family": families[i], "cls": classes[i],
                      "behavior_refusal_rate": y[i],
                      **{sig: xvals[sig][i] for sig in signals}} for i in range(len(ids))},
        "rho_table": rho_table,
        "selection": {"rule_verbatim": RULE_VERBATIM, **sel},
        "loo": loo,
        "ablation_12_8_4": {k: rho_table[k]["rho"] for k in ("index12", "index8", "index4") if k in rho_table},
        "clusters": clusters,
        "success_criteria": criteria,
        "family_confound": {
            "rho_index_vs_family_max": float(rho_index_fam),
            "rho_index_vs_behavior": float(rho_index_beh) if rho_index_beh == rho_index_beh else None,
            "rho_behavior_vs_family_max": float(rho_beh_fam),
            "flag_family_confound": confound_flag,
            "family_means": fam_means,
        },
        "n_models": len(ids),
    }