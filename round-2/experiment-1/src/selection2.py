"""PART C: iteration-2 screening -- new candidates vs the harm-knowledge probe.

Applies the pre-registered iteration-2 selection rule mechanically (no reading
of the ranking before applying it).  Percentile bootstrap (2000 model
resamples) over the 16 in-zoo models, seeds 0/1/2; family-confound diagnostic
and leave-one-family-out transfer for EVERY candidate.

The champion is the harm-knowledge probe (probe_auroc_fixed) UNLESS a NEW
candidate c1/c2/c3 satisfies all four conditions of the rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

from screening import percentile_bootstrap, spearman_rho

NEW_RULE_VERBATIM = (
    "for each candidate compute Spearman rho vs B1 refusal over the 16 in-zoo "
    "models with a percentile bootstrap (2000 model resamples) giving p5; a NEW "
    "candidate is declared the winner iff (i) p5 >= 0.50, (ii) its p5 exceeds "
    "the runner-up's p5 by >= 0.10, (iii) its p5 exceeds the probe's p5 by "
    ">= 0.10 (it must strictly beat the champion to displace it), and (iv) the "
    "family-confound diagnostic max_f |rho(candidate, is_family_f)| < "
    "rho(candidate, behavior); otherwise the harm-knowledge probe remains the "
    "champion"
)

# Order: NEW candidates first, then the reference signals (probe champion
# included), then iteration-1 baselines.
CANDIDATES2 = [
    "c1", "c2", "c3",
    "probe", "index12", "index8", "index4",
    "sigma_max", "template_logprob", "arditi_mag",
    "sharpness", "notch_pr", "spectral_effrank",
]
NEW_CANDIDATES = ("c1", "c2", "c3")
CHAMPION_NAME = "probe"

RULE_THRESHOLDS = {
    "p5_min": 0.50,
    "margin_vs_runnerup": 0.10,
    "margin_vs_champion": 0.10,
    "n_resamples": 2000,
    "seeds": [0, 1, 2],
    "family_confound_lt_behavior": True,
}


def candidate_rho_table(
    xvals: dict[str, list[float]],
    y: list[float],
    n_resamples: int = 2000,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, dict[str, Any]]:
    """Per candidate: point rho + percentile bootstrap p5/CI for each seed."""
    yv = np.asarray(y, dtype=float)
    table: dict[str, dict[str, Any]] = {}
    for cand in CANDIDATES2:
        x = np.asarray([float(v) if v is not None and v == v else np.nan for v in xvals[cand]])
        n_valid = int(np.sum(~(np.isnan(x) | np.isnan(yv))))
        rho = spearman_rho(list(x), list(y))
        per_seed = {}
        for s in seeds:
            b = percentile_bootstrap(x, yv, n_resamples=n_resamples, seed=s)
            per_seed[s] = {"p5": b["p5"], "mean": b["mean"],
                           "ci95_lo": b["ci95_lo"], "ci95_hi": b["ci95_hi"], "n": b["n"]}
        p5s = [per_seed[s]["p5"] for s in seeds if per_seed[s]["p5"] == per_seed[s]["p5"]]
        p5_mean = float(np.mean(p5s)) if p5s else float("nan")
        table[cand] = {
            "candidate": cand,
            "rho": rho,
            "n_valid": n_valid,
            "p5_mean": p5_mean,
            "p5_per_seed": per_seed,
            "ci95_mean": {s: per_seed[s]["ci95_lo"] for s in seeds},
        }
        logger.info(f"[sel2] rho({cand:>18s}) = {rho:+.3f} | p5(mean over seeds) = {p5_mean:+.3f} | n={n_valid}")
    return table


def family_confound_all(
    xvals: dict[str, list[float]],
    y: list[float],
    families: list[str],
) -> dict[str, Any]:
    """For EVERY candidate: max_f |rho(cand, one-hot family_f)| vs rho(cand, behavior)."""
    fam_set = sorted(set(families))
    rows: dict[str, dict[str, Any]] = {}
    for cand in CANDIDATES2:
        x = xvals[cand]
        rho_beh = spearman_rho(x, y)
        per_fam = {}
        for f in fam_set:
            onehot = [1.0 if fa == f else 0.0 for fa in families]
            per_fam[f] = spearman_rho(x, onehot)
        max_abs = max((abs(v) for v in per_fam.values() if v == v), default=float("nan"))
        confound_lt = bool(
            max_abs == max_abs and rho_beh == rho_beh and abs(max_abs) < abs(rho_beh)
        )
        rows[cand] = {
            "rho_behavior": rho_beh,
            "max_f_abs_rho_family": max_abs,
            "per_family": per_fam,
            "confound_lt_behavior": confound_lt,
        }
    max_over_cands = max((rows[c].get("max_f_abs_rho_family") or float("nan") for c in CANDIDATES2),
                         default=float("nan"))
    return {"per_candidate": rows, "max_new_candidate_confound": max_over_cands}


def leave_one_family_out_all(
    xvals: dict[str, list[float]],
    y: list[float],
    families: list[str],
) -> dict[str, Any]:
    """LOO transfer: for every candidate, mean rho over families when that
    family is EXCLUDED."""
    fam_set = sorted(set(families))
    per_cand: dict[str, dict[str, Any]] = {}
    for cand in CANDIDATES2:
        x = xvals[cand]
        excl = {}
        for f in fam_set:
            m = [i for i, fa in enumerate(families) if fa != f]
            excl[f] = spearman_rho([x[i] for i in m], [y[i] for i in m])
        vals = [v for v in excl.values() if v == v]
        per_cand[cand] = {"per_family_excl": excl,
                          "mean_excl": float(np.mean(vals)) if vals else float("nan")}
    return {"per_candidate": per_cand}


def select_champion(
    rho_table: dict[str, dict[str, Any]],
    xvals: dict[str, list[float]],
    y: list[float],
    families: list[str],
) -> dict[str, Any]:
    """Apply the pre-registered rule MECHANICALLY (no inspection of values)."""
    cond: dict[str, Any] = {}
    ranked = sorted(CANDIDATES2, key=lambda c: rho_table[c]["p5_mean"], reverse=True)
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    p5_top = rho_table[top]["p5_mean"]
    p5_runner = rho_table[runner_up]["p5_mean"] if runner_up else float("nan")
    p5_probe = rho_table[CHAMPION_NAME]["p5_mean"]
    cond["i_p5_ge_0.50"] = bool(p5_top == p5_top and p5_top >= RULE_THRESHOLDS["p5_min"])
    cond["ii_margin_vs_runnerup_ge_0.10"] = bool(
        p5_top == p5_top and p5_runner == p5_runner
        and (p5_top - p5_runner) >= RULE_THRESHOLDS["margin_vs_runnerup"]
    )
    cond["iii_margin_vs_probe_ge_0.10"] = bool(
        p5_top == p5_top and p5_probe == p5_probe
        and (p5_top - p5_probe) >= RULE_THRESHOLDS["margin_vs_champion"]
    )
    fam = family_confound_all(xvals, y, families)["per_candidate"]
    cond["iv_family_confound_lt_behavior"] = bool(fam[top]["confound_lt_behavior"])

    is_new = top in NEW_CANDIDATES
    all_four = all(cond.values())
    champion = top if (is_new and all_four) else CHAMPION_NAME
    reason = (
        f"top p5 candidate = {top} ({'NEW' if is_new else 'reference/baseline'}); "
        f"rule conditions (i)..(iv) = {[int(cond[k]) for k in ('i_p5_ge_0.50', 'ii_margin_vs_runnerup_ge_0.10', 'iii_margin_vs_probe_ge_0.10', 'iv_family_confound_lt_behavior')]}; "
        + (f"NEW candidate {top} wins" if (is_new and all_four)
           else f"harm-knowledge probe remains champion ({CHAMPION_NAME})")
    )
    return {
        "champion": champion,
        "top_p5_candidate": top,
        "is_new_winner": bool(is_new and all_four),
        "runner_up": runner_up,
        "p5_top": p5_top,
        "p5_runner_up": p5_runner,
        "p5_probe": p5_probe,
        "margin_vs_runnerup": float(p5_top - p5_runner) if p5_top == p5_top and p5_runner == p5_runner else None,
        "margin_vs_probe": float(p5_top - p5_probe) if p5_top == p5_top and p5_probe == p5_probe else None,
        "conditions": cond,
        "family_confound_of_top": fam[top],
        "reason": reason,
        "rule_verbatim": NEW_RULE_VERBATIM,
        "thresholds": RULE_THRESHOLDS,
    }


def evaluate_champion_on_cohort(
    champion: str,
    champ_values_inzoo: list[float],
    champ_values_cohort: list[float],
    refusal_cohort: list[float],
    champ_values_b2: list[float],
    b2_rates: list[float],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Champion on the NEVER-RUN cohort (n=7, honest wide bootstrap) and vs the
    stored in-zoo B2 refusal rates (check (a): rho >= 0.6 point, p5 >= 0.3)."""
    rho_cohort = spearman_rho(champ_values_cohort, refusal_cohort)
    b = percentile_bootstrap(
        np.asarray(champ_values_cohort, dtype=float),
        np.asarray(refusal_cohort, dtype=float),
        n_resamples=n_resamples, seed=seed,
    )
    bz = None
    if champ_values_b2 and b2_rates:
        bz = {
            "rho": spearman_rho(champ_values_b2, b2_rates),
            "bootstrap": percentile_bootstrap(
                np.asarray(champ_values_b2, dtype=float),
                np.asarray(b2_rates, dtype=float),
                n_resamples=n_resamples, seed=seed,
            ),
            "pass_rho_ge_0.6": bool(
                spearman_rho(champ_values_b2, b2_rates) == spearman_rho(champ_values_b2, b2_rates)
                and spearman_rho(champ_values_b2, b2_rates) >= 0.6
            ),
            "pass_p5_ge_0.3": bool(
                percentile_bootstrap(
                    np.asarray(champ_values_b2, dtype=float),
                    np.asarray(b2_rates, dtype=float),
                    n_resamples=n_resamples, seed=seed,
                )["p5"] >= 0.3
            ),
        }
        if bz["rho"] != bz["rho"]:
            bz = None
    return {
        "champion": champion,
        "never_run": {
            "n": len(refusal_cohort),
            "rho": rho_cohort,
            "bootstrap": b,
            "small_sample_caveat": "n=7: any bootstrap interval is wide; report is honest + informative",
        },
        "inzoo_b2": bz,
        "note_b2": None if bz else "stored in-zoo B2 rates unavailable -> check (a) deferred to the parallel EVALUATION artifact",
    }