"""Candidate signals (index, sharpness, notch, spectral) + baselines
(sigma, probe AUROC, template log-prob, Arditi magnitude)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import corpus
from engine import ModelHandle, layer_grid

EPS = 1e-9


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def first_pc(X: np.ndarray) -> np.ndarray:
    """First right singular vector of row-centered X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    u = Vt[0]
    return u / (np.linalg.norm(u) + EPS)


# ---------------------------------------------------------------------------
# 3.1 Refusal-action index
# ---------------------------------------------------------------------------
def refusal_action_index(
    H: np.ndarray,          # [L, n, D]
    d: np.ndarray,          # [n] refusal intent per prompt
    harm_idx: list[int],
    ben_idx: list[int],
    op: str = "pearson",
) -> dict[str, Any]:
    """Per-layer: PCA first PC of harmful states (fit on harmful only);
    p_i = projection of (h_i - mean_harm); correlate p_i with refusal intent d_i
    over ALL prompts (primary) and harmful-only (secondary)."""
    L, n, D = H.shape
    harm = np.array(harm_idx)
    all_idx = harm_idx + ben_idx
    prof_all = np.zeros(L)
    prof_harm = np.zeros(L)
    prof_spear = np.zeros(L)
    projmag = np.zeros(L)
    us = []
    for l in range(L):
        Hh = H[l, harm]
        mean_harm = Hh.mean(axis=0)
        u = first_pc(Hh)
        us.append(u)
        p = (H[l] - mean_harm) @ u
        prof_all[l] = _pearson(p, d)
        prof_harm[l] = _pearson(p[harm], d[harm])
        prof_spear[l] = _spearman(p, d)
        projmag[l] = float(np.mean(np.abs(p[harm])))
    if op == "spearman":
        prof = prof_spear
    else:
        prof = prof_all
    l_max = int(np.argmax(np.abs(prof)))
    return {
        "profile": prof.tolist(),
        "profile_pearson_all": prof_all.tolist(),
        "profile_pearson_harmonly": prof_harm.tolist(),
        "profile_spearman": prof_spear.tolist(),
        "projmag_profile": projmag.tolist(),
        "index": float(prof[l_max]),
        "best_layer": l_max,
        "u_at_best": us[l_max],
        "profile_len": L,
    }


def index_on_subset(H: np.ndarray, d: np.ndarray, harm_idx: list[int], ben_idx: list[int]) -> dict[str, Any]:
    """Ablation: re-fit PCA + corr within the subset of A12 (A8 / A4)."""
    return refusal_action_index(H, d, harm_idx, ben_idx, op="pearson")


# ---------------------------------------------------------------------------
# 3.2 Interpolation boundary sharpness
# ---------------------------------------------------------------------------
def interpolation_sharpness(
    h: ModelHandle,
    anchors_ben: list[str],
    anchors_harm: list[str],
    H: np.ndarray,               # [L, nA12all, D] forward states (24 prompts)
    d_all: np.ndarray,           # refusal intent of the 24 A12 prompts
    harm_idx: list[int],
    ben_idx: list[int],
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """For each layer in the grid: patch decision-position state of anchor i
    with (1-a)*h_ben + a*h_harm, read first-gen logits, compute d(a), fit
    linear slope; steepness = max over layers/anchors of |beta1| / 4."""
    if alphas is None:
        alphas = [round(a * 0.1, 3) for a in range(11)]
    L = h.L
    D = h.D
    # anchor encodings
    ben_encs = [h.encode_prompt(p) for p in anchors_ben]
    har_encs = [h.encode_prompt(p) for p in anchors_harm]
    na = len(ben_encs)
    grid = layer_grid(L)
    best = {"steepness": 0.0, "layer": -1, "anchor": -1, "beta1": 0.0, "beta0": 0.0, "r2": 0.0}
    slopes: dict[int, list[float]] = {}
    per_anchor: dict[int, dict[int, list[float]]] = {}  # layer -> anchor -> d(a) list
    for lay in grid:
        per_anchor[lay] = {}
        # build 6 anchors x 11 alphas rows.
        # pre-forward hook on layers[lay] receives hidden[lay] == post-layer-(lay-1)
        # state == H[lay-1] in our convention; interpolate THAT state.
        src = lay - 1
        rows = []
        for i in range(na):
            zb = H[src, ben_idx[i]]      # post-layer-(lay-1) state of benign anchor
            zh = H[src, harm_idx[i]]     # post-layer-(lay-1) state of harmful anchor
            for a in alphas:
                z = (1 - a) * zb + a * zh
                rows.append({
                    "enc": ben_encs[i],
                    "z": torch.from_numpy(z.astype(np.float16)).to(h.model.device),
                })
        fl = h.patched_logits_at_first_gen(rows, patch_layer=lay)
        d_patch = h.refusal_intent([r["enc"] for r in rows], fl,
                                   patch_rows=[{"row": k, "z": rows[k]["z"]} for k in range(len(rows))],
                                   patch_layer=lay)
        dmat = d_patch.reshape(na, len(alphas))
        for i in range(na):
            y = dmat[i]
            A = np.vstack([np.ones(len(alphas)), np.array(alphas)]).T
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            beta1 = float(coef[1])
            ss_res = float(np.sum((y - A @ coef) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
            r2 = 1.0 - ss_res / ss_tot
            per_anchor[lay][i] = y.tolist()
            st = abs(beta1) / 4.0
            if st > best["steepness"]:
                best = {"steepness": st, "layer": lay, "anchor": i,
                        "beta1": beta1, "beta0": float(coef[0]), "r2": r2}
        slopes[lay] = [abs(float(np.polyfit(alphas, dmat[i], 1)[0])) / 4.0 for i in range(na)]
        logger.info(f"sharpness layer {lay}: max|slope|/4 = {max(slopes[lay]):.3f}")
    return {
        "steepness": best["steepness"],
        "best_layer": best["layer"],
        "best_anchor": best["anchor"],
        "best_beta1": best["beta1"],
        "best_beta0": best["beta0"],
        "best_r2": best["r2"],
        "slopes_per_layer": {str(k): v for k, v in slopes.items()},
        "d_curves": {str(k): v for k, v in per_anchor.items()},
        "grid": grid,
        "alphas": alphas,
    }


# ---------------------------------------------------------------------------
# 3.3 Layer-notch fingerprint (per-model stats; pair stats done in screening)
# ---------------------------------------------------------------------------
def notch_stats(prof: np.ndarray, L: int) -> dict[str, Any]:
    c = np.abs(np.asarray(prof, dtype=float))
    s = c.sum()
    pr = float(s * s / (np.sum(c * c) + EPS)) if s > EPS else 0.0
    l_max = int(np.argmax(c))
    peak_gap = float(abs(l_max - (L - 1) / 2.0) / L)
    return {"participation_ratio": pr, "peak_layer": l_max,
            "peak_gap": peak_gap, "peak_val": float(c[l_max])}


# ---------------------------------------------------------------------------
# 3.4 Zero-prompt weight-spectral signature (safetensors direct read)
# ---------------------------------------------------------------------------
def spectral_signature(model_id: str, L: int) -> dict[str, Any]:
    import os
    from pathlib import Path

    def _find_repo(model_id: str) -> Path | None:
        org, _, name = model_id.partition("/")
        hub = Path(os.environ.get("HF_HOME", "hf_cache")) / "hub"
        for base in (hub, Path(os.environ.get("HF_HOME", "hf_cache"))):
            d = base / f"models--{org}--{name}" / "snapshots"
            if d.is_dir():
                for s in sorted(d.iterdir()):
                    if s.is_dir() and any(s.glob("*.safetensors")):
                        return s
        return None

    snap = _find_repo(model_id)
    if snap is None:
        return {"error": "no snapshot found", "eff_rank": None, "contrast": None}
    files = sorted(snap.glob("*.safetensors"))
    if not files:
        return {"error": "no safetensors files", "eff_rank": None, "contrast": None}
    late = list(range(L - 4, L))
    down = {"eff_rank": [], "contrast": [], "norm": []}
    up = {"eff_rank": [], "contrast": []}
    import safetensors

    def _svd_stats(t: np.ndarray) -> tuple[float, float]:
        # t: float32 2D matrix -> eff_rank (exp entropy of normalized sq-singulars) + contrast.
        # Fast exact spectrum: singular values of t = sqrt(eigs of the smaller Gram
        # t @ t^T or t^T @ t).  Full SVD of 2048x5120 in float64 took ~1-2 min per
        # model; eigvalsh of the smaller Gram takes <1s with identical results.
        a, b = t.shape
        g = t.astype(np.float64)
        gram = g @ g.T if a <= b else g.T @ g
        w = np.linalg.eigvalsh(gram)
        s = np.sqrt(np.clip(w, 0.0, None))
        s = np.sort(s)[::-1]
        s = s[s > 0]
        if len(s) == 0:
            return float("nan"), float("nan")
        p = s**2 / (s**2).sum()
        ent = -np.sum(p * np.log(p + EPS))
        return float(np.exp(ent)), float(s[0] / (s.mean() + EPS))

    dw_names = {f"model.layers.{i}.mlp.down_proj.weight" for i in late}
    uw_names = {f"model.layers.{i}.mlp.up_proj.weight" for i in late}
    for f in files:
        try:
            sf = safetensors.safe_open(str(f), framework="pt", device="cpu")
        except Exception as exc:
            return {"error": f"safe_open failed: {exc}", "eff_rank": None, "contrast": None}
        keys = set(sf.keys())
        for name in keys & dw_names:
            t = sf.get_slice(name)[:].float().numpy()
            er, ct = _svd_stats(t)
            down["eff_rank"].append(er)
            down["contrast"].append(ct)
            down["norm"].append(float(np.linalg.norm(t)))
        for name in keys & uw_names:
            t = sf.get_slice(name)[:].float().numpy()
            er, ct = _svd_stats(t)
            up["eff_rank"].append(er)
            up["contrast"].append(ct)
        del sf
    if not down["eff_rank"]:
        return {"error": "no down_proj weights found", "eff_rank": None, "contrast": None}
    return {
        "eff_rank": float(np.mean(down["eff_rank"])),
        "contrast": float(np.mean(down["contrast"])),
        "min_norm": float(np.min(down["norm"])),
        "eff_rank_layers": down["eff_rank"],
        "up_proj_effrank": float(np.mean(up["eff_rank"])) if up["eff_rank"] else None,
        "up_proj_contrast": float(np.mean(up["contrast"])) if up["contrast"] else None,
    }


# ---------------------------------------------------------------------------
# 4.1 AMS-style scanner sigma
# ---------------------------------------------------------------------------
def scanner_sigma(H: np.ndarray, harm_idx: list[int], ben_idx: list[int]) -> dict[str, Any]:
    L = H.shape[0]
    harm, ben = np.array(harm_idx), np.array(ben_idx)
    prof = np.zeros(L)
    for l in range(L):
        X = H[l]
        w = first_pc(X)  # pooled first PC
        ph = X[harm] @ w
        pb = X[ben] @ w
        mu_h, mu_b = ph.mean(), pb.mean()
        var = (ph.var() + pb.var()) / 2.0
        prof[l] = float((mu_h - mu_b) / math.sqrt(var + EPS))
    l_max = int(np.argmax(np.abs(prof)))
    return {"sigma_max": float(prof[l_max]), "best_layer": l_max,
            "sigma_profile": prof.tolist()}


# ---------------------------------------------------------------------------
# 4.2 Supervised latent probe AUROC
# ---------------------------------------------------------------------------
def probe_auroc(H: np.ndarray, harm_idx: list[int], ben_idx: list[int], seed: int = 0,
                layers: list[int] | None = None) -> dict[str, Any]:
    """Logistic probe on (harmful=1, benign=0) decision-position states.

    v1 (first run) fit raw D-dim logistic on n=24 states: with D=1024..2048 and
    5-fold CV the probe is overfit noise (test AUROC bunched at 0.1..1.0 even for
    the random-weights control, which scored 0.1).  The well-implemented variant
    below fits a PCA projection INSIDE each training fold (no leakage) to
    min(20, n_train-2) components before logistic regression; this is the
    standard latent-probe recipe and is what the knowledge-action literature
    reports (~0.98 AUROC where harm is decodable).  Raw-D AUROC is still
    reported as `auroc_raw_fixed` for comparison."""
    L = H.shape[0]
    y = np.array([1] * len(harm_idx) + [0] * len(ben_idx))
    from sklearn.decomposition import PCA

    def _auroc(X: np.ndarray, use_pca: bool) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            if use_pca and len(tr) > 6:
                # k = 8 principal components of the training fold (fixed, modest
                # dimensionality; <= (n_train-2)//2 guards against the previous
                # k=1 degenerate PCA from min(20, n_unique(y)-1, n_train-2)).
                k = max(2, min(8, (len(tr) - 2) // 2))
                k = min(k, Xtr.shape[1])
                pca = PCA(n_components=k, random_state=seed)
                pca.fit(Xtr)
                Xtr = pca.transform(Xtr)
                Xte = pca.transform(Xte)
            clf = LogisticRegression(max_iter=3000)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], p))
        return float(np.mean(aucs)) if aucs else float("nan")

    fixed_layer = max(0, L - 6)
    a_fixed = _auroc(H[fixed_layer], use_pca=True)
    a_raw_fixed = _auroc(H[fixed_layer], use_pca=False)
    if layers is not None:
        au = [float("nan")] * L
        for ll in layers:
            au[ll] = _auroc(H[ll], use_pca=True)
    else:
        au = [_auroc(H[l], use_pca=True) for l in range(L)]
    au_valid = [a for a in au if a == a]  # drop nan
    best_layer = int(np.argmax(au)) if au_valid else -1
    return {
        "auroc_fixed": a_fixed,
        "auroc_raw_fixed": a_raw_fixed,
        "fixed_layer": fixed_layer,
        "auroc_max": float(np.max(au_valid)) if au_valid else float("nan"),
        "best_layer": best_layer,
        "auroc_profile": au,
    }


def first_token_responsiveness(first_logits: np.ndarray) -> dict[str, float]:
    """Diagnostic: how 'chat-responsive' is the model's first generated token?

    For models whose first token is a fixed artifact (Qwen3 base/abliterated
    emit a literal ' thinking' header), the position-0 refusal-intent contrast
    d = max_R logP - max_C logP is compressed toward a constant, so the
    refusal-action index cannot reflect their behavior.  Report the mean
    top-1 token, its probability, and the normalized softmax entropy at the
    decision position so the screen can flag unresponsive families."""
    import math
    lg = torch.from_numpy(first_logits).float()
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    p = torch.softmax(lg, dim=-1)
    top1 = p.max(dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1)
    ent_norm = ent / math.log(p.shape[-1])
    return {
        "first_tok_top1_prob": float(top1.values.mean().item()),
        "first_tok_entropy_norm": float(ent_norm.mean().item()),
    }


# ---------------------------------------------------------------------------
# 4.3 Refusal-template log-prob baseline
# ---------------------------------------------------------------------------
def template_logprob(h: ModelHandle, encs_har: list, first_logits_har: np.ndarray) -> float:
    """Mean over harmful prompts of max_c logP_R(c)/len(c) (refusal-only,
    length-normalized)."""

    def refusalscore(encs: list, fl: np.ndarray) -> np.ndarray:
        lp = h.score_prefixes(encs, fl, corpus.REFUSAL_CANDIDATES)
        out = np.zeros(len(encs))
        for i in range(len(encs)):
            best = -1e9
            for j, cand in enumerate(corpus.REFUSAL_CANDIDATES):
                lenc = max(1, len(h.cand_tokens(cand)))
                best = max(best, lp[i, j] / lenc)
            out[i] = best
        return out

    scores = refusalscore(encs_har, first_logits_har)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 4.4 Arditi refusal-direction magnitude
# ---------------------------------------------------------------------------
def arditi_magnitude(H: np.ndarray, harm_idx: list[int], ben_idx: list[int], layer: int) -> dict[str, Any]:
    """Diff-in-means refusal direction at `layer`; mean |proj| of harmful
    states onto u relative to the benign mean."""
    harm, ben = np.array(harm_idx), np.array(ben_idx)
    mu_h = H[layer, harm].mean(axis=0)
    mu_b = H[layer, ben].mean(axis=0)
    u = mu_h - mu_b
    nrm = np.linalg.norm(u)
    if nrm < EPS:
        return {"arditi_mag": 0.0, "arditi_std": 0.0, "u": None}
    u = u / nrm
    proj = np.abs((H[layer, harm] - mu_b) @ u)
    return {"arditi_mag": float(proj.mean()), "arditi_std": float(proj.std()),
            "u": u}


def cos_alignment(u1: np.ndarray, u2: np.ndarray | None) -> float:
    if u2 is None or np.linalg.norm(u1) < EPS or np.linalg.norm(u2) < EPS:
        return float("nan")
    return float(abs(float(u1 @ u2) / (np.linalg.norm(u1) * np.linalg.norm(u2) + EPS)))