"""Iteration-2 candidate signals: activation-space structure of the harm axis.

All three operate on the SAME objects as iteration 1 -- H [L, n, D] fp32
decision-state residuals from one batched forward and d [n] refusal-intent
contrast -- and reuse signals.first_pc (top right singular vector of
row-centered X).

C1  harm-subspace concentration: participation ratio of the singular spectrum
    of the row-centered harmful decision states at layer l,
        PR = (sum s)^2 / sum(s^2),   s = singular values of H[l, harm]
    PR in [1, min(n_harm, D)-1]; small = harm states live in a low-dimensional
    (structured / concept-like) subspace, large = high-entropy / isotropic.
    Reported at the probe's best layer and as the max over layers.

C2  harm-axis / probe-axis alignment: |cosine(u_harm(l), w_probe(l))| with
    u_harm(l) = first PC of the harmful states and w_probe(l) = the supervised
    logistic-probe discriminant direction mapped back to INPUT space,
        w_full = pca.components_.T @ coef_  (normalized),
    where the probe is the iteration-1 recipe (PCA(k<=8)-then-logistic) fitted
    ONCE on the FULL 24 states (no CV, seed 0).  C2 asks whether the harm axis
    the unsupervised geometry finds is the same axis a supervised probe uses.

C3  cross-layer axis stability: mean |cosine(u_harm(l-1), u_harm(l))| over the
    late half of the stack (layers ceil(L/2) .. L-1).  High = the harm axis is
    a persistent global direction rather than a per-layer artifact.

Subset ablations: C1/C3 are defined on harmful states only, so they recompute
directly on the 8-prompt (A8) and 4-prompt (A4) harmful subsets.  C2 on a
subset recomputes u_harm from the subset while keeping the FULL-set probe
direction (direction stability) AND refits the probe direction on the subset
(report both, labeled).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from signals import first_pc

EPS = 1e-9


def participation_ratio(s: np.ndarray) -> float:
    """(sum s)^2 / sum(s^2) of a singular-value spectrum."""
    s = np.asarray(s, dtype=float)
    s = s[s > EPS]
    if len(s) == 0:
        return float("nan")
    return float((s.sum() ** 2) / (s**2).sum())


def harm_subspace_pr(
    H: np.ndarray,
    harm_idx: list[int],
    probe_best_layer: int | None = None,
) -> dict[str, Any]:
    """C1: per-layer participation ratio of the harmful decision states."""
    L = H.shape[0]
    harm = np.array(harm_idx)
    profile = np.full(L, float("nan"))
    for l in range(L):
        X = H[l, harm]
        Xc = X - X.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
        profile[l] = participation_ratio(s)
    bl = probe_best_layer if probe_best_layer is not None and 0 <= probe_best_layer < L else max(0, L - 6)
    valid = [v for v in profile if v == v]
    return {
        "value_at_probe_best_layer": float(profile[bl]) if profile[bl] == profile[bl] else float("nan"),
        "at_probe_best_layer": bl,
        "max": float(np.max(valid)) if valid else float("nan"),
        "argmax_layer": int(np.nanargmax(profile)) if valid else -1,
        "profile": profile.tolist(),
    }


def probe_direction_full(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic supervised probe discriminant direction in INPUT space.

    Fits the iteration-1 probe recipe (PCA(k)-then-LogisticRegression, k =
    max(2, min(8, (n-2)//2))) on the FULL set (no CV) and maps the logistic
    weight back: w_full = pca.components_.T @ coef_, normalized.
    """
    n = X.shape[0]
    k = max(2, min(8, (n - 2) // 2))
    k = min(k, X.shape[1])
    pca = PCA(n_components=k, random_state=seed)
    Z = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Z, y)
    coef = np.asarray(clf.coef_).ravel()
    w = pca.components_.T @ coef
    nrm = np.linalg.norm(w)
    if nrm < EPS:
        return np.zeros_like(w)
    return w / nrm


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return float("nan")
    return float(abs(float(u @ v) / (nu * nv)))


def harm_probe_alignment(
    H: np.ndarray,
    harm_idx: list[int],
    ben_idx: list[int],
    probe_direction: dict[int, np.ndarray] | None = None,
    refit: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """C2: per-layer |cos(u_harm(l), w_probe(l))|.

    probe_direction: optional {layer: w_full} fitted on the FULL 24 states
    (direction-stability variant).  If refit=True (or when probe_direction is
    None) the probe direction is refit on the given subset's states instead.
    """
    L = H.shape[0]
    harm = np.array(harm_idx)
    ben = np.array(ben_idx)
    y = np.array([1] * len(harm) + [0] * len(ben))
    use_full_dir = (probe_direction is not None) and (not refit)
    wprobe: dict[int, np.ndarray] = {}
    profile = np.full(L, float("nan"))
    for l in range(L):
        u = first_pc(H[l, harm])
        if use_full_dir:
            w = probe_direction[l]
        else:
            Xl = np.vstack([H[l, harm], H[l, ben]])
            w = probe_direction_full(Xl, y, seed=seed)
        wprobe[l] = w
        profile[l] = _cos(u, w)
    valid = [v for v in profile if v == v]
    return {
        "max": float(np.max(valid)) if valid else float("nan"),
        "argmax_layer": int(np.nanargmax(profile)) if valid else -1,
        "mean": float(np.mean(valid)) if valid else float("nan"),
        "profile": profile.tolist(),
        "refit": bool(refit or not use_full_dir),
    }


def cross_layer_stability(
    H: np.ndarray,
    harm_idx: list[int],
) -> dict[str, Any]:
    """C3: mean |cos(u_harm(l-1), u_harm(l))| over the late half (ceil(L/2)..L-1)."""
    L = H.shape[0]
    harm = np.array(harm_idx)
    us: list[np.ndarray] = [first_pc(H[l, harm]) for l in range(L)]
    adj = np.full(L, float("nan"))
    for l in range(1, L):
        adj[l] = _cos(us[l - 1], us[l])
    late = list(range(int(math.ceil(L / 2.0)), L))
    late_vals = [adj[l] for l in late if adj[l] == adj[l]]
    return {
        "mean_late_half": float(np.mean(late_vals)) if late_vals else float("nan"),
        "late_half_layers": late,
        "full_adj_profile": adj.tolist(),
        "n_late": len(late_vals),
    }


# ---------------------------------------------------------------------------
# Full candidate bundle for a model's forward states
# ---------------------------------------------------------------------------
def compute_candidates(
    H: np.ndarray,
    d: np.ndarray,
    harm_idx: list[int],
    ben_idx: list[int],
    a8_harm: list[int],
    a4_harm: list[int],
    probe_auroc_profile: list[float] | None = None,
    fallback_probe_layer: int | None = None,
) -> dict[str, Any]:
    """C1/C2/C3 on A12 and the nested A8/A4 ablations, one call.

    a8_harm / a4_harm: the harmful-side subset indices into the [0..11] harm
    block (i.e. A8_IDX / A4_IDX).  The matching benign subsets are
    [12 + i for i in a8_harm] etc.  probe_auroc_profile: [L] auroc profile
    (seed-0 probe) used to pick C1's 'probe best layer'.
    """
    L = H.shape[0]
    if probe_auroc_profile is not None and any(v == v for v in probe_auroc_profile):
        bl = int(np.nanargmax([float(v) if v == v else float("-inf") for v in probe_auroc_profile]))
    else:
        bl = fallback_probe_layer if fallback_probe_layer is not None else max(0, L - 6)

    # full-set probe direction per layer (fitted on the 24 A12 states)
    y_full = np.array([1] * 12 + [0] * 12)
    wprobe_full: dict[int, np.ndarray] = {}
    for l in range(L):
        wprobe_full[l] = probe_direction_full(H[l], y_full, seed=0)

    out: dict[str, Any] = {}

    # ---- A12 (12 harmful + 12 benign)
    c1 = harm_subspace_pr(H, harm_idx, probe_best_layer=bl)
    c2 = harm_probe_alignment(H, harm_idx, ben_idx, probe_direction=wprobe_full)
    c3 = cross_layer_stability(H, harm_idx)
    out["a12"] = {
        "c1": c1,
        "c1_at_probe_best_layer": c1["value_at_probe_best_layer"],
        "c1_max": c1["max"],
        "c2": c2,
        "c2_max": c2["max"],
        "c3": c3,
        "c3_mean": c3["mean_late_half"],
        "probe_best_layer": bl,
    }

    # ---- A8 (8 harmful + 8 benign)
    h8, b8 = list(a8_harm), [12 + i for i in a8_harm]
    c1_8 = harm_subspace_pr(H, h8, probe_best_layer=bl)
    c2_8 = harm_probe_alignment(H, h8, b8, probe_direction=wprobe_full, refit=False)
    c2_8_refit = harm_probe_alignment(H, h8, b8, probe_direction=None, refit=True)
    c3_8 = cross_layer_stability(H, h8)
    out["a8"] = {
        "c1_max": c1_8["max"],
        "c1_at_probe_best_layer": c1_8["value_at_probe_best_layer"],
        "c2_max_fixed_dir": c2_8["max"],
        "c2_max_refit": c2_8_refit["max"],
        "c3_mean": c3_8["mean_late_half"],
        "n_harm": len(h8),
    }

    # ---- A4 (4 harmful + 4 benign)
    h4, b4 = list(a4_harm), [12 + i for i in a4_harm]
    c1_4 = harm_subspace_pr(H, h4, probe_best_layer=bl)
    c2_4 = harm_probe_alignment(H, h4, b4, probe_direction=wprobe_full, refit=False)
    c2_4_refit = harm_probe_alignment(H, h4, b4, probe_direction=None, refit=True)
    c3_4 = cross_layer_stability(H, h4)
    out["a4"] = {
        "c1_max": c1_4["max"],
        "c1_at_probe_best_layer": c1_4["value_at_probe_best_layer"],
        "c2_max_fixed_dir": c2_4["max"],
        "c2_max_refit": c2_4_refit["max"],
        "c3_mean": c3_4["mean_late_half"],
        "n_harm": len(h4),
    }

    out["notes"] = (
        "c1 = harm-subspace participation ratio (small = low-dim harmon subspace); "
        "c2 = |cos(harm PC, probe direction)| (fixed-dir = probe fitted on full A12; "
        "refit = probe refit on the subset); c3 = mean |cos(u(l-1), u(l))| over "
        "late half.  For the screen, c1/c2/c3 use their A12 max-over-layers values."
    )
    return out


def extract_screen_values(cands: dict[str, Any]) -> dict[str, float]:
    """The three per-model screen values (A12, max over layers)."""
    return {
        "c1": cands["a12"]["c1_max"],
        "c2": cands["a12"]["c2_max"],
        "c3": cands["a12"]["c3_mean"],
    }