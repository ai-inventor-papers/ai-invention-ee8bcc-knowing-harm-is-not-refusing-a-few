"""PART D: calibration of probe AUROC -> measured B1 refusal rate.

Primary mapping: sklearn IsotonicRegression(out_of_bounds='clip',
increasing=True) of refusal_rate on probe_auroc_fixed, fit on the 16 in-zoo
models ONLY.  Output: the monotone curve as a table of >=4 knots (unique
sortable (auroc, predicted) pairs).

Secondary mapping: logistic on the logit scale --
    x' = logit(clip(auroc, 1e-6, 1-1e-6)),  P(refusal) = sigmoid(a + b x').

Both mappings are then APPLIED to the never-run cohort's probe AUROCs and to
the in-zoo B2 (stored) refusal rates.  Reported metrics: MAE, ECE over 10
fixed bins on [0,1] (empty bins contribute 0), Brier score, Spearman rho.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.isotonic import IsotonicRegression

EPS = 1e-12


def _clip(x: np.ndarray, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def logit(x: np.ndarray) -> np.ndarray:
    x = _clip(x)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_isotonic_knots(x: list[float], y: list[float]) -> tuple[IsotonicRegression, list[tuple[float, float]]]:
    """Fit isotonic; return (model, knots) with >=4 unique sortable pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(x, y)
    pred = iso.predict(x)
    pairs = sorted(set(zip(x.tolist(), [float(round(p, 6)) for p in pred.tolist()])))
    if len(pairs) < 4:
        # guarantee >=4 knots by appending the axis limits through the model
        lo_, hi_ = float(np.min(x)), float(np.max(x))
        extras = [(lo_, float(iso.predict([lo_])[0])), (hi_, float(iso.predict([hi_])[0]))]
        pairs = sorted(set(pairs + extras))
    return iso, pairs


def fit_logit(x: list[float], y: list[float]) -> tuple[float, float]:
    """P(refusal) = sigmoid(a + b * logit(x)); returns (a, b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y)) & (np.asarray(y) > -0.5) & (np.asarray(y) < 1.5)
    x, y = x[m], y[m]
    xp = logit(x)
    # logistic regression weight fit via scipy least squares on log-odds
    yc = np.clip(y, 1e-4, 1.0 - 1e-4)
    logodds = np.log(yc / (1.0 - yc))
    A = np.vstack([np.ones_like(xp), xp]).T
    coef, *_ = np.linalg.lstsq(A, logodds, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_logit(a: float, b: float, x: list[float]) -> np.ndarray:
    return sigmoid(a + b * logit(np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _as_float_arrays(pred, truth) -> tuple[np.ndarray, np.ndarray]:
    """Coerce list-or-array inputs into float arrays (lists cannot be indexed
    by numpy boolean masks)."""
    return np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)


def mae(pred, truth) -> float:
    pred, truth = _as_float_arrays(pred, truth)
    m = ~(np.isnan(pred) | np.isnan(truth))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[m] - truth[m])))


def brier(pred, truth) -> float:
    pred, truth = _as_float_arrays(pred, truth)
    m = ~(np.isnan(pred) | np.isnan(truth))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((pred[m] - truth[m]) ** 2))


def ece(pred, truth, n_bins: int = 10) -> float:
    """Expected calibration error over n_bins fixed bins on [0,1]; empty bins
    contribute 0 (they are weighted by zero confidence mass)."""
    pred, truth = _as_float_arrays(pred, truth)
    m = ~(np.isnan(pred) | np.isnan(truth))
    pred, truth = pred[m], truth[m]
    if len(pred) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (pred >= lo) & (pred < hi) if i < n_bins - 1 else (pred >= lo) & (pred <= hi)
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        conf = float(pred[sel].mean())
        acc = float(truth[sel].mean())
        tot += (n_sel / len(pred)) * abs(conf - acc)
    return float(tot)


def spearman(pred, truth) -> float:
    pred, truth = _as_float_arrays(pred, truth)
    m = ~(np.isnan(pred) | np.isnan(truth))
    pred, truth = pred[m], truth[m]
    if len(pred) < 4 or np.std(pred) < 1e-12 or np.std(truth) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(pred, truth).statistic)


def evaluate(pred: np.ndarray, truth: np.ndarray, name: str) -> dict[str, Any]:
    out = {
        "mae": mae(pred, truth),
        "ece": ece(pred, truth, n_bins=10),
        "brier": brier(pred, truth),
        "spearman": spearman(pred, truth),
        "n": int(np.sum(~(np.isnan(np.asarray(pred, dtype=float)) | np.isnan(np.asarray(truth, dtype=float))))),
    }
    logger.info(f"calibration eval {name}: MAE={out['mae']:.3f} ECE={out['ece']:.3f} "
                f"Brier={out['brier']:.3f} rho={out['spearman']:+.3f} n={out['n']}")
    return out


# Protocol texts (quoted verbatim in method_out.json).
CALIBRATION_RULE_VERBATIM = (
    "Fit on the 16 in-zoo models (B1 rates) ONLY: primary = "
    "IsotonicRegression(out_of_bounds='clip', increasing=True) of refusal_rate "
    "on probe_auroc_fixed, reported as a monotone knot table (>=4 knots); "
    "secondary = P(refusal) = sigmoid(a + b*logit(clip(auroc,1e-6,1-1e-6))). "
    "Apply to the never-run cohort's probe AUROCs and to the stored in-zoo B2 "
    "rates; report MAE, ECE (10 fixed bins), Brier, Spearman.  PREREGISTERED "
    "SUCCESS on the never-run cohort: rho >= 0.6 (point) AND MAE <= 0.15; "
    "calibration protocol success on BOTH (a) in-zoo B2 and (b) never-run: "
    "MAE <= 0.15 AND ECE <= 0.15 (n=7 and n=16 small-sample caveats noted)."
)

CALIBRATION_THRESHOLDS = {
    "never_run_rho_ge": 0.6,
    "mae_le": 0.15,
    "ece_le": 0.15,
    "n_bins_ece": 10,
    "isotonic_increasing": True,
    "logit_clip": [1e-6, 1 - 1e-6],
}