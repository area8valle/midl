from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _midrank(scores.astype(np.float64))
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_delong(predictions: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    m = n_pos
    n = predictions.shape[1] - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    k = predictions.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(predictions[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    order = np.argsort(-labels)
    n_pos = int(labels.sum())
    stacked = np.vstack([preds_a[order], preds_b[order]]).astype(np.float64)
    aucs, cov = _fast_delong(stacked, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def bootstrap_ci(
    metric: Metric,
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        values[b] = metric(scores[idx], labels[idx])
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.shape[0], b.shape[0]
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    best_j = -1.0
    best = (0.5, 0.0, 0.0)
    for t in np.unique(scores):
        pred = scores >= t
        tp = float(np.sum(pred & (labels == 1)))
        fn = float(np.sum(~pred & (labels == 1)))
        tn = float(np.sum(~pred & (labels == 0)))
        fp = float(np.sum(pred & (labels == 0)))
        sens = tp / max(tp + fn, 1.0)
        spec = tn / max(tn + fp, 1.0)
        if sens + spec - 1.0 > best_j:
            best_j = sens + spec - 1.0
            best = (float(t), sens, spec)
    return best


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def calibration_slope(logits: np.ndarray, labels: np.ndarray, iters: int = 50) -> float:
    a, b = 0.0, 1.0
    x = logits.astype(np.float64)
    y = labels.astype(np.float64)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a + b * x)))
        w = p * (1 - p) + 1e-9
        r = y - p
        g0, g1 = r.sum(), (r * x).sum()
        h00, h01, h11 = w.sum(), (w * x).sum(), (w * x * x).sum()
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        a += (h11 * g0 - h01 * g1) / det
        b += (-h01 * g0 + h00 * g1) / det
    return float(b)


def continuous_nri(new: np.ndarray, ref: np.ndarray, labels: np.ndarray) -> float:
    ev = labels == 1
    nonev = labels == 0
    up_ev = float(np.mean(new[ev] > ref[ev]))
    down_ev = float(np.mean(new[ev] < ref[ev]))
    up_non = float(np.mean(new[nonev] > ref[nonev]))
    down_non = float(np.mean(new[nonev] < ref[nonev]))
    return (up_ev - down_ev) + (down_non - up_non)


def integrated_discrimination(new: np.ndarray, ref: np.ndarray, labels: np.ndarray) -> float:
    ev = labels == 1
    nonev = labels == 0
    return float((new[ev].mean() - ref[ev].mean()) - (new[nonev].mean() - ref[nonev].mean()))


def harrell_cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    num, den = 0.0, 0.0
    n = time.shape[0]
    for i in range(n):
        if event[i] != 1:
            continue
        later = time > time[i]
        den += float(later.sum())
        num += float(np.sum(later & (risk[i] > risk)))
        num += 0.5 * float(np.sum(later & (risk[i] == risk)))
    return num / den if den > 0 else 0.5


def auc_prc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.clip(tp + fp, 1e-9, None)
    recall = tp / max(float(y.sum()), 1.0)
    ap, prev = 0.0, 0.0
    for p, r in zip(precision, recall, strict=True):
        ap += (r - prev) * p
        prev = r
    return float(ap)


def net_benefit(
    probs: np.ndarray, labels: np.ndarray, thresholds: tuple[float, ...]
) -> dict[float, float]:
    n = labels.shape[0]
    out: dict[float, float] = {}
    for pt in thresholds:
        pred = probs >= pt
        tp = float(np.sum(pred & (labels == 1)))
        fp = float(np.sum(pred & (labels == 0)))
        out[pt] = tp / n - (fp / n) * (pt / (1 - pt))
    return out


def evaluate_progression(scores: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-scores))
    lo, hi = bootstrap_ci(auroc, scores, labels, seed=seed)
    thr, sens, spec = youden_threshold(scores, labels)
    return {
        "auroc": auroc(scores, labels),
        "ci_low": lo,
        "ci_high": hi,
        "sensitivity": sens,
        "specificity": spec,
        "brier": brier_score(probs, labels),
        "auc_prc": auc_prc(scores, labels),
    }


def evaluate_tkr(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> dict[str, float]:
    return {
        "c_index": harrell_cindex(risk, time, event),
        "auc_prc": auc_prc(risk, event),
    }
