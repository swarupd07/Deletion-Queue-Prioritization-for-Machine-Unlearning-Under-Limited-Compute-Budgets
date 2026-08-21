"""Mechanism, statistics, queue-aware scheduling, and oracle experiments for RLS.

This module is intentionally separate from the existing experiment scripts.
It adds four paper-facing components:

1. Mechanism diagnostics: target alignment, cancellation, and ranking staleness.
2. Paired statistical tests with bootstrap confidence intervals and Holm correction.
3. Queue-Directional Score (QDS), a feasible queue-aware scheduling policy.
4. A target-aware greedy reference for small queues only.

The public entry point is ``run_repeated_study``.  It accepts the existing RLS
model, held-out arrays, and a queue-builder callback, so the same function can
be called from the Diabetes, Criteo, or other dataset runners.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from rls_influence import RLS


EPS = 1e-12
SCORED_METHODS = ("sis", "cooks", "leverage", "residual", "qds")
STATIC_SCORE_METHODS = SCORED_METHODS
MAIN_METHODS = SCORED_METHODS + ("random",)


def clone_state(model: RLS) -> RLS:
    """Clone mutable RLS state without deep-copying a potentially huge history."""
    other = RLS(model.dim, model.lam)
    other.P = model.P.copy()
    other.w = model.w.copy()
    other.n_seen = model.n_seen
    other.X_hist = model.X_hist       # read-only shared history
    other.y_hist = model.y_hist
    return other


def _queue_xy(model: RLS, queue: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Works whether history is stored as a list (original code) or ndarray.
    X = np.asarray(model.X_hist)
    y = np.asarray(model.y_hist)
    return X[queue], y[queue]


def deletion_vectors(model: RLS, queue: Sequence[int]) -> Dict[str, np.ndarray]:
    """Return exact *current-state* single-deletion vectors and scalar scores.

    Row i of ``delta`` is w_{-i} - w for deleting queue[i] in isolation from
    the current model.  QDS uses the alignment of each vector with the sum of
    all queue vectors, a first-order approximation to the queue-level update.
    It does not use the fully-deleted target. The QDS scalar is

        <delta_i, sum_j delta_j> / (||delta_i||^2 + eps),

    so it rewards alignment with the approximate queue direction while
    penalizing a large isolated update that could overshoot that direction.
    """
    queue = np.asarray(queue, dtype=int)
    Xq, yq = _queue_xy(model, queue)
    PX = Xq @ model.P
    h = np.einsum("ij,ij->i", Xq, PX)
    residual = yq - Xq @ model.w
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-8, np.copysign(1e-8, denom + EPS), denom)

    delta = -PX * (residual / denom)[:, None]
    sis = np.einsum("ij,ij->i", delta, delta)
    cooks = residual**2 * h / denom**2
    leverage = h.copy()
    residual_score = residual**2

    approximate_queue_direction = delta.sum(axis=0)
    # Marginal contribution to ||sum_j delta_j||^2, up to an additive term.
    # A large positive value means the point supports the aggregate queue update.
    qds = (delta @ approximate_queue_direction) / np.maximum(sis, EPS)

    return {
        "delta": delta,
        "sis": sis,
        "cooks": cooks,
        "leverage": leverage,
        "residual": residual_score,
        "qds": qds,
    }


def timed_priority_order(
    model: RLS,
    queue: Sequence[int],
    method: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    """Time scoring plus sorting and return a descending priority order."""
    queue = np.asarray(queue, dtype=int)
    if method == "random":
        # Random/FIFO-style baselines pay no influence-scoring cost. The RNG
        # cost is deliberately outside the compute budget, as queue arrival
        # order is also external to model computation.
        return rng.permutation(queue), 0.0, {}
    if method not in SCORED_METHODS:
        raise ValueError(f"Unknown method: {method}")

    # Compute only the quantities required by this standalone method. This is
    # essential for fair timing: residual scoring must not pay for P @ x, and
    # leverage must not pay for SIS vector norms.
    start = time.perf_counter()
    Xq, yq = _queue_xy(model, queue)
    if method == "residual":
        score = (yq - Xq @ model.w) ** 2
    else:
        PX = Xq @ model.P
        h = np.einsum("ij,ij->i", Xq, PX)
        if method == "leverage":
            score = h
        else:
            residual = yq - Xq @ model.w
            denom = 1.0 - h
            denom = np.where(
                np.abs(denom) < 1e-8,
                np.copysign(1e-8, denom + EPS),
                denom,
            )
            if method == "cooks":
                score = residual**2 * h / denom**2
            else:
                delta = -PX * (residual / denom)[:, None]
                if method == "sis":
                    score = np.einsum("ij,ij->i", delta, delta)
                else:  # qds: alignment per unit of individual update energy
                    score = (delta @ delta.sum(axis=0)) / np.maximum(
                        np.einsum("ij,ij->i", delta, delta), EPS
                    )
    order = queue[np.argsort(-score, kind="stable")]
    elapsed = time.perf_counter() - start
    return order, elapsed, {method: score}


def measure_unlearn_cost(
    model: RLS,
    candidates: Sequence[int],
    timing_reps: int = 11,
    seed: int = 0,
) -> float:
    """Median wall-clock cost of one exact unlearn operation.

    Every repetition starts from the same state. Model copying is outside the
    timed region, and each sample is deleted at most once in a repetition.
    """
    candidates = np.asarray(candidates, dtype=int)
    if len(candidates) == 0:
        raise ValueError("candidates must be non-empty")
    rng = np.random.default_rng(seed)
    sample_size = min(len(candidates), 256)
    per_deletion = []
    for _ in range(timing_reps):
        chosen = rng.choice(candidates, size=sample_size, replace=False)
        working = clone_state(model)
        start = time.perf_counter()
        for idx in chosen:
            working.unlearn(int(idx))
        per_deletion.append((time.perf_counter() - start) / sample_size)
    return float(np.median(per_deletion))


def fully_deleted_target(model: RLS, queue: Sequence[int]) -> np.ndarray:
    target = clone_state(model)
    for idx in queue:
        target.unlearn(int(idx))
    return target.w.copy()


def _progress(w: np.ndarray, w0: np.ndarray, target: np.ndarray) -> float:
    total = float(np.sum((w0 - target) ** 2))
    if total <= EPS:
        return 1.0
    gap = float(np.sum((w - target) ** 2))
    return 1.0 - gap / total


def _cosine_rows(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(direction)
    return (vectors @ direction) / np.maximum(denom, EPS)


def cancellation_diagnostics(
    vectors: np.ndarray,
    target_direction: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int = 100_000,
) -> Dict[str, float]:
    """Quantify direction alignment and cancellation for selected updates."""
    if len(vectors) == 0:
        return {
            "mean_target_cosine": math.nan,
            "negative_target_cosine_fraction": math.nan,
            "cancellation_ratio": math.nan,
            "mean_pairwise_cosine": math.nan,
        }
    target_cos = _cosine_rows(vectors, target_direction)
    sum_norm = float(np.linalg.norm(vectors.sum(axis=0)))
    norm_sum = float(np.linalg.norm(vectors, axis=1).sum())

    n = len(vectors)
    if n < 2:
        pair_cos = math.nan
    else:
        total_pairs = n * (n - 1) // 2
        count = min(max_pairs, total_pairs)
        a = rng.integers(0, n, size=count)
        b = rng.integers(0, n - 1, size=count)
        b = b + (b >= a)
        va, vb = vectors[a], vectors[b]
        denom = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)
        pair_cos = float(np.mean(np.einsum("ij,ij->i", va, vb) / np.maximum(denom, EPS)))

    return {
        "mean_target_cosine": float(np.mean(target_cos)),
        "negative_target_cosine_fraction": float(np.mean(target_cos < 0.0)),
        # 1 means perfect directional reinforcement; near 0 means cancellation.
        "cancellation_ratio": sum_norm / max(norm_sum, EPS),
        "mean_pairwise_cosine": pair_cos,
    }


def ranking_staleness(
    model: RLS,
    queue: Sequence[int],
    method: str,
    checkpoints: Iterable[float] = (0.05, 0.10, 0.25, 0.50),
) -> Dict[str, float]:
    """Spearman correlation between initial and updated scores on survivors."""
    if method not in STATIC_SCORE_METHODS:
        raise ValueError("Staleness is defined only for static scalar-score methods")
    queue = np.asarray(queue, dtype=int)
    initial = deletion_vectors(model, queue)[method]
    initial_by_id = {int(i): float(s) for i, s in zip(queue, initial)}
    static_order = queue[np.argsort(-initial, kind="stable")]
    working = clone_state(model)
    output = {}
    deleted = 0
    for fraction in sorted(set(float(x) for x in checkpoints)):
        target_deleted = min(len(queue) - 1, int(round(fraction * len(queue))))
        for idx in static_order[deleted:target_deleted]:
            working.unlearn(int(idx))
        deleted = target_deleted
        remaining = static_order[deleted:]
        if len(remaining) < 3:
            rho = math.nan
        else:
            current = deletion_vectors(working, remaining)[method]
            old = np.asarray([initial_by_id[int(i)] for i in remaining])
            rho = float(spearmanr(old, current).statistic)
        output[f"rho_after_{fraction:g}"] = rho
    return output


def greedy_target_reference(
    model: RLS,
    queue: Sequence[int],
    target_w: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Expensive one-step look-ahead reference; use only on small queues.

    At each step it tries every remaining request and selects the deletion that
    minimizes exact parameter distance to the fully-deleted target. Because it
    uses target_w and O(steps * queue_size) candidate downdates, this is a
    diagnostic upper reference, not a deployable equal-budget policy.
    """
    remaining = [int(i) for i in queue]
    working = clone_state(model)
    chosen = []
    for _ in range(min(steps, len(remaining))):
        best_idx, best_gap = None, float("inf")
        for idx in remaining:
            trial = clone_state(working)
            trial.unlearn(idx)
            gap = float(np.sum((trial.w - target_w) ** 2))
            if gap < best_gap:
                best_idx, best_gap = idx, gap
        working.unlearn(best_idx)
        chosen.append(best_idx)
        remaining.remove(best_idx)
    return np.asarray(chosen, dtype=int)


def evaluate_one_queue(
    model: RLS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    queue: Sequence[int],
    budget_fractions: Sequence[float],
    timing_reps: int,
    seed: int,
    staleness_checkpoints: Sequence[float] = (0.05, 0.10, 0.25, 0.50),
    oracle_queue_size: int = 30,
) -> Dict:
    """Run all methods on one queue under identical wall-clock budgets."""
    queue = np.asarray(queue, dtype=int)
    if len(np.unique(queue)) != len(queue):
        raise ValueError("A deletion queue must contain unique training indices")
    rng = np.random.default_rng(seed)
    w0 = model.w.copy()
    target_w = fully_deleted_target(model, queue)
    target_direction = target_w - w0
    unlearn_sec = measure_unlearn_cost(model, queue, timing_reps, seed)

    # A 100% budget means enough measured deletion time to process Q requests
    # if no scoring were required.
    budgets_sec = np.asarray(budget_fractions, dtype=float) * len(queue) * unlearn_sec
    result = {
        "queue_size": int(len(queue)),
        "unlearn_seconds": unlearn_sec,
        "budgets_seconds": budgets_sec.tolist(),
        "budget_fractions": list(map(float, budget_fractions)),
        "methods": {},
        "staleness": {},
    }

    initial_bundle = deletion_vectors(model, queue)
    for method in MAIN_METHODS:
        order, score_seconds, bundle = timed_priority_order(model, queue, method, rng)
        counts = np.floor(np.maximum(0.0, budgets_sec - score_seconds) / unlearn_sec).astype(int)
        counts = np.clip(counts, 0, len(queue))
        requested_counts = set(int(x) for x in counts)
        working = clone_state(model)
        progress_by_count = {0: _progress(working.w, w0, target_w)}
        mse_by_count = {0: float(np.mean((y_test - X_test @ working.w) ** 2))}
        for step, idx in enumerate(order, start=1):
            if step > max(requested_counts, default=0):
                break
            working.unlearn(int(idx))
            if step in requested_counts:
                progress_by_count[step] = _progress(working.w, w0, target_w)
                mse_by_count[step] = float(np.mean((y_test - X_test @ working.w) ** 2))

        # Diagnose a common top-25% prefix independent of whether scoring cost
        # leaves zero deletions at a very small wall-clock budget.
        diagnostic_k = max(1, int(math.ceil(0.25 * len(queue))))
        selected = order[:diagnostic_k]
        pos = {int(idx): j for j, idx in enumerate(queue)}
        selected_vectors = initial_bundle["delta"][[pos[int(i)] for i in selected]]
        mechanism = cancellation_diagnostics(selected_vectors, target_direction, rng)

        result["methods"][method] = {
            "score_sort_seconds": float(score_seconds),
            "score_cost_deletion_equivalents": float(score_seconds / max(unlearn_sec, EPS)),
            "deletions_completed": counts.tolist(),
            "progress": [progress_by_count[int(k)] for k in counts],
            "test_mse": [mse_by_count[int(k)] for k in counts],
            "mechanism_top_25pct": mechanism,
        }
        if method in STATIC_SCORE_METHODS:
            result["staleness"][method] = ranking_staleness(
                model, queue, method, staleness_checkpoints
            )

    # Small-queue target-aware reference, evaluated separately from equal budget.
    oq = queue[: min(oracle_queue_size, len(queue))]
    oracle_target = fully_deleted_target(model, oq)
    oracle_steps = max(1, len(oq) // 2)
    oracle_order = greedy_target_reference(model, oq, oracle_target, oracle_steps)
    oracle_working = clone_state(model)
    for idx in oracle_order:
        oracle_working.unlearn(int(idx))
    result["greedy_reference"] = {
        "queue_size": int(len(oq)),
        "steps": int(oracle_steps),
        "progress": _progress(oracle_working.w, w0, oracle_target),
        "note": "Target-aware one-step look-ahead; diagnostic reference, not a fair deployable policy.",
    }
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(draws, [alpha, 1.0 - alpha]).tolist())


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_statistics(
    raw: Mapping[str, np.ndarray],
    reference: str = "random",
    seed: int = 1234,
) -> list[Dict]:
    """Paired tests at every budget; intended for progress or test MSE arrays.

    ``raw[method]`` must have shape (repeats, budgets), with matched rows.
    """
    rng = np.random.default_rng(seed)
    ref = np.asarray(raw[reference], dtype=float)
    records = []
    for method, values in raw.items():
        if method == reference:
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != ref.shape:
            raise ValueError("Paired arrays must have identical shapes")
        for budget_idx in range(values.shape[1]):
            diff = values[:, budget_idx] - ref[:, budget_idx]
            ci_low, ci_high = _bootstrap_mean_ci(diff, rng)
            sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else math.nan
            dz = float(np.mean(diff) / sd) if sd > EPS else math.nan
            t_p = float(ttest_rel(values[:, budget_idx], ref[:, budget_idx]).pvalue)
            try:
                w_p = float(wilcoxon(diff, zero_method="pratt").pvalue)
            except ValueError:
                w_p = 1.0
            records.append({
                "method": method,
                "reference": reference,
                "budget_index": budget_idx,
                "mean_difference": float(np.mean(diff)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "cohens_dz": dz,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            })
    for field in ("paired_t_p", "wilcoxon_p"):
        adjusted = _holm_adjust([row[field] for row in records])
        for row, value in zip(records, adjusted):
            row[field + "_holm"] = float(value)
    return records


def run_repeated_study(
    model: RLS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    queue_builder: Callable[[int], Sequence[int]],
    repeats: int = 30,
    budget_fractions: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
    timing_reps: int = 11,
    oracle_queue_size: int = 30,
    seed: int = 7000,
) -> Dict:
    """Repeat matched-queue experiments and calculate inferential statistics."""
    repetitions = []
    for rep in range(repeats):
        queue = np.asarray(queue_builder(rep), dtype=int)
        repetitions.append(evaluate_one_queue(
            model=model,
            X_test=X_test,
            y_test=y_test,
            queue=queue,
            budget_fractions=budget_fractions,
            timing_reps=timing_reps,
            seed=seed + rep,
            oracle_queue_size=oracle_queue_size,
        ))

    summary = {
        "repeats": repeats,
        "budget_fractions": list(map(float, budget_fractions)),
        "methods": {},
        "paired_progress_vs_random": [],
        "paired_mse_vs_random": [],
        "mechanism_summary": {},
        "staleness_summary": {},
        "greedy_reference_summary": {},
        "repetitions": repetitions,
    }
    progress_raw, mse_raw = {}, {}
    for method in MAIN_METHODS:
        progress = np.asarray([r["methods"][method]["progress"] for r in repetitions])
        mse = np.asarray([r["methods"][method]["test_mse"] for r in repetitions])
        counts = np.asarray([r["methods"][method]["deletions_completed"] for r in repetitions])
        progress_raw[method], mse_raw[method] = progress, mse
        summary["methods"][method] = {
            "progress_mean": progress.mean(axis=0).tolist(),
            "progress_std": progress.std(axis=0, ddof=1).tolist(),
            "test_mse_mean": mse.mean(axis=0).tolist(),
            "test_mse_std": mse.std(axis=0, ddof=1).tolist(),
            "deletions_mean": counts.mean(axis=0).tolist(),
            "deletions_std": counts.std(axis=0, ddof=1).tolist(),
            "score_cost_deletion_equivalents_mean": float(np.mean([
                r["methods"][method]["score_cost_deletion_equivalents"] for r in repetitions
            ])),
        }
    summary["paired_progress_vs_random"] = paired_statistics(progress_raw, "random", seed + 1)
    summary["paired_mse_vs_random"] = paired_statistics(mse_raw, "random", seed + 2)

    for method in MAIN_METHODS:
        records = [r["methods"][method]["mechanism_top_25pct"] for r in repetitions]
        summary["mechanism_summary"][method] = {}
        for metric in records[0]:
            values = np.asarray([x[metric] for x in records], dtype=float)
            summary["mechanism_summary"][method][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
            }
    for method in STATIC_SCORE_METHODS:
        records = [r["staleness"][method] for r in repetitions]
        summary["staleness_summary"][method] = {}
        for checkpoint in records[0]:
            values = np.asarray([x[checkpoint] for x in records], dtype=float)
            summary["staleness_summary"][method][checkpoint] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
            }
    oracle_values = np.asarray(
        [r["greedy_reference"]["progress"] for r in repetitions], dtype=float
    )
    summary["greedy_reference_summary"] = {
        "progress_mean": float(np.mean(oracle_values)),
        "progress_std": float(np.std(oracle_values, ddof=1)),
        "queue_size": int(repetitions[0]["greedy_reference"]["queue_size"]),
        "steps": int(repetitions[0]["greedy_reference"]["steps"]),
    }
    return summary