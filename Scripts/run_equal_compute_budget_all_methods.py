"""Equal-total-wall-clock-budget experiment for SIS-prioritized unlearning.

This is a NEW experiment. It does not change the existing fixed-number-of-
deletions queue experiment.

For every queue realization and every scored method, the experiment measures:
  * the wall-clock cost of computing scores for every queued request;
  * the wall-clock cost of sorting those scores;
  * the average wall-clock cost of one exact RLS.unlearn() operation.

At each fixed total compute budget, SIS, Cook's, leverage, and residual pay
their own independently measured scoring + sorting costs first and spend only
the remainder on exact deletions. Random pays no scoring cost and spends the
whole budget on exact deletions. All methods are therefore charged against the
same wall-clock budget. Evaluation time and the one-off construction of the
fully-deleted target are deliberately excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rls_influence import RLS


BUDGET_FRACTIONS = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
SCORED_METHODS = ("sis", "cooks", "leverage", "residual")
METHODS = (*SCORED_METHODS, "random")


def make_synthetic(n=400, d=10, n_outliers=20, noise=0.3, seed=1):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    X = np.hstack([np.ones((n, 1)), X])
    w_true = r.normal(size=d + 1) * 1.5
    y = X @ w_true + r.normal(scale=noise, size=n)
    outlier_idx = r.choice(n, size=n_outliers, replace=False)
    X[outlier_idx, 1:] *= r.uniform(4, 7, size=(n_outliers, d))
    y[outlier_idx] += r.normal(scale=8.0, size=n_outliers)
    return X, y, outlier_idx


def scores_for_queue(
    model: RLS, queue: np.ndarray, method: str
) -> np.ndarray:
    """Compute one score as a standalone pipeline over the Q requested rows.

    Timing each method separately prevents SIS from unfairly reusing work that
    was performed while computing another score.
    """
    if method not in SCORED_METHODS:
        raise ValueError(f"unknown scored method: {method}")

    Xq = np.asarray([model.X_hist[int(i)] for i in queue], dtype=float)
    if method == "residual":
        yq = np.asarray([model.y_hist[int(i)] for i in queue], dtype=float)
        return (yq - Xq @ model.w) ** 2

    Px = Xq @ model.P
    h = np.einsum("ij,ij->i", Xq, Px)
    if method == "leverage":
        return h

    yq = np.asarray([model.y_hist[int(i)] for i in queue], dtype=float)
    residual_sq = (yq - Xq @ model.w) ** 2
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-8, np.copysign(1e-8, denom), denom)
    if method == "cooks":
        return residual_sq * h / denom**2

    px_norm_sq = np.einsum("ij,ij->i", Px, Px)
    return residual_sq * px_norm_sq / denom**2


def median_timed_call(fn, n_reps: int):
    """Return (last result, median elapsed seconds) for a side-effect-free fn."""
    times = []
    result = None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(n_reps):
            t0 = time.perf_counter_ns()
            result = fn()
            times.append((time.perf_counter_ns() - t0) * 1e-9)
    finally:
        if gc_was_enabled:
            gc.enable()
    return result, float(np.median(times))


def benchmark_average_unlearn_cost(
    model: RLS, queue: np.ndarray, n_reps: int
) -> float:
    """Median total queue time divided by Q: average exact deletion cost."""
    totals = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(n_reps):
            working = model.copy()  # copying is setup and is not timed
            t0 = time.perf_counter_ns()
            for i in queue:
                working.unlearn(int(i))
            totals.append((time.perf_counter_ns() - t0) * 1e-9)
    finally:
        if gc_was_enabled:
            gc.enable()
    return float(np.median(totals) / len(queue))


def make_target(model: RLS, queue: np.ndarray) -> np.ndarray:
    target = model.copy()
    for i in queue:
        target.unlearn(int(i))
    return target.w.copy()


def evaluate_prefixes(
    model: RLS,
    order: np.ndarray,
    ks: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w0: np.ndarray,
    w_target: np.ndarray,
):
    """Evaluate only requested prefix lengths; evaluation is outside budget."""
    requested = set(int(k) for k in ks)
    snapshots = {0: model.w.copy()}
    working = model.copy()
    for step, i in enumerate(order, start=1):
        working.unlearn(int(i))
        if step in requested:
            snapshots[step] = working.w.copy()

    total_change = float(np.sum((w0 - w_target) ** 2))
    progress, mse = [], []
    for k in ks:
        w = snapshots[int(k)]
        remaining_gap = float(np.sum((w - w_target) ** 2))
        p = 1.0 - (remaining_gap / total_change if total_change > 1e-12 else 0.0)
        progress.append(p)
        mse.append(float(np.mean((y_test - X_test @ w) ** 2)))
    return np.asarray(progress), np.asarray(mse)


def run_realization(
    model: RLS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    queue: np.ndarray,
    seed: int,
    timing_reps: int,
):
    queue = np.asarray(queue, dtype=int)
    Q = len(queue)
    w0 = model.w.copy()
    w_target = make_target(model, queue)

    # Warm up the two hot kernels once; warm-up time is never charged.
    for method in SCORED_METHODS:
        _ = scores_for_queue(model, queue, method)
    warm = model.copy()
    warm.unlearn(int(queue[0]))

    scoring_costs = {}
    sorting_costs = {}
    priority_costs = {}
    orders = {}
    for method in SCORED_METHODS:
        method_scores, t_score = median_timed_call(
            lambda m=method: scores_for_queue(model, queue, m), timing_reps
        )
        positions, t_sort = median_timed_call(
            lambda s=method_scores: np.argsort(-s, kind="stable"), timing_reps
        )
        scoring_costs[method] = t_score
        sorting_costs[method] = t_sort
        priority_costs[method] = t_score + t_sort
        orders[method] = queue[positions]

    t_unlearn_avg = benchmark_average_unlearn_cost(model, queue, timing_reps)

    # A 100% budget equals the measured cost of Q exact deletions with no score.
    budgets_sec = BUDGET_FRACTIONS * Q * t_unlearn_avg
    orders["random"] = np.random.default_rng(seed).permutation(queue)
    overhead = {**priority_costs, "random": 0.0}

    out = {
        "queue_size": int(Q),
        "scoring_costs_sec": scoring_costs,
        "sorting_costs_sec": sorting_costs,
        "priority_costs_sec": priority_costs,
        "average_unlearn_sec": t_unlearn_avg,
        "budgets_sec": budgets_sec.tolist(),
        "methods": {},
    }

    for method in METHODS:
        scoring_completed = budgets_sec >= overhead[method]
        remaining = np.where(
            scoring_completed, budgets_sec - overhead[method], 0.0
        )
        # The epsilon prevents an exact Q*t_unlearn_avg budget from becoming
        # Q-1 solely because of floating-point representation.
        ks = np.minimum(
            Q, np.floor((remaining + t_unlearn_avg * 1e-9) / t_unlearn_avg).astype(int)
        )
        progress, mse = evaluate_prefixes(
            model, orders[method], ks, X_test, y_test, w0, w_target
        )
        # If the deadline is shorter than a complete scoring pass, the scored
        # policy cannot obtain a valid ranking. It consumes the available
        # budget, completes no deletion, and leaves the model at w0.
        charged = np.where(
            scoring_completed,
            overhead[method] + ks * t_unlearn_avg,
            budgets_sec,
        )
        out["methods"][method] = {
            "scoring_completed": scoring_completed.tolist(),
            "n_deleted": ks.tolist(),
            "charged_compute_sec": charged.tolist(),
            "progress": progress.tolist(),
            "test_mse": mse.tolist(),
        }
    return out


def aggregate_realizations(realizations):
    out = {
        "n_repeats": len(realizations),
        "budget_fraction": BUDGET_FRACTIONS.tolist(),
        "queue_size_mean": float(np.mean([r["queue_size"] for r in realizations])),
    }
    budgets = np.asarray([r["budgets_sec"] for r in realizations], dtype=float)
    out["budget_sec_mean"] = budgets.mean(axis=0).tolist()
    out["budget_sec_std"] = budgets.std(axis=0, ddof=1).tolist()
    for key in ("average_unlearn_sec",):
        values = np.asarray([r[key] for r in realizations])
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=1))

    out["timing"] = {}
    for method in SCORED_METHODS:
        out["timing"][method] = {}
        for source_key, output_key in (
            ("scoring_costs_sec", "score_sec"),
            ("sorting_costs_sec", "sort_sec"),
            ("priority_costs_sec", "priority_total_sec"),
        ):
            values = np.asarray(
                [r[source_key][method] for r in realizations], dtype=float
            )
            out["timing"][method][f"{output_key}_mean"] = float(values.mean())
            out["timing"][method][f"{output_key}_std"] = float(
                values.std(ddof=1)
            )

    out["methods"] = {}
    for method in METHODS:
        out["methods"][method] = {}
        for metric in (
            "scoring_completed",
            "n_deleted",
            "charged_compute_sec",
            "progress",
            "test_mse",
        ):
            values = np.asarray(
                [r["methods"][method][metric] for r in realizations], dtype=float
            )
            out["methods"][method][f"{metric}_mean"] = values.mean(axis=0).tolist()
            out["methods"][method][f"{metric}_std"] = values.std(
                axis=0, ddof=1
            ).tolist()
    return out


def prepare_datasets():
    datasets = {}

    X, y, outlier_idx = make_synthetic(
        n=400, d=10, n_outliers=20, noise=0.3, seed=1
    )
    Xtr, Xte, ytr, yte, itr, _ = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, random_state=0
    )
    poison = np.where(np.isin(itr, outlier_idx))[0]
    benign = np.where(~np.isin(itr, outlier_idx))[0]

    def synthetic_queue(rep):
        r = np.random.default_rng(1000 + rep)
        q = np.concatenate([poison, r.choice(benign, size=40, replace=False)])
        r.shuffle(q)
        return q

    datasets["synthetic"] = (Xtr, ytr, Xte, yte, synthetic_queue)

    data = load_diabetes()
    Xd = StandardScaler().fit_transform(data.data)
    Xd = np.hstack([np.ones((len(Xd), 1)), Xd])
    yd = (data.target - data.target.mean()) / data.target.std()
    Xtr, Xte, ytr, yte = train_test_split(
        Xd, yd, test_size=0.3, random_state=0
    )

    def diabetes_queue(rep):
        return np.random.default_rng(2000 + rep).choice(
            len(Xtr), size=60, replace=False
        )

    datasets["diabetes"] = (Xtr, ytr, Xte, yte, diabetes_queue)

    Xl, yl, _ = make_synthetic(
        n=2000, d=25, n_outliers=60, noise=0.4, seed=3
    )
    Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(
        Xl, yl, test_size=0.3, random_state=0
    )

    def large_queue(rep):
        return np.random.default_rng(3000 + rep).choice(
            len(Xtr_l), size=60, replace=False
        )

    datasets["large_synthetic_n2000_d26"] = (
        Xtr_l,
        ytr_l,
        Xte_l,
        yte_l,
        large_queue,
    )
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-reps", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("equal_compute_budget_all_methods_results.json"),
    )
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to report a standard deviation")
    if args.timing_reps < 3 or args.timing_reps % 2 == 0:
        raise ValueError("--timing-reps must be an odd integer >= 3")

    results = {
        "experiment": "equal_total_wall_clock_budget",
        "budget_definition": (
            "fraction of measured wall-clock cost of deleting the entire queue "
            "with no scoring; evaluation and target construction excluded"
        ),
        "datasets": {},
    }

    for name, (Xtr, ytr, Xte, yte, queue_builder) in prepare_datasets().items():
        model = RLS(dim=Xtr.shape[1], lam=1.0).fit_stream(Xtr, ytr)
        realizations = []
        for rep in range(args.repeats):
            realizations.append(
                run_realization(
                    model,
                    Xte,
                    yte,
                    queue_builder(rep),
                    seed=9000 + rep,
                    timing_reps=args.timing_reps,
                )
            )
        results["datasets"][name] = aggregate_realizations(realizations)
        print(f"finished {name}: {args.repeats} queue realizations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()