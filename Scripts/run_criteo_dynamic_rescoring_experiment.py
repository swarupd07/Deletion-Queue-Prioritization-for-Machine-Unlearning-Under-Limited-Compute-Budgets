"""Criteo equal-compute experiment with dynamic 10% batch rescoring.

For SIS, Cook's, leverage, and residual, the policy repeatedly:

    1. scores every request still in the deletion queue using the CURRENT w, P;
    2. sorts the remaining requests in descending score order;
    3. exactly unlearns one batch (default: 10% of the original queue);
    4. recomputes scores using the newly updated w, P.

Random uses one random queue order and pays no scoring cost. Every policy is
evaluated at the same total wall-clock budgets. Repeated scoring and sorting
costs are charged to the scored methods.

This is a new experiment and does not modify the fixed-ranking runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rls_influence import RLS
from run_criteo_equal_compute_budget import (
    EXPECTED_TOTAL_ROWS,
    OFFICIAL_URL,
    build_large_rls_state,
    obtain_dataset,
)
from run_equal_compute_budget_all_methods import (
    BUDGET_FRACTIONS,
    METHODS,
    SCORED_METHODS,
    benchmark_average_unlearn_cost,
    make_target,
    median_timed_call,
    scores_for_queue,
)


def metrics_for_weights(
    weights: list[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    w0: np.ndarray,
    w_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target progress and held-out MSE for saved weight vectors."""
    total_change = float(np.sum((w0 - w_target) ** 2))
    progress, mse = [], []
    for w in weights:
        remaining_gap = float(np.sum((w - w_target) ** 2))
        value = 1.0 - (
            remaining_gap / total_change if total_change > 1e-12 else 0.0
        )
        progress.append(value)
        mse.append(float(np.mean((y_test - X_test @ w) ** 2)))
    return np.asarray(progress), np.asarray(mse)


def run_dynamic_scored_policy(
    model: RLS,
    queue: np.ndarray,
    method: str,
    budgets_sec: np.ndarray,
    average_unlearn_sec: float,
    batch_size: int,
    timing_reps: int,
) -> dict:
    """Run one dynamically rescored policy along the largest budget path."""
    working = model.copy()
    remaining = np.asarray(queue, dtype=int).copy()
    max_budget = float(budgets_sec[-1])

    # State at each deadline. A score pass is treated as indivisible: if its
    # cost exceeds a deadline, that deadline sees the pre-score model.
    snapshots: list[np.ndarray | None] = [None] * len(budgets_sec)
    deleted_at_budget = np.zeros(len(budgets_sec), dtype=int)
    charged_at_budget = np.zeros(len(budgets_sec), dtype=float)
    rounds_at_budget = np.zeros(len(budgets_sec), dtype=int)
    next_budget = 0

    charged = 0.0
    n_deleted = 0
    rounds_completed = 0
    score_total = 0.0
    sort_total = 0.0

    def fill_deadlines_before(event_end: float, incomplete_score: bool) -> None:
        """Save state for deadlines that occur before an event completes."""
        nonlocal next_budget
        while (
            next_budget < len(budgets_sec)
            and budgets_sec[next_budget] + 1e-15 < event_end
        ):
            snapshots[next_budget] = working.w.copy()
            deleted_at_budget[next_budget] = n_deleted
            # As in the original experiment, an incomplete scoring pass uses
            # the whole available deadline. For an unstarted exact deletion,
            # only completed work is charged.
            charged_at_budget[next_budget] = (
                float(budgets_sec[next_budget]) if incomplete_score else charged
            )
            rounds_at_budget[next_budget] = rounds_completed
            next_budget += 1

    while len(remaining) and charged < max_budget:
        scores, score_sec = median_timed_call(
            lambda: scores_for_queue(working, remaining, method), timing_reps
        )
        positions, sort_sec = median_timed_call(
            lambda: np.argsort(-scores, kind="stable"), timing_reps
        )
        priority_cost = score_sec + sort_sec

        fill_deadlines_before(charged + priority_cost, incomplete_score=True)
        if charged + priority_cost > max_budget + 1e-15:
            break

        charged += priority_cost
        score_total += score_sec
        sort_total += sort_sec
        rounds_completed += 1

        # A round deletes at most 10% of the ORIGINAL queue, then rescoring is
        # mandatory. If the final deadline cuts through a batch, use the
        # affordable prefix but do not start another score pass.
        selected_positions = positions[: min(batch_size, len(remaining))]
        selected_ids = remaining[selected_positions]
        completed_this_round = 0

        for request_id in selected_ids:
            deletion_end = charged + average_unlearn_sec
            fill_deadlines_before(deletion_end, incomplete_score=False)
            if deletion_end > max_budget + 1e-15:
                break
            working.unlearn(int(request_id))
            charged = deletion_end
            n_deleted += 1
            completed_this_round += 1

            while (
                next_budget < len(budgets_sec)
                and charged <= budgets_sec[next_budget] + 1e-15
            ):
                # Do not fill a future deadline yet unless another event would
                # cross it; this loop only handles exact equality.
                if abs(charged - budgets_sec[next_budget]) > 1e-15:
                    break
                snapshots[next_budget] = working.w.copy()
                deleted_at_budget[next_budget] = n_deleted
                charged_at_budget[next_budget] = charged
                rounds_at_budget[next_budget] = rounds_completed
                next_budget += 1

        if completed_this_round:
            remove_mask = np.ones(len(remaining), dtype=bool)
            remove_mask[selected_positions[:completed_this_round]] = False
            remaining = remaining[remove_mask]

        if completed_this_round < len(selected_ids):
            break

    # No more complete operations fit, or the queue is empty.
    while next_budget < len(budgets_sec):
        snapshots[next_budget] = working.w.copy()
        deleted_at_budget[next_budget] = n_deleted
        charged_at_budget[next_budget] = charged
        rounds_at_budget[next_budget] = rounds_completed
        next_budget += 1

    return {
        "weights": [w for w in snapshots if w is not None],
        "n_deleted": deleted_at_budget,
        "charged_compute_sec": charged_at_budget,
        "rescoring_rounds_completed": rounds_at_budget,
        "scoring_completed": rounds_at_budget > 0,
        "score_sec_total": score_total,
        "sort_sec_total": sort_total,
    }


def run_random_policy(
    model: RLS,
    queue: np.ndarray,
    seed: int,
    budgets_sec: np.ndarray,
    average_unlearn_sec: float,
) -> dict:
    """Random baseline: no score or sort cost, only exact deletions."""
    order = np.random.default_rng(seed).permutation(queue)
    ks = np.minimum(
        len(queue),
        np.floor(
            (budgets_sec + average_unlearn_sec * 1e-9) / average_unlearn_sec
        ).astype(int),
    )
    requested = set(int(k) for k in ks)
    snapshots = {0: model.w.copy()}
    working = model.copy()
    for step, request_id in enumerate(order, start=1):
        working.unlearn(int(request_id))
        if step in requested:
            snapshots[step] = working.w.copy()
        if step >= int(ks[-1]):
            break
    return {
        "weights": [snapshots[int(k)] for k in ks],
        "n_deleted": ks,
        "charged_compute_sec": ks * average_unlearn_sec,
        "rescoring_rounds_completed": np.zeros(len(ks), dtype=int),
        "scoring_completed": np.ones(len(ks), dtype=bool),
        "score_sec_total": 0.0,
        "sort_sec_total": 0.0,
    }


def run_realization_dynamic(
    model: RLS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    queue: np.ndarray,
    seed: int,
    timing_reps: int,
    batch_fraction: float,
) -> dict:
    queue = np.asarray(queue, dtype=int)
    Q = len(queue)
    batch_size = max(1, int(np.ceil(batch_fraction * Q)))
    w0 = model.w.copy()
    w_target = make_target(model, queue)

    # Warm up kernels outside the measured budget.
    for method in SCORED_METHODS:
        _ = scores_for_queue(model, queue, method)
    warm = model.copy()
    warm.unlearn(int(queue[0]))

    average_unlearn_sec = benchmark_average_unlearn_cost(
        model, queue, timing_reps
    )
    budgets_sec = BUDGET_FRACTIONS * Q * average_unlearn_sec

    out = {
        "queue_size": Q,
        "batch_size": batch_size,
        "batch_fraction": batch_fraction,
        "average_unlearn_sec": average_unlearn_sec,
        "budgets_sec": budgets_sec.tolist(),
        "methods": {},
    }

    for method in METHODS:
        if method == "random":
            policy = run_random_policy(
                model, queue, seed, budgets_sec, average_unlearn_sec
            )
        else:
            policy = run_dynamic_scored_policy(
                model,
                queue,
                method,
                budgets_sec,
                average_unlearn_sec,
                batch_size,
                timing_reps,
            )
        progress, mse = metrics_for_weights(
            policy.pop("weights"), X_test, y_test, w0, w_target
        )
        out["methods"][method] = {
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in policy.items()
            },
            "progress": progress.tolist(),
            "test_mse": mse.tolist(),
        }
    return out


def aggregate_dynamic(realizations: list[dict]) -> dict:
    out = {
        "n_repeats": len(realizations),
        "budget_fraction": BUDGET_FRACTIONS.tolist(),
        "queue_size_mean": float(np.mean([r["queue_size"] for r in realizations])),
        "batch_size_mean": float(np.mean([r["batch_size"] for r in realizations])),
        "batch_fraction": float(realizations[0]["batch_fraction"]),
    }
    for key in ("budgets_sec",):
        values = np.asarray([r[key] for r in realizations], dtype=float)
        out["budget_sec_mean"] = values.mean(axis=0).tolist()
        out["budget_sec_std"] = values.std(axis=0, ddof=1).tolist()

    values = np.asarray([r["average_unlearn_sec"] for r in realizations])
    out["average_unlearn_sec_mean"] = float(values.mean())
    out["average_unlearn_sec_std"] = float(values.std(ddof=1))

    out["timing"] = {}
    out["methods"] = {}
    for method in METHODS:
        score_values = np.asarray(
            [r["methods"][method]["score_sec_total"] for r in realizations]
        )
        sort_values = np.asarray(
            [r["methods"][method]["sort_sec_total"] for r in realizations]
        )
        priority_values = score_values + sort_values
        out["timing"][method] = {
            "score_sec_mean": float(score_values.mean()),
            "score_sec_std": float(score_values.std(ddof=1)),
            "sort_sec_mean": float(sort_values.mean()),
            "sort_sec_std": float(sort_values.std(ddof=1)),
            "priority_total_sec_mean": float(priority_values.mean()),
            "priority_total_sec_std": float(priority_values.std(ddof=1)),
        }

        out["methods"][method] = {}
        for metric in (
            "scoring_completed",
            "rescoring_rounds_completed",
            "n_deleted",
            "charged_compute_sec",
            "progress",
            "test_mse",
        ):
            metric_values = np.asarray(
                [r["methods"][method][metric] for r in realizations], dtype=float
            )
            out["methods"][method][f"{metric}_mean"] = (
                metric_values.mean(axis=0).tolist()
            )
            out["methods"][method][f"{metric}_std"] = (
                metric_values.std(axis=0, ddof=1).tolist()
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/criteo"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--hash-dim", type=int, default=32)
    parser.add_argument("--lambda-ridge", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=20_000)
    parser.add_argument("--queue-pool-size", type=int, default=25_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument("--batch-fraction", type=float, default=0.10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-reps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("criteo_dynamic_rescoring_results.json"),
    )
    args = parser.parse_args()

    if args.queue_size > args.queue_pool_size:
        raise ValueError("--queue-size cannot exceed --queue-pool-size")
    if not 0.0 < args.batch_fraction <= 1.0:
        raise ValueError("--batch-fraction must be in (0, 1]")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.timing_reps < 3 or args.timing_reps % 2 == 0:
        raise ValueError("--timing-reps must be an odd integer >= 3")

    data_path = args.data_path or obtain_dataset(
        args.data_dir, accept_license=args.accept_license
    )
    model, X_test, y_test, n_train, n_test_stream = build_large_rls_state(
        data_path=data_path,
        max_rows=args.max_rows,
        train_fraction=args.train_fraction,
        chunksize=args.chunksize,
        hash_dim=args.hash_dim,
        lam=args.lambda_ridge,
        queue_pool_size=args.queue_pool_size,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    for rep in range(args.repeats):
        queue = np.random.default_rng(args.seed + 10_000 + rep).choice(
            args.queue_pool_size, size=args.queue_size, replace=False
        )
        realizations.append(
            run_realization_dynamic(
                model,
                X_test,
                y_test,
                queue,
                seed=args.seed + 20_000 + rep,
                timing_reps=args.timing_reps,
                batch_fraction=args.batch_fraction,
            )
        )
        print(f"finished queue realization {rep + 1}/{args.repeats}")

    summary = aggregate_dynamic(realizations)
    summary.update(
        {
            "n_total_rows_used": args.max_rows,
            "n_train": n_train,
            "n_test_in_stream": n_test_stream,
            "n_test_evaluated": len(y_test),
            "dim_including_bias": model.dim,
            "hash_dim": args.hash_dim,
            "target": "click",
            "test_metric": "unclipped linear-prediction MSE (Brier-style)",
            "queue_source": (
                "simulated row-level deletion requests uniformly sampled from "
                "real Criteo training impressions"
            ),
        }
    )
    results = {
        "experiment": "criteo_dynamic_batch_rescoring_equal_compute_budget",
        "source_url": OFFICIAL_URL,
        "official_total_rows": EXPECTED_TOTAL_ROWS,
        "ranking_direction": "descending",
        "rescore_rule": (
            "score remaining queue using current w and P after every completed "
            "batch; batch size is a fraction of the original queue"
        ),
        "budget_definition": (
            "fraction of measured wall-clock cost of deleting the entire queue "
            "with no scoring; every repeated score and sort pass is charged"
        ),
        "datasets": {"Criteo Attribution": summary},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
