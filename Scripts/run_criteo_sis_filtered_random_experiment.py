"""Equal-budget Criteo experiment for SIS-filtered random deletion.

The proposed policy performs one SIS scoring and sorting pass, excludes a
bottom-SIS portion of the deletion queue, and randomly deletes without
replacement only from the remaining requests.

The number excluded is determined by measured computation:

    exclusion_count = floor(
        score_cost_multiplier * (SIS score + sort time) / average_unlearn_time
    )

Excluded requests are NOT unlearned; they remain pending. The fully-deleted
target still removes the entire original queue. SIS scoring and sorting are
charged to the policy's total wall-clock budget. The Random baseline pays no
scoring cost and samples without replacement from the complete queue.
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
    benchmark_average_unlearn_cost,
    evaluate_prefixes,
    make_target,
    median_timed_call,
    scores_for_queue,
)


METHODS = ("sis_filtered_random", "random")


def time_sis_and_sort(
    model: RLS, queue: np.ndarray, timing_reps: int
) -> tuple[np.ndarray, float, float]:
    """Return SIS ascending positions and independently measured costs."""
    scores, score_sec = median_timed_call(
        lambda: scores_for_queue(model, queue, "sis"), timing_reps
    )
    ascending_positions, sort_sec = median_timed_call(
        lambda: np.argsort(scores, kind="stable"), timing_reps
    )
    return ascending_positions, score_sec, sort_sec


def run_realization(
    model: RLS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    queue: np.ndarray,
    seed: int,
    timing_reps: int,
    score_cost_multiplier: float,
) -> dict:
    queue = np.asarray(queue, dtype=int)
    Q = len(queue)
    if Q < 2:
        raise ValueError("queue must contain at least two requests")

    w0 = model.w.copy()
    w_target = make_target(model, queue)

    # Warm up both kernels. Warm-up time is not charged.
    _ = scores_for_queue(model, queue, "sis")
    _ = np.argsort(np.arange(Q, dtype=float), kind="stable")
    warm = model.copy()
    warm.unlearn(int(queue[0]))

    ascending_positions, score_sec, sort_sec = time_sis_and_sort(
        model, queue, timing_reps
    )
    priority_cost = score_sec + sort_sec
    average_unlearn_sec = benchmark_average_unlearn_cost(
        model, queue, timing_reps
    )

    score_deletion_equivalent = priority_cost / average_unlearn_sec
    requested_exclusion_count = int(
        np.floor(score_cost_multiplier * score_deletion_equivalent)
    )
    # Keep at least one request eligible so the proposed policy is defined.
    exclusion_count = min(Q - 1, max(0, requested_exclusion_count))

    excluded_positions = ascending_positions[:exclusion_count]
    eligible_positions = ascending_positions[exclusion_count:]
    excluded_queue = queue[excluded_positions]
    eligible_queue = queue[eligible_positions]

    rng = np.random.default_rng(seed)
    filtered_random_order = rng.permutation(eligible_queue)
    random_order = np.random.default_rng(seed + 1).permutation(queue)

    # A 100% budget is the measured cost of exactly deleting all Q requests
    # without scoring, matching the previous equal-compute experiments.
    budgets_sec = BUDGET_FRACTIONS * Q * average_unlearn_sec

    out = {
        "queue_size": Q,
        "score_cost_multiplier": score_cost_multiplier,
        "sis_score_sec": score_sec,
        "sis_sort_sec": sort_sec,
        "priority_cost_sec": priority_cost,
        "score_deletion_equivalent": score_deletion_equivalent,
        "requested_exclusion_count": requested_exclusion_count,
        "exclusion_count": exclusion_count,
        "eligible_count": int(len(eligible_queue)),
        "average_unlearn_sec": average_unlearn_sec,
        "budgets_sec": budgets_sec.tolist(),
        "methods": {},
    }

    # SIS-filtered random policy: scoring must finish before any deletion.
    scoring_completed = budgets_sec >= priority_cost
    remaining_budget = np.where(
        scoring_completed, budgets_sec - priority_cost, 0.0
    )
    filtered_ks = np.minimum(
        len(eligible_queue),
        np.floor(
            (remaining_budget + average_unlearn_sec * 1e-9)
            / average_unlearn_sec
        ).astype(int),
    )
    progress, mse = evaluate_prefixes(
        model,
        filtered_random_order,
        filtered_ks,
        X_test,
        y_test,
        w0,
        w_target,
    )
    filtered_charged = np.where(
        scoring_completed,
        priority_cost + filtered_ks * average_unlearn_sec,
        budgets_sec,
    )
    out["methods"]["sis_filtered_random"] = {
        "scoring_completed": scoring_completed.tolist(),
        "n_deleted": filtered_ks.tolist(),
        "n_pending_excluded": [exclusion_count] * len(BUDGET_FRACTIONS),
        "charged_compute_sec": filtered_charged.tolist(),
        "progress": progress.tolist(),
        "test_mse": mse.tolist(),
    }

    # Pure random baseline: entire queue is eligible and no score is charged.
    random_ks = np.minimum(
        Q,
        np.floor(
            (budgets_sec + average_unlearn_sec * 1e-9) / average_unlearn_sec
        ).astype(int),
    )
    progress, mse = evaluate_prefixes(
        model,
        random_order,
        random_ks,
        X_test,
        y_test,
        w0,
        w_target,
    )
    out["methods"]["random"] = {
        "scoring_completed": [True] * len(BUDGET_FRACTIONS),
        "n_deleted": random_ks.tolist(),
        "n_pending_excluded": [0] * len(BUDGET_FRACTIONS),
        "charged_compute_sec": (random_ks * average_unlearn_sec).tolist(),
        "progress": progress.tolist(),
        "test_mse": mse.tolist(),
    }

    # Useful audit fields: no excluded request may appear in the filtered
    # deletion order, and every eligible request appears exactly once.
    if np.intersect1d(excluded_queue, filtered_random_order).size:
        raise AssertionError("an excluded request entered the deletion order")
    if len(np.unique(filtered_random_order)) != len(filtered_random_order):
        raise AssertionError("filtered random order contains a duplicate")

    return out


def aggregate_realizations(realizations: list[dict]) -> dict:
    out = {
        "n_repeats": len(realizations),
        "budget_fraction": BUDGET_FRACTIONS.tolist(),
        "queue_size_mean": float(np.mean([r["queue_size"] for r in realizations])),
        "score_cost_multiplier": float(realizations[0]["score_cost_multiplier"]),
    }

    scalar_metrics = (
        "sis_score_sec",
        "sis_sort_sec",
        "priority_cost_sec",
        "score_deletion_equivalent",
        "requested_exclusion_count",
        "exclusion_count",
        "eligible_count",
        "average_unlearn_sec",
    )
    for metric in scalar_metrics:
        values = np.asarray([r[metric] for r in realizations], dtype=float)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std(ddof=1))

    budgets = np.asarray([r["budgets_sec"] for r in realizations], dtype=float)
    out["budget_sec_mean"] = budgets.mean(axis=0).tolist()
    out["budget_sec_std"] = budgets.std(axis=0, ddof=1).tolist()

    out["methods"] = {}
    for method in METHODS:
        out["methods"][method] = {}
        for metric in (
            "scoring_completed",
            "n_deleted",
            "n_pending_excluded",
            "charged_compute_sec",
            "progress",
            "test_mse",
        ):
            values = np.asarray(
                [r["methods"][method][metric] for r in realizations], dtype=float
            )
            out["methods"][method][f"{metric}_mean"] = (
                values.mean(axis=0).tolist()
            )
            out["methods"][method][f"{metric}_std"] = (
                values.std(axis=0, ddof=1).tolist()
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
    parser.add_argument("--score-cost-multiplier", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-reps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("criteo_sis_filtered_random_results.json"),
    )
    args = parser.parse_args()

    if args.queue_size > args.queue_pool_size:
        raise ValueError("--queue-size cannot exceed --queue-pool-size")
    if args.score_cost_multiplier < 0:
        raise ValueError("--score-cost-multiplier must be nonnegative")
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

    realizations = []
    for rep in range(args.repeats):
        queue = np.random.default_rng(args.seed + 10_000 + rep).choice(
            args.queue_pool_size, size=args.queue_size, replace=False
        )
        realizations.append(
            run_realization(
                model=model,
                X_test=X_test,
                y_test=y_test,
                queue=queue,
                seed=args.seed + 20_000 + rep,
                timing_reps=args.timing_reps,
                score_cost_multiplier=args.score_cost_multiplier,
            )
        )
        print(f"finished queue realization {rep + 1}/{args.repeats}")

    summary = aggregate_realizations(realizations)
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
        "experiment": "criteo_sis_filtered_random_equal_compute_budget",
        "source_url": OFFICIAL_URL,
        "official_total_rows": EXPECTED_TOTAL_ROWS,
        "policy": (
            "compute and sort SIS once; exclude the bottom-SIS requests; "
            "randomly delete without replacement only from the retained queue"
        ),
        "exclusion_rule": (
            "floor(score_cost_multiplier * priority_cost_sec / "
            "average_unlearn_sec), capped at queue_size - 1"
        ),
        "budget_definition": (
            "fraction of measured wall-clock cost of deleting the entire "
            "original queue with no scoring; SIS scoring and sorting are charged"
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