"""Compare SIS-, Cook's-, and randomly-filtered deletion queues on Criteo.

Each policy excludes low-priority requests and then exactly unlearns a random
sample without replacement from its retained queue:

* SIS-filtered: exclude the lowest SIS requests.
* Cook-filtered: exclude the lowest Cook's-score requests.
* Random-filtered: randomly exclude the same number as SIS.

For SIS and Cook's, the exclusion count is based on that method's independently
measured scoring and sorting cost:

    K_method = floor(
        score_cost_multiplier * priority_cost_method / average_unlearn_cost
    )

Random uses K_random = K_SIS, pays no scoring cost, and serves as a matched
filtering control. Excluded requests are not unlearned; they remain pending.
The fully-deleted target removes the entire original queue.
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


METHODS = ("sis_filtered", "cooks_filtered", "random_filtered")
SCORED_FILTERS = {
    "sis_filtered": "sis",
    "cooks_filtered": "cooks",
}


def timed_ascending_order(
    model: RLS,
    queue: np.ndarray,
    score_method: str,
    timing_reps: int,
) -> tuple[np.ndarray, float, float]:
    """Return queue positions in ascending score order and measured costs."""
    scores, score_sec = median_timed_call(
        lambda: scores_for_queue(model, queue, score_method), timing_reps
    )
    ascending_positions, sort_sec = median_timed_call(
        lambda: np.argsort(scores, kind="stable"), timing_reps
    )
    return ascending_positions, score_sec, sort_sec


def evaluate_filtered_policy(
    model: RLS,
    deletion_order: np.ndarray,
    exclusion_count: int,
    priority_cost_sec: float,
    budgets_sec: np.ndarray,
    average_unlearn_sec: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w0: np.ndarray,
    w_target: np.ndarray,
) -> dict:
    """Evaluate a retained random queue under all equal-compute deadlines."""
    scoring_completed = budgets_sec >= priority_cost_sec
    remaining_budget = np.where(
        scoring_completed, budgets_sec - priority_cost_sec, 0.0
    )
    ks = np.minimum(
        len(deletion_order),
        np.floor(
            (remaining_budget + average_unlearn_sec * 1e-9)
            / average_unlearn_sec
        ).astype(int),
    )
    progress, mse = evaluate_prefixes(
        model,
        deletion_order,
        ks,
        X_test,
        y_test,
        w0,
        w_target,
    )
    charged = np.where(
        scoring_completed,
        priority_cost_sec + ks * average_unlearn_sec,
        budgets_sec,
    )
    return {
        "scoring_completed": scoring_completed.tolist(),
        "n_deleted": ks.tolist(),
        "n_pending_excluded": [int(exclusion_count)] * len(BUDGET_FRACTIONS),
        "charged_compute_sec": charged.tolist(),
        "progress": progress.tolist(),
        "test_mse": mse.tolist(),
    }


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

    # Warm-up is outside the measured budget.
    for score_method in SCORED_FILTERS.values():
        _ = scores_for_queue(model, queue, score_method)
    warm = model.copy()
    warm.unlearn(int(queue[0]))

    timings = {}
    ascending_orders = {}
    for policy, score_method in SCORED_FILTERS.items():
        positions, score_sec, sort_sec = timed_ascending_order(
            model, queue, score_method, timing_reps
        )
        ascending_orders[policy] = positions
        timings[policy] = {
            "score_sec": score_sec,
            "sort_sec": sort_sec,
            "priority_cost_sec": score_sec + sort_sec,
        }

    average_unlearn_sec = benchmark_average_unlearn_cost(
        model, queue, timing_reps
    )
    budgets_sec = BUDGET_FRACTIONS * Q * average_unlearn_sec

    exclusion_counts = {}
    score_equivalents = {}
    for policy in SCORED_FILTERS:
        equivalent = timings[policy]["priority_cost_sec"] / average_unlearn_sec
        requested = int(np.floor(score_cost_multiplier * equivalent))
        score_equivalents[policy] = equivalent
        exclusion_counts[policy] = min(Q - 1, max(0, requested))

    # Random-filtered is count-matched to SIS, as requested.
    exclusion_counts["random_filtered"] = exclusion_counts["sis_filtered"]

    rng = np.random.default_rng(seed)
    retained_orders = {}
    excluded_sets = {}

    for offset, policy in enumerate(SCORED_FILTERS):
        k = exclusion_counts[policy]
        positions = ascending_orders[policy]
        excluded = queue[positions[:k]]
        retained = queue[positions[k:]]
        excluded_sets[policy] = excluded
        retained_orders[policy] = np.random.default_rng(
            seed + 100 + offset
        ).permutation(retained)

    random_permutation = rng.permutation(queue)
    random_k = exclusion_counts["random_filtered"]
    excluded_sets["random_filtered"] = random_permutation[:random_k]
    retained_orders["random_filtered"] = np.random.default_rng(
        seed + 200
    ).permutation(random_permutation[random_k:])

    out = {
        "queue_size": Q,
        "score_cost_multiplier": score_cost_multiplier,
        "average_unlearn_sec": average_unlearn_sec,
        "budgets_sec": budgets_sec.tolist(),
        "timing": timings,
        "score_deletion_equivalent": score_equivalents,
        "exclusion_count": exclusion_counts,
        "eligible_count": {
            policy: int(len(retained_orders[policy])) for policy in METHODS
        },
        "methods": {},
    }

    for policy in METHODS:
        overhead = (
            timings[policy]["priority_cost_sec"]
            if policy in SCORED_FILTERS
            else 0.0
        )
        out["methods"][policy] = evaluate_filtered_policy(
            model=model,
            deletion_order=retained_orders[policy],
            exclusion_count=exclusion_counts[policy],
            priority_cost_sec=overhead,
            budgets_sec=budgets_sec,
            average_unlearn_sec=average_unlearn_sec,
            X_test=X_test,
            y_test=y_test,
            w0=w0,
            w_target=w_target,
        )

        # Audits: excluded requests never enter unlearn order; retained order
        # contains no duplicate request.
        if np.intersect1d(
            excluded_sets[policy], retained_orders[policy]
        ).size:
            raise AssertionError(f"{policy}: excluded request entered deletion order")
        if len(np.unique(retained_orders[policy])) != len(retained_orders[policy]):
            raise AssertionError(f"{policy}: duplicate deletion request")

    if exclusion_counts["random_filtered"] != exclusion_counts["sis_filtered"]:
        raise AssertionError("random exclusion count is not matched to SIS")
    return out


def mean_std(values: np.ndarray) -> tuple[object, object]:
    """Return scalar or list mean/std in a JSON-friendly form."""
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    if mean.ndim == 0:
        return float(mean), float(std)
    return mean.tolist(), std.tolist()


def aggregate_realizations(realizations: list[dict]) -> dict:
    out = {
        "n_repeats": len(realizations),
        "budget_fraction": BUDGET_FRACTIONS.tolist(),
        "queue_size_mean": float(np.mean([r["queue_size"] for r in realizations])),
        "score_cost_multiplier": float(realizations[0]["score_cost_multiplier"]),
    }

    for metric in ("average_unlearn_sec", "budgets_sec"):
        values = np.asarray([r[metric] for r in realizations], dtype=float)
        mean, std = mean_std(values)
        output_name = "budget_sec" if metric == "budgets_sec" else metric
        out[f"{output_name}_mean"] = mean
        out[f"{output_name}_std"] = std

    out["timing"] = {}
    for policy in SCORED_FILTERS:
        out["timing"][policy] = {}
        for metric in ("score_sec", "sort_sec", "priority_cost_sec"):
            values = np.asarray(
                [r["timing"][policy][metric] for r in realizations], dtype=float
            )
            mean, std = mean_std(values)
            out["timing"][policy][f"{metric}_mean"] = mean
            out["timing"][policy][f"{metric}_std"] = std

    out["filtering"] = {}
    for policy in METHODS:
        out["filtering"][policy] = {}
        for metric in ("exclusion_count", "eligible_count"):
            values = np.asarray(
                [r[metric][policy] for r in realizations], dtype=float
            )
            mean, std = mean_std(values)
            out["filtering"][policy][f"{metric}_mean"] = mean
            out["filtering"][policy][f"{metric}_std"] = std
        if policy in SCORED_FILTERS:
            values = np.asarray(
                [
                    r["score_deletion_equivalent"][policy]
                    for r in realizations
                ],
                dtype=float,
            )
            mean, std = mean_std(values)
            out["filtering"][policy]["score_deletion_equivalent_mean"] = mean
            out["filtering"][policy]["score_deletion_equivalent_std"] = std

    out["methods"] = {}
    for policy in METHODS:
        out["methods"][policy] = {}
        for metric in (
            "scoring_completed",
            "n_deleted",
            "n_pending_excluded",
            "charged_compute_sec",
            "progress",
            "test_mse",
        ):
            values = np.asarray(
                [r["methods"][policy][metric] for r in realizations], dtype=float
            )
            mean, std = mean_std(values)
            out["methods"][policy][f"{metric}_mean"] = mean
            out["methods"][policy][f"{metric}_std"] = std
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
        default=Path("criteo_sis_cooks_random_filter_results.json"),
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
        "experiment": "criteo_sis_cooks_random_filtered_equal_compute_budget",
        "source_url": OFFICIAL_URL,
        "official_total_rows": EXPECTED_TOTAL_ROWS,
        "policies": {
            "sis_filtered": (
                "exclude bottom SIS requests, then randomly delete retained requests"
            ),
            "cooks_filtered": (
                "exclude bottom Cook's requests, then randomly delete retained requests"
            ),
            "random_filtered": (
                "randomly exclude the SIS-matched count, then randomly delete retained requests"
            ),
        },
        "budget_definition": (
            "fraction of measured wall-clock cost of deleting the entire original "
            "queue with no scoring; SIS and Cook scoring/sorting are charged"
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