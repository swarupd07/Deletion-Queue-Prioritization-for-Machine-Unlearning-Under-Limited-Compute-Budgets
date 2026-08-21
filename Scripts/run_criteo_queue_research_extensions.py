"""Run queue-research extensions using an exported Criteo RLS state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from queue_research_extensions import run_repeated_study
from rls_influence import RLS


def load_rls_state(path: Path):
    data = np.load(path, allow_pickle=False)

    required = {
        "P",
        "w",
        "X_history",
        "y_history",
        "X_test",
        "y_test",
        "lam",
        "n_seen",
    }
    missing = required.difference(data.files)

    if missing:
        raise ValueError(f"RLS-state NPZ is missing arrays: {sorted(missing)}")

    X_history = np.asarray(data["X_history"])
    y_history = np.asarray(data["y_history"])
    X_test = np.asarray(data["X_test"])
    y_test = np.asarray(data["y_test"])

    if len(X_history) != len(y_history):
        raise ValueError("X_history and y_history have different lengths")

    model = RLS(
        dim=X_history.shape[1],
        lam=float(np.asarray(data["lam"]).item()),
    )
    model.P = np.asarray(data["P"], dtype=np.float64).copy()
    model.w = np.asarray(data["w"], dtype=np.float64).copy()
    model.X_hist = X_history
    model.y_hist = y_history
    model.n_seen = int(np.asarray(data["n_seen"]).item())

    return model, X_test, y_test


def flatten_statistics(result, key, metric_name):
    rows = []
    budgets = result["budget_fractions"]

    for record in result[key]:
        row = dict(record)
        row["metric"] = metric_name
        row["budget_fraction"] = budgets[row.pop("budget_index")]
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=("rls-state",),
        default="rls-state",
    )
    parser.add_argument(
        "--npz-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--queue-size", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-reps", type=int, default=11)
    parser.add_argument("--oracle-queue-size", type=int, default=30)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
    )
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("criteo_queue_extensions.json"),
    )

    args = parser.parse_args()

    model, X_test, y_test = load_rls_state(args.npz_file)
    queue_pool_size = len(model.X_hist)

    if args.queue_size > queue_pool_size:
        raise ValueError(
            f"queue-size {args.queue_size} exceeds "
            f"stored queue pool size {queue_pool_size}"
        )

    def queue_builder(rep):
        rng = np.random.default_rng(args.seed + rep)
        return rng.choice(
            queue_pool_size,
            size=args.queue_size,
            replace=False,
        )

    result = run_repeated_study(
        model=model,
        X_test=X_test,
        y_test=y_test,
        queue_builder=queue_builder,
        repeats=args.repeats,
        budget_fractions=args.budgets,
        timing_reps=args.timing_reps,
        oracle_queue_size=args.oracle_queue_size,
        seed=args.seed,
    )

    result.update(
        {
            "dataset": "rls-state",
            "n_train": int(model.n_seen),
            "queue_pool_size": int(queue_pool_size),
            "n_test": int(len(X_test)),
            "dimension": int(model.dim),
            "queue_size": int(args.queue_size),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = args.output.with_name(
        args.output.stem + "_paired_tests.csv"
    )

    rows = (
        flatten_statistics(
            result,
            "paired_progress_vs_random",
            "progress",
        )
        + flatten_statistics(
            result,
            "paired_mse_vs_random",
            "test_mse",
        )
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {args.output}")
    print(f"Saved {csv_path}")

    print("\nMean progress at each equal-compute budget:")
    for method, values in result["methods"].items():
        formatted = ", ".join(
            f"{100 * value:.2f}%"
            for value in values["progress_mean"]
        )
        print(f"  {method:9s}: {formatted}")


if __name__ == "__main__":
    main()
