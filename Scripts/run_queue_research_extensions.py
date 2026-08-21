"""Runnable example for queue_research_extensions.py.

Examples (PowerShell uses the backtick for line continuation):

    python run_queue_research_extensions.py --dataset diabetes --repeats 30

    python run_queue_research_extensions.py `
      --dataset npz `
      --npz-file criteo_preprocessed.npz `
      --queue-size 20000 `
      --repeats 30 `
      --timing-reps 11

The NPZ file must contain X_train, y_train, X_test, and y_test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from queue_research_extensions import run_repeated_study
from rls_influence import RLS


def load_data(args):
    if args.dataset == "npz":
        if args.npz_file is None:
            raise ValueError("--npz-file is required when --dataset npz")
        data = np.load(args.npz_file, mmap_mode="r")
        required = {"X_train", "y_train", "X_test", "y_test"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing arrays: {sorted(missing)}")
        return tuple(np.asarray(data[k]) for k in ("X_train", "X_test", "y_train", "y_test"))

    data = load_diabetes()
    X = StandardScaler().fit_transform(data.data)
    X = np.column_stack([np.ones(len(X)), X])
    y = (data.target - data.target.mean()) / data.target.std()
    return train_test_split(X, y, test_size=0.30, random_state=0)


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
    parser.add_argument("--dataset", choices=("diabetes", "npz"), default="diabetes")
    parser.add_argument("--npz-file", type=Path)
    parser.add_argument("--queue-size", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-reps", type=int, default=11)
    parser.add_argument("--lambda-ridge", type=float, default=1.0)
    parser.add_argument("--oracle-queue-size", type=int, default=30)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--output", type=Path, default=Path("queue_research_extensions.json"))
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args)
    if args.queue_size > len(X_train):
        raise ValueError(f"queue-size {args.queue_size} exceeds n_train={len(X_train)}")

    model = RLS(dim=X_train.shape[1], lam=args.lambda_ridge)
    model.fit_stream(X_train, y_train)

    def queue_builder(rep):
        rng = np.random.default_rng(args.seed + rep)
        return rng.choice(len(X_train), size=args.queue_size, replace=False)

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
    result["dataset"] = args.dataset
    result["n_train"] = int(len(X_train))
    result["n_test"] = int(len(X_test))
    result["dimension"] = int(X_train.shape[1])
    result["queue_size"] = int(args.queue_size)

    args.output.write_text(json.dumps(result, indent=2))
    csv_path = args.output.with_name(args.output.stem + "_paired_tests.csv")
    rows = (
        flatten_statistics(result, "paired_progress_vs_random", "progress")
        + flatten_statistics(result, "paired_mse_vs_random", "test_mse")
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {args.output}")
    print(f"Saved {csv_path}")
    print("\nMean progress at each equal-compute budget:")
    for method, values in result["methods"].items():
        formatted = ", ".join(f"{100*x:.2f}%" for x in values["progress_mean"])
        print(f"  {method:9s}: {formatted}")


if __name__ == "__main__":
    main()