"""Million-scale equal-compute-budget experiment on real Criteo logs.

The script downloads the official Criteo Attribution Modeling dataset, reads
it in chunks, and fits the same ridge solution as RLS using sufficient
statistics:

    P = (X.T @ X + lambda I)^-1
    w = P @ X.T @ y

This is mathematically identical to the final model produced by streaming RLS,
but avoids storing millions of Python objects. Only a uniformly sampled pool
of training rows is retained for deletion queues.

Target: click (0/1). Held-out MSE is therefore a Brier-style squared error for
the un-clipped linear-probability predictions, consistent with the existing
evaluation code.

Important: observations are real production impressions, but the deletion
request queues are simulated because the public dataset does not contain
actual deletion-request logs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rls_influence import RLS
from run_equal_compute_budget_all_methods import (
    aggregate_realizations,
    run_realization,
)


OFFICIAL_URL = (
    "https://criteostorage.blob.core.windows.net/criteo-research-datasets/"
    "criteo_attribution_dataset.zip"
)
EXPECTED_TOTAL_ROWS = 16_468_027
CATEGORICAL_COLUMNS = ["campaign", *[f"cat{i}" for i in range(1, 10)]]
REQUIRED_COLUMNS = {
    "timestamp",
    "uid",
    "campaign",
    "click",
    "cost",
    "time_since_last_click",
    *[f"cat{i}" for i in range(1, 10)],
}


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if total:
                print(
                    f"\rDownloaded {downloaded / 2**20:,.0f} / "
                    f"{total / 2**20:,.0f} MiB",
                    end="",
                    flush=True,
                )
    print()
    temporary.replace(destination)


def obtain_dataset(data_dir: Path, accept_license: bool) -> Path:
    """Download and extract the official compressed TSV if it is absent."""
    expected = data_dir / "criteo_attribution_dataset.tsv.gz"
    if expected.exists():
        return expected

    candidates = list(data_dir.rglob("criteo_attribution_dataset.tsv.gz"))
    if candidates:
        return candidates[0]

    if not accept_license:
        raise RuntimeError(
            "Dataset not found. Re-run with --accept-license after reviewing "
            "the Criteo CC BY-NC-SA 4.0 dataset terms: "
            "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/"
        )

    archive = data_dir / "criteo_attribution_dataset.zip"
    if not archive.exists():
        download_with_progress(OFFICIAL_URL, archive)

    print(f"Extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        members = [
            m for m in zf.infolist()
            if m.filename.endswith("criteo_attribution_dataset.tsv.gz")
        ]
        if len(members) != 1:
            raise RuntimeError(
                "Expected exactly one criteo_attribution_dataset.tsv.gz in archive"
            )
        expected.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(members[0]) as src, expected.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    return expected


def make_features(frame: pd.DataFrame, hash_dim: int) -> np.ndarray:
    """Create bounded dense features using deterministic signed hashing."""
    n = len(frame)
    # intercept + categorical hash buckets + four bounded numeric features
    X = np.zeros((n, 1 + hash_dim + 4), dtype=np.float64)
    X[:, 0] = 1.0
    rows = np.arange(n)

    for column_number, column in enumerate(CATEGORICAL_COLUMNS, start=1):
        values = frame[column].fillna(-1).to_numpy(dtype=np.int64).astype(np.uint64)
        salt = np.uint64(column_number * 2_654_435_761)
        mixed = values * np.uint64(11_400_714_819_323_198_485) + salt
        mixed ^= mixed >> np.uint64(33)
        buckets = (mixed % np.uint64(hash_dim)).astype(np.int64)
        signs = np.where((mixed & np.uint64(1)) == 0, 1.0, -1.0)
        np.add.at(X, (rows, 1 + buckets), signs / np.sqrt(len(CATEGORICAL_COLUMNS)))

    numeric_start = 1 + hash_dim
    timestamp = np.maximum(frame["timestamp"].to_numpy(dtype=float), 0.0)
    cost = np.maximum(frame["cost"].to_numpy(dtype=float), 0.0)
    since_click_raw = frame["time_since_last_click"].to_numpy(dtype=float)
    since_click = np.maximum(since_click_raw, 0.0)

    X[:, numeric_start] = np.log1p(timestamp) / 16.0
    X[:, numeric_start + 1] = np.log1p(cost * 1_000_000.0) / 12.0
    X[:, numeric_start + 2] = np.log1p(since_click) / 16.0
    X[:, numeric_start + 3] = (since_click_raw < 0).astype(float)
    return X


def update_uniform_pool(
    pool_X: np.ndarray | None,
    pool_y: np.ndarray | None,
    pool_keys: np.ndarray | None,
    X: np.ndarray,
    y: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
):
    """Keep the rows with the smallest independent random keys."""
    keys = rng.random(len(X))
    if pool_X is None:
        combined_X, combined_y, combined_keys = X, y, keys
    else:
        combined_X = np.vstack([pool_X, X])
        combined_y = np.concatenate([pool_y, y])
        combined_keys = np.concatenate([pool_keys, keys])

    if len(combined_keys) > capacity:
        keep = np.argpartition(combined_keys, capacity - 1)[:capacity]
        combined_X = combined_X[keep]
        combined_y = combined_y[keep]
        combined_keys = combined_keys[keep]
    return combined_X, combined_y, combined_keys


def build_large_rls_state(
    data_path: Path,
    max_rows: int,
    train_fraction: float,
    chunksize: int,
    hash_dim: int,
    lam: float,
    queue_pool_size: int,
    max_test_rows: int,
    seed: int,
):
    if max_rows <= 1_000_000:
        raise ValueError("Use --max-rows greater than 1,000,000 for this experiment")
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("--train-fraction must be in [0.5, 1.0)")

    train_limit = int(max_rows * train_fraction)
    dim = 1 + hash_dim + 4
    A = lam * np.eye(dim, dtype=np.float64)
    b = np.zeros(dim, dtype=np.float64)
    pool_X = pool_y = pool_keys = None
    X_test_parts, y_test_parts = [], []
    n_test_kept = 0
    rows_seen = 0
    rng = np.random.default_rng(seed)

    reader = pd.read_csv(
        data_path,
        sep="\t",
        compression="infer",
        chunksize=chunksize,
        nrows=max_rows,
        low_memory=False,
    )

    for chunk_number, frame in enumerate(reader, start=1):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

        chunk_start = rows_seen
        chunk_end = rows_seen + len(frame)
        train_count = max(0, min(chunk_end, train_limit) - chunk_start)

        if train_count:
            train_frame = frame.iloc[:train_count]
            X_train = make_features(train_frame, hash_dim)
            y_train = train_frame["click"].to_numpy(dtype=np.float64)
            A += X_train.T @ X_train
            b += X_train.T @ y_train
            pool_X, pool_y, pool_keys = update_uniform_pool(
                pool_X,
                pool_y,
                pool_keys,
                X_train,
                y_train,
                queue_pool_size,
                rng,
            )

        if train_count < len(frame) and n_test_kept < max_test_rows:
            test_frame = frame.iloc[train_count:]
            remaining = max_test_rows - n_test_kept
            test_frame = test_frame.iloc[:remaining]
            X_part = make_features(test_frame, hash_dim)
            y_part = test_frame["click"].to_numpy(dtype=np.float64)
            X_test_parts.append(X_part)
            y_test_parts.append(y_part)
            n_test_kept += len(test_frame)

        rows_seen = chunk_end
        print(
            f"\rProcessed {rows_seen:,}/{max_rows:,} rows",
            end="",
            flush=True,
        )
    print()

    if rows_seen < max_rows:
        raise ValueError(
            f"Requested {max_rows:,} rows but the file contained only {rows_seen:,}"
        )
    if pool_X is None or len(pool_X) < queue_pool_size:
        raise RuntimeError("Could not construct the requested deletion pool")
    if not X_test_parts:
        raise RuntimeError("No held-out observations were retained")

    P = np.linalg.inv(A)
    w = P @ b
    model = RLS(dim=dim, lam=lam)
    model.P = P
    model.w = w
    model.n_seen = train_limit
    model.X_hist = [row.copy() for row in pool_X]
    model.y_hist = pool_y.tolist()
    X_test = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)
    return model, X_test, y_test, train_limit, rows_seen - train_limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/criteo"))
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Existing .tsv or .tsv.gz; skips automatic download",
    )
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--hash-dim", type=int, default=32)
    parser.add_argument("--lambda-ridge", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=500)
    parser.add_argument("--queue-pool-size", type=int, default=5_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-reps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("criteo_equal_compute_budget_results.json"),
    )
    args = parser.parse_args()

    if args.queue_size > args.queue_pool_size:
        raise ValueError("--queue-size cannot exceed --queue-pool-size")
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
                model,
                X_test,
                y_test,
                queue,
                seed=args.seed + 20_000 + rep,
                timing_reps=args.timing_reps,
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
        "experiment": "criteo_equal_total_wall_clock_budget_all_methods",
        "source_url": OFFICIAL_URL,
        "official_total_rows": EXPECTED_TOTAL_ROWS,
        "budget_definition": (
            "fraction of measured wall-clock cost of deleting the entire queue "
            "with no scoring; evaluation and target construction excluded"
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