# Create plots and a CSV table for the equal-compute-budget experiment

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("sis", "fifo", "random")
LABELS = {"sis": "SIS", "fifo": "FIFO", "random": "Random"}
COLORS = {"sis": "#d62728", "fifo": "#1f77b4", "random": "#7f7f7f"}
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "diabetes": "Diabetes",
    "large_synthetic_n2000_d26": "Large Synthetic",
}


def add_curve(ax, x, mean, std, method):
    mean, std = np.asarray(mean), np.asarray(std)
    ax.plot(x, mean, marker="o", lw=2, label=LABELS[method], color=COLORS[method])
    ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("equal_compute_budget_results.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    results = json.loads(args.input.read_text(encoding="utf-8"))
    datasets = results["datasets"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(15, 4.1 * len(datasets)), squeeze=False
    )
    rows = []
    for row_idx, (dataset_name, ds) in enumerate(datasets.items()):
        x = 100 * np.asarray(ds["budget_fraction"])
        for method in METHODS:
            m = ds["methods"][method]
            add_curve(
                axes[row_idx, 0], x, m["progress_mean"], m["progress_std"], method
            )
            add_curve(
                axes[row_idx, 1], x, m["test_mse_mean"], m["test_mse_std"], method
            )
            add_curve(
                axes[row_idx, 2], x, m["n_deleted_mean"], m["n_deleted_std"], method
            )

            for j, budget in enumerate(x):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "budget_pct": float(budget),
                        "method": LABELS[method],
                        "progress_mean": m["progress_mean"][j],
                        "progress_std": m["progress_std"][j],
                        "test_mse_mean": m["test_mse_mean"][j],
                        "test_mse_std": m["test_mse_std"][j],
                        "n_deleted_mean": m["n_deleted_mean"][j],
                        "n_deleted_std": m["n_deleted_std"][j],
                        "total_budget_sec_mean": ds["budget_sec_mean"][j],
                        "total_budget_sec_std": ds["budget_sec_std"][j],
                        "charged_compute_sec_mean": m["charged_compute_sec_mean"][j],
                        "charged_compute_sec_std": m["charged_compute_sec_std"][j],
                        "sis_score_sec_mean": ds["sis_score_sec_mean"],
                        "sis_sort_sec_mean": ds["sis_sort_sec_mean"],
                        "average_unlearn_sec_mean": ds["average_unlearn_sec_mean"],
                    }
                )

        axes[row_idx, 0].set_ylabel(
            f"{DATASET_LABELS.get(dataset_name, dataset_name)}\nProgress"
        )
        axes[row_idx, 1].set_ylabel("Held-out test MSE")
        axes[row_idx, 2].set_ylabel("Exact deletions completed")
        for ax in axes[row_idx]:
            ax.set_xlabel("Equal total wall-clock budget (%)")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)

    fig.suptitle(
        "SIS prioritization under an equal total compute budget\n"
        "Mean +/- 1 standard deviation across queue realizations",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output_dir / "equal_compute_budget.png", dpi=220)
    fig.savefig(args.output_dir / "equal_compute_budget.pdf", bbox_inches="tight")
    plt.close(fig)

    table_path = args.output_dir / "equal_compute_budget_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
