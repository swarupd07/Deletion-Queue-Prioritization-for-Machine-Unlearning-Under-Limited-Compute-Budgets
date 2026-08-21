"""Plot one SIS-filtered-random equal-compute-budget experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("sis_filtered_random", "random")
LABELS = {
    "sis_filtered_random": "SIS-filtered random",
    "random": "Random",
}
COLORS = {
    "sis_filtered_random": "#d62728",
    "random": "#7f7f7f",
}


def add_curve(ax, x, mean, std, method):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2.2,
        color=COLORS[method],
        label=LABELS[method],
    )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=COLORS[method],
        alpha=0.18,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sis_filtered_random_m20.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_sis_filtered_random_m20"),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix; default is inferred from the multiplier",
    )
    args = parser.parse_args()

    results = json.loads(args.input.read_text(encoding="utf-8"))
    dataset_name, ds = next(iter(results["datasets"].items()))
    multiplier = float(ds["score_cost_multiplier"])
    multiplier_text = f"{multiplier:g}"
    prefix = args.prefix or f"sis_filtered_random_m{multiplier_text}"
    x = 100.0 * np.asarray(ds["budget_fraction"], dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    for method in METHODS:
        values = ds["methods"][method]
        add_curve(
            axes[0],
            x,
            values["progress_mean"],
            values["progress_std"],
            method,
        )
        add_curve(
            axes[1],
            x,
            values["test_mse_mean"],
            values["test_mse_std"],
            method,
        )
        add_curve(
            axes[2],
            x,
            values["n_deleted_mean"],
            values["n_deleted_std"],
            method,
        )

    axes[0].set_ylabel("Criterion attribution progress")
    axes[1].set_ylabel("Held-out test MSE")
    axes[2].set_ylabel("Exact deletions completed")
    for ax in axes:
        ax.set_xlabel("Equal total wall-clock budget (%)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    excluded_mean = ds["exclusion_count_mean"]
    excluded_std = ds["exclusion_count_std"]
    equivalent_mean = ds["score_deletion_equivalent_mean"]
    queue_size = ds["queue_size_mean"]
    fig.suptitle(
        "SIS-filtered random deletion under an equal compute budget\n"
        f"{dataset_name} | multiplier={multiplier_text} | "
        f"queue={queue_size:,.0f} | excluded={excluded_mean:,.1f} "
        f"+/- {excluded_std:,.1f} | score cost={equivalent_mean:,.1f} "
        "deletion-equivalents",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    png_path = args.output_dir / f"{prefix}.png"
    pdf_path = args.output_dir / f"{prefix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for method in METHODS:
        values = ds["methods"][method]
        for j, budget_pct in enumerate(x):
            rows.append(
                {
                    "dataset": dataset_name,
                    "score_cost_multiplier": multiplier,
                    "budget_pct": float(budget_pct),
                    "method": LABELS[method],
                    "progress_mean": values["progress_mean"][j],
                    "progress_std": values["progress_std"][j],
                    "test_mse_mean": values["test_mse_mean"][j],
                    "test_mse_std": values["test_mse_std"][j],
                    "n_deleted_mean": values["n_deleted_mean"][j],
                    "n_deleted_std": values["n_deleted_std"][j],
                    "pending_excluded_mean": values[
                        "n_pending_excluded_mean"
                    ][j],
                    "pending_excluded_std": values[
                        "n_pending_excluded_std"
                    ][j],
                    "charged_compute_sec_mean": values[
                        "charged_compute_sec_mean"
                    ][j],
                    "charged_compute_sec_std": values[
                        "charged_compute_sec_std"
                    ][j],
                    "total_budget_sec_mean": ds["budget_sec_mean"][j],
                    "total_budget_sec_std": ds["budget_sec_std"][j],
                    "sis_score_sec_mean": ds["sis_score_sec_mean"],
                    "sis_sort_sec_mean": ds["sis_sort_sec_mean"],
                    "score_deletion_equivalent_mean": ds[
                        "score_deletion_equivalent_mean"
                    ],
                    "exclusion_count_mean": ds["exclusion_count_mean"],
                    "eligible_count_mean": ds["eligible_count_mean"],
                }
            )

    csv_path = args.output_dir / f"{prefix}_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()