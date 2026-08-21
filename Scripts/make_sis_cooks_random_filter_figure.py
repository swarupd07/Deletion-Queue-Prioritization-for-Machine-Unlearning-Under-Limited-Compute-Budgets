"""Plot one SIS/Cook's/Random filtered equal-compute experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("sis_filtered", "cooks_filtered", "random_filtered")
LABELS = {
    "sis_filtered": "SIS-filtered",
    "cooks_filtered": "Cook's-filtered",
    "random_filtered": "Random-filtered",
}
COLORS = {
    "sis_filtered": "#d62728",
    "cooks_filtered": "#9467bd",
    "random_filtered": "#7f7f7f",
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
        default=Path("sis_cooks_random_filter_m5.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_sis_cooks_random_filter_m5"),
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
    prefix = args.prefix or f"sis_cooks_random_filter_m{multiplier_text}"
    x = 100.0 * np.asarray(ds["budget_fraction"], dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))

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

    sis_excluded = ds["filtering"]["sis_filtered"]["exclusion_count_mean"]
    cooks_excluded = ds["filtering"]["cooks_filtered"]["exclusion_count_mean"]
    random_excluded = ds["filtering"]["random_filtered"]["exclusion_count_mean"]
    queue_size = ds["queue_size_mean"]
    fig.suptitle(
        "Score-filtered random deletion under an equal compute budget\n"
        f"{dataset_name} | multiplier={multiplier_text} | queue={queue_size:,.0f} | "
        f"mean excluded: SIS={sis_excluded:,.1f}, "
        f"Cook's={cooks_excluded:,.1f}, Random={random_excluded:,.1f}",
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
        filtering = ds["filtering"][method]
        timing = ds["timing"].get(method, {})
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
                    "exclusion_count_mean": filtering["exclusion_count_mean"],
                    "exclusion_count_std": filtering["exclusion_count_std"],
                    "eligible_count_mean": filtering["eligible_count_mean"],
                    "score_sec_mean": timing.get("score_sec_mean", 0.0),
                    "sort_sec_mean": timing.get("sort_sec_mean", 0.0),
                    "priority_cost_sec_mean": timing.get(
                        "priority_cost_sec_mean", 0.0
                    ),
                    "score_deletion_equivalent_mean": filtering.get(
                        "score_deletion_equivalent_mean", 0.0
                    ),
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