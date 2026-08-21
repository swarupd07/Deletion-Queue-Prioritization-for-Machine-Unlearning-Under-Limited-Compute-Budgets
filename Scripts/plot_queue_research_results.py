'''"""Plot queue-research extension JSON files.

Creates, for every input JSON:
  * <name>_overview.{png,pdf}
  * <name>_staleness.{png,pdf}
  * <name>_statistics.{png,pdf}

When two or more Criteo queue sizes are supplied, also creates:
  * criteo_queue_scaling.{png,pdf}

PowerShell example:

python plot_queue_research_results.py `
  diabetes_queue_extensions.json `
  criteo_queue_test.json `
  criteo_queue_2000_validation.json `
  criteo_queue_20000_final.json `
  --output-dir queue_research_figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["sis", "cooks", "leverage", "residual", "qds", "random"]
METHOD_LABEL = {
    "sis": "SIS",
    "cooks": "Cook's",
    "leverage": "Leverage",
    "residual": "Residual",
    "qds": "QDS",
    "random": "Random",
}
COLORS = {
    "sis": "#d62728",
    "cooks": "#9467bd",
    "leverage": "#2ca02c",
    "residual": "#ff7f0e",
    "qds": "#1f77b4",
    "random": "#7f7f7f",
}


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    required = {"budget_fractions", "methods"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return result


def available_methods(result: dict) -> list[str]:
    return [method for method in METHOD_ORDER if method in result["methods"]]


def display_title(path: Path, result: dict) -> str:
    dataset = str(result.get("dataset", "dataset")).replace("rls-state", "Criteo")
    queue_size = result.get("queue_size")
    if queue_size is None and result.get("repetitions"):
        queue_size = result["repetitions"][0].get("queue_size")
    return f"{dataset.title()} — queue size {queue_size:,}" if queue_size else dataset.title()


def save_both(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def line_with_band(ax, x, mean, std, method, percentage=False) -> None:
    scale = 100.0 if percentage else 1.0
    mean = np.asarray(mean, dtype=float) * scale
    std = np.asarray(std, dtype=float) * scale
    ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        markersize=4,
        color=COLORS[method],
        label=METHOD_LABEL[method],
    )
    if len(std) == len(mean) and np.all(np.isfinite(std)):
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.12)


def plot_overview(path: Path, result: dict, output_dir: Path) -> None:
    budgets = 100.0 * np.asarray(result["budget_fractions"], dtype=float)
    methods = available_methods(result)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(display_title(path, result), fontsize=17, fontweight="bold")

    # A: Progress. Symmetric log retains both ordinary 0--100% progress and
    # very negative values such as -38,000% without clipping or hiding them.
    ax = axes[0, 0]
    for method in methods:
        values = result["methods"][method]
        line_with_band(
            ax, budgets, values["progress_mean"], values["progress_std"], method, True
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(100.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("symlog", linthresh=100.0, linscale=1.0)
    ax.set_title("Progress toward fully deleted target")
    ax.set_ylabel("Progress (%) — symmetric-log scale")

    # B: Held-out MSE.
    ax = axes[0, 1]
    for method in methods:
        values = result["methods"][method]
        line_with_band(
            ax, budgets, values["test_mse_mean"], values["test_mse_std"], method
        )
    ax.set_title("Held-out test MSE")
    ax.set_ylabel("MSE")

    # C: Exact deletions completed.
    ax = axes[0, 2]
    for method in methods:
        values = result["methods"][method]
        line_with_band(
            ax,
            budgets,
            values["deletions_mean"],
            values["deletions_std"],
            method,
        )
    ax.set_title("Exact deletions completed")
    ax.set_ylabel("Deletion count")

    # D: Scoring/sorting cost in exact-deletion equivalents.
    ax = axes[1, 0]
    costs = [
        result["methods"][m].get("score_cost_deletion_equivalents_mean", 0.0)
        for m in methods
    ]
    ax.bar([METHOD_LABEL[m] for m in methods], costs, color=[COLORS[m] for m in methods])
    ax.set_title("Priority cost")
    ax.set_ylabel("Exact-deletion equivalents")
    ax.tick_params(axis="x", rotation=35)

    mechanism = result.get("mechanism_summary", {})

    # E: Alignment with actual fully-deleted direction (diagnostic only).
    ax = axes[1, 1]
    mech_methods = [m for m in methods if m in mechanism]
    alignment = [mechanism[m]["mean_target_cosine"]["mean"] for m in mech_methods]
    alignment_std = [mechanism[m]["mean_target_cosine"]["std"] for m in mech_methods]
    ax.bar(
        [METHOD_LABEL[m] for m in mech_methods],
        alignment,
        yerr=alignment_std,
        capsize=3,
        color=[COLORS[m] for m in mech_methods],
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Top-25% update alignment")
    ax.set_ylabel("Mean cosine with target direction")
    ax.tick_params(axis="x", rotation=35)

    # F: Cancellation ratio; lower values mean stronger cancellation.
    ax = axes[1, 2]
    cancellation = [mechanism[m]["cancellation_ratio"]["mean"] for m in mech_methods]
    cancellation_std = [mechanism[m]["cancellation_ratio"]["std"] for m in mech_methods]
    ax.bar(
        [METHOD_LABEL[m] for m in mech_methods],
        cancellation,
        yerr=cancellation_std,
        capsize=3,
        color=[COLORS[m] for m in mech_methods],
    )
    ax.set_title("Top-25% cancellation ratio")
    ax.set_ylabel(r"$\|\sum_i \Delta\theta_i\| / \sum_i\|\Delta\theta_i\|$")
    ax.tick_params(axis="x", rotation=35)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        if ax not in axes[1, :]:
            ax.set_xlabel("Equal total compute budget (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_both(fig, output_dir / f"{path.stem}_overview")


def checkpoint_percent(key: str) -> float:
    match = re.search(r"rho_after_([0-9.]+)", key)
    return 100.0 * float(match.group(1)) if match else math.nan


def plot_staleness(path: Path, result: dict, output_dir: Path) -> None:
    staleness = result.get("staleness_summary", {})
    if not staleness:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHOD_ORDER:
        if method not in staleness:
            continue
        items = sorted(staleness[method].items(), key=lambda item: checkpoint_percent(item[0]))
        x = np.asarray([checkpoint_percent(k) for k, _ in items])
        mean = np.asarray([v["mean"] for _, v in items])
        std = np.asarray([v["std"] for _, v in items])
        ax.plot(x, mean, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABEL[method])
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.12)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Queue deleted using initial static ranking (%)")
    ax.set_ylabel("Spearman correlation: initial vs current scores")
    ax.set_title(f"Ranking staleness\n{display_title(path, result)}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    save_both(fig, output_dir / f"{path.stem}_staleness")


def plot_statistics(path: Path, result: dict, output_dir: Path) -> None:
    records = result.get("paired_progress_vs_random", [])
    if not records:
        return
    budgets = 100.0 * np.asarray(result["budget_fractions"], dtype=float)
    methods = [m for m in METHOD_ORDER if m != "random" and any(r["method"] == m for r in records)]
    matrix = np.full((len(methods), len(budgets)), np.nan)
    significant = np.zeros_like(matrix, dtype=bool)
    for r in records:
        if r["method"] not in methods:
            continue
        i = methods.index(r["method"])
        j = int(r["budget_index"])
        matrix[i, j] = 100.0 * float(r["mean_difference"])
        significant[i, j] = float(r.get("wilcoxon_p_holm", 1.0)) < 0.05

    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(budgets)):
            value = matrix[i, j]
            if np.isfinite(value):
                star = "*" if significant[i, j] else ""
                ax.text(j, i, f"{value:.1f}{star}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(budgets)), [f"{x:g}%" for x in budgets])
    ax.set_yticks(range(len(methods)), [METHOD_LABEL[m] for m in methods])
    ax.set_xlabel("Equal total compute budget")
    ax.set_title(f"Paired progress difference versus random\n{display_title(path, result)}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean progress difference (percentage points)")
    fig.text(0.5, 0.01, "* Holm-adjusted paired Wilcoxon p < 0.05", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_both(fig, output_dir / f"{path.stem}_statistics")


def is_criteo(result: dict, path: Path) -> bool:
    return result.get("dataset") == "rls-state" or "criteo" in path.stem.lower()


def queue_size(result: dict) -> int:
    if "queue_size" in result:
        return int(result["queue_size"])
    return int(result["repetitions"][0]["queue_size"])


def plot_criteo_scaling(items: list[tuple[Path, dict]], output_dir: Path) -> None:
    items = [(p, r) for p, r in items if is_criteo(r, p)]
    distinct = sorted({queue_size(r) for _, r in items})
    if len(distinct) < 2:
        return
    # If the same queue size occurs more than once, keep the file with more repeats.
    by_size = {}
    for path, result in items:
        q = queue_size(result)
        if q not in by_size or result.get("repeats", 0) > by_size[q][1].get("repeats", 0):
            by_size[q] = (path, result)
    queue_sizes = sorted(by_size)
    budgets = np.asarray(next(iter(by_size.values()))[1]["budget_fractions"], dtype=float)
    ncols = 3
    nrows = int(math.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.3 * nrows), squeeze=False)
    for budget_idx, budget in enumerate(budgets):
        ax = axes.flat[budget_idx]
        for method in METHOD_ORDER:
            y = []
            valid_x = []
            for q in queue_sizes:
                result = by_size[q][1]
                if method in result["methods"]:
                    valid_x.append(q)
                    y.append(100.0 * result["methods"][method]["progress_mean"][budget_idx])
            ax.plot(valid_x, y, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABEL[method])
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=100.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Budget {100 * budget:g}%")
        ax.set_xlabel("Queue size (log scale)")
        ax.set_ylabel("Progress (%) — symmetric-log")
        ax.grid(alpha=0.25)
    for index in range(len(budgets), nrows * ncols):
        axes.flat[index].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("Criteo queue-size scaling under equal compute budgets", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_both(fig, output_dir / "criteo_queue_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("queue_research_figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = [(path, load_result(path)) for path in args.results]
    for path, result in items:
        plot_overview(path, result, args.output_dir)
        plot_staleness(path, result, args.output_dir)
        plot_statistics(path, result, args.output_dir)
    plot_criteo_scaling(items, args.output_dir)
    print(f"Saved figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
'''


"""Plot queue-research extension JSON files.

Creates, for every input JSON:
  * <name>_overview.{png,pdf}
  * <name>_staleness.{png,pdf}
  * <name>_statistics.{png,pdf}

When two or more Criteo queue sizes are supplied, also creates:
  * criteo_queue_scaling.{png,pdf}

PowerShell example:

python plot_queue_research_results.py `
  diabetes_queue_extensions.json `
  criteo_queue_test.json `
  criteo_queue_2000_validation.json `
  criteo_queue_20000_final.json `
  --output-dir queue_research_figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["sis", "cooks", "leverage", "residual", "qds", "random"]
METHOD_LABEL = {
    "sis": "SIS",
    "cooks": "Cook's",
    "leverage": "Leverage",
    "residual": "Residual",
    "qds": "QDS",
    "random": "Random",
}
COLORS = {
    "sis": "#d62728",
    "cooks": "#9467bd",
    "leverage": "#2ca02c",
    "residual": "#ff7f0e",
    "qds": "#1f77b4",
    "random": "#7f7f7f",
}


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    required = {"budget_fractions", "methods"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return result


def available_methods(result: dict) -> list[str]:
    return [method for method in METHOD_ORDER if method in result["methods"]]


def display_title(path: Path, result: dict) -> str:
    dataset = str(result.get("dataset", "dataset")).replace("rls-state", "Criteo")
    queue_size = result.get("queue_size")
    if queue_size is None and result.get("repetitions"):
        queue_size = result["repetitions"][0].get("queue_size")
    return f"{dataset.title()} — queue size {queue_size:,}" if queue_size else dataset.title()


def save_both(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def line_with_band(ax, x, mean, std, method, percentage=False) -> None:
    scale = 100.0 if percentage else 1.0
    mean = np.asarray(mean, dtype=float) * scale
    std = np.asarray(std, dtype=float) * scale
    ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        markersize=4,
        color=COLORS[method],
        label=METHOD_LABEL[method],
    )
    if len(std) == len(mean) and np.all(np.isfinite(std)):
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.12)


def plot_overview(path: Path, result: dict, output_dir: Path) -> None:
    budgets = 100.0 * np.asarray(result["budget_fractions"], dtype=float)
    methods = available_methods(result)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(display_title(path, result), fontsize=17, fontweight="bold")

    # A: Progress. Symmetric log retains both ordinary 0--100% progress and
    # very negative values such as -38,000% without clipping or hiding them.
    ax = axes[0, 0]
    for method in methods:
        values = result["methods"][method]
        line_with_band(
            ax, budgets, values["progress_mean"], values["progress_std"], method, True
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(100.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("symlog", linthresh=100.0, linscale=1.0)
    ax.set_title("Progress toward fully deleted target")
    ax.set_ylabel("Progress (%) — symmetric-log scale")

    # B: Held-out MSE.
    ax = axes[0, 1]
    for method in methods:
        values = result["methods"][method]
        line_with_band(
            ax, budgets, values["test_mse_mean"], values["test_mse_std"], method
        )
    ax.set_title("Held-out test MSE")
    ax.set_ylabel("MSE")

    # C: Exact deletions completed.
    ax = axes[0, 2]
    for method in methods:
        values = result["methods"][method]
        line_with_band(
            ax,
            budgets,
            values["deletions_mean"],
            values["deletions_std"],
            method,
        )
    ax.set_title("Exact deletions completed")
    ax.set_ylabel("Deletion count")

    # D: Scoring/sorting cost in exact-deletion equivalents.
    ax = axes[1, 0]
    costs = [
        result["methods"][m].get("score_cost_deletion_equivalents_mean", 0.0)
        for m in methods
    ]
    ax.bar([METHOD_LABEL[m] for m in methods], costs, color=[COLORS[m] for m in methods])
    ax.set_title("Priority cost")
    ax.set_ylabel("Exact-deletion equivalents")
    ax.tick_params(axis="x", rotation=35)

    mechanism = result.get("mechanism_summary", {})

    # E: Alignment with actual fully-deleted direction (diagnostic only).
    ax = axes[1, 1]
    mech_methods = [m for m in methods if m in mechanism]
    alignment = [mechanism[m]["mean_target_cosine"]["mean"] for m in mech_methods]
    alignment_std = [mechanism[m]["mean_target_cosine"]["std"] for m in mech_methods]
    ax.bar(
        [METHOD_LABEL[m] for m in mech_methods],
        alignment,
        yerr=alignment_std,
        capsize=3,
        color=[COLORS[m] for m in mech_methods],
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Top-25% update alignment")
    ax.set_ylabel("Mean cosine with target direction")
    ax.tick_params(axis="x", rotation=35)

    # F: Cancellation ratio; lower values mean stronger cancellation.
    ax = axes[1, 2]
    cancellation = [mechanism[m]["cancellation_ratio"]["mean"] for m in mech_methods]
    cancellation_std = [mechanism[m]["cancellation_ratio"]["std"] for m in mech_methods]
    ax.bar(
        [METHOD_LABEL[m] for m in mech_methods],
        cancellation,
        yerr=cancellation_std,
        capsize=3,
        color=[COLORS[m] for m in mech_methods],
    )
    ax.set_title("Top-25% cancellation ratio")
    ax.set_ylabel(r"$\|\sum_i \Delta\theta_i\| / \sum_i\|\Delta\theta_i\|$")
    ax.tick_params(axis="x", rotation=35)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        if ax not in axes[1, :]:
            ax.set_xlabel("Equal total compute budget (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_both(fig, output_dir / f"{path.stem}_overview")


def checkpoint_percent(key: str) -> float:
    match = re.search(r"rho_after_([0-9.]+)", key)
    return 100.0 * float(match.group(1)) if match else math.nan


def plot_staleness(path: Path, result: dict, output_dir: Path) -> None:
    staleness = result.get("staleness_summary", {})
    if not staleness:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHOD_ORDER:
        if method not in staleness:
            continue
        items = sorted(staleness[method].items(), key=lambda item: checkpoint_percent(item[0]))
        x = np.asarray([checkpoint_percent(k) for k, _ in items])
        mean = np.asarray([v["mean"] for _, v in items])
        std = np.asarray([v["std"] for _, v in items])
        ax.plot(x, mean, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABEL[method])
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.12)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Queue deleted using initial static ranking (%)")
    ax.set_ylabel("Spearman correlation: initial vs current scores")
    ax.set_title(f"Ranking staleness\n{display_title(path, result)}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    save_both(fig, output_dir / f"{path.stem}_staleness")


def plot_statistics(path: Path, result: dict, output_dir: Path) -> None:
    records = result.get("paired_progress_vs_random", [])
    if not records:
        return
    budgets = 100.0 * np.asarray(result["budget_fractions"], dtype=float)
    methods = [m for m in METHOD_ORDER if m != "random" and any(r["method"] == m for r in records)]
    matrix = np.full((len(methods), len(budgets)), np.nan)
    significant = np.zeros_like(matrix, dtype=bool)
    for r in records:
        if r["method"] not in methods:
            continue
        i = methods.index(r["method"])
        j = int(r["budget_index"])
        matrix[i, j] = 100.0 * float(r["mean_difference"])
        significant[i, j] = float(r.get("wilcoxon_p_holm", 1.0)) < 0.05

    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(budgets)):
            value = matrix[i, j]
            if np.isfinite(value):
                star = "*" if significant[i, j] else ""
                ax.text(j, i, f"{value:.1f}{star}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(budgets)), [f"{x:g}%" for x in budgets])
    ax.set_yticks(range(len(methods)), [METHOD_LABEL[m] for m in methods])
    ax.set_xlabel("Equal total compute budget")
    ax.set_title(f"Paired progress difference versus random\n{display_title(path, result)}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean progress difference (percentage points)")
    fig.text(0.5, 0.01, "* Holm-adjusted paired Wilcoxon p < 0.05", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_both(fig, output_dir / f"{path.stem}_statistics")


def is_criteo(result: dict, path: Path) -> bool:
    return result.get("dataset") == "rls-state" or "criteo" in path.stem.lower()


def queue_size(result: dict) -> int:
    if "queue_size" in result:
        return int(result["queue_size"])
    return int(result["repetitions"][0]["queue_size"])


def plot_criteo_scaling(items: list[tuple[Path, dict]], output_dir: Path) -> None:
    items = [(p, r) for p, r in items if is_criteo(r, p)]
    distinct = sorted({queue_size(r) for _, r in items})
    if len(distinct) < 2:
        return
    # If the same queue size occurs more than once, keep the file with more repeats.
    by_size = {}
    for path, result in items:
        q = queue_size(result)
        if q not in by_size or result.get("repeats", 0) > by_size[q][1].get("repeats", 0):
            by_size[q] = (path, result)
    queue_sizes = sorted(by_size)
    budgets = np.asarray(next(iter(by_size.values()))[1]["budget_fractions"], dtype=float)
    ncols = 3
    nrows = int(math.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.3 * nrows), squeeze=False)
    for budget_idx, budget in enumerate(budgets):
        ax = axes.flat[budget_idx]
        for method in METHOD_ORDER:
            y = []
            valid_x = []
            for q in queue_sizes:
                result = by_size[q][1]
                if method in result["methods"]:
                    valid_x.append(q)
                    y.append(100.0 * result["methods"][method]["progress_mean"][budget_idx])
            ax.plot(valid_x, y, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABEL[method])
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=100.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Budget {100 * budget:g}%")
        ax.set_xlabel("Queue size (log scale)")
        ax.set_ylabel("Progress (%) — symmetric-log")
        ax.grid(alpha=0.25)
    for index in range(len(budgets), nrows * ncols):
        axes.flat[index].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("Criteo queue-size scaling under equal compute budgets", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_both(fig, output_dir / "criteo_queue_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("queue_research_figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = [(path, load_result(path)) for path in args.results]
    for path, result in items:
        plot_overview(path, result, args.output_dir)
        plot_staleness(path, result, args.output_dir)
        plot_statistics(path, result, args.output_dir)
    plot_criteo_scaling(items, args.output_dir)
    print(f"Saved figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
