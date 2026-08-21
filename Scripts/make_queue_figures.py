import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

with open("queue_results.json") as f:
    Q = json.load(f)

methods = ["sis", "cooks", "leverage", "naive", "random"]
method_labels = {
    "sis": "SIS (recursive DFBETA)",
    "cooks": "Cook's-style",
    "leverage": "Leverage-only",
    "naive": "Naive residual",
    "random": "Random (no score)",
}
colors = {
    "sis": "#1B4F8C",
    "cooks": "#E8871E",
    "leverage": "#8E44AD",
    "naive": "#C0392B",
    "random": "#7F8C8D",
}
widths = {"sis": 3.2, "cooks": 1.9, "leverage": 1.9, "naive": 1.9, "random": 1.9}
styles = {"sis": "-", "cooks": "--", "leverage": "--", "naive": ":", "random": (0, (1, 1.4))}
zorders = {"sis": 5, "cooks": 4, "leverage": 3, "naive": 2, "random": 1}

# ---------- Figure 5: progress toward the fully-processed target model ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
for ax, ds, title in zip(axes, ["synthetic", "diabetes"],
                          [f"Synthetic  (~{Q['synthetic']['n_queue_mean']:.0f} requests incl. {Q['synthetic']['n_poison_in_queue_mean']:.0f} poisoned, avg of {Q['synthetic']['n_repeats']} runs)",
                           f"Diabetes  (~{Q['diabetes']['n_queue_mean']:.0f} requests, avg of {Q['diabetes']['n_repeats']} runs)"]):
    grid = np.array(Q[ds]["budget_grid_pct"])
    for m in methods:
        mean = np.array(Q[ds][m]["progress_mean"]) * 100
        std = np.array(Q[ds][m]["progress_std"]) * 100
        ax.plot(grid, mean, label=method_labels[m], color=colors[m],
                 linewidth=widths[m], linestyle=styles[m], zorder=zorders[m])
        ax.fill_between(grid, mean - std, mean + std, color=colors[m], alpha=0.10, zorder=zorders[m] - 0.5)
    ax.set_xlabel("% of deletion queue processed (compute budget spent)")
    ax.set_ylabel("% progress toward fully-compliant model")
    ax.set_title(title, pad=10, fontsize=11.5)
    ax.set_ylim(-3, 103)
    ax.set_xlim(5, 100)
    ax.grid(alpha=0.25, linewidth=0.6)
axes[0].legend(loc="lower right", frameon=False)
fig.suptitle("Prioritized unlearning: how much of the required change is achieved under a limited budget?",
             fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig5_priority_progress.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Figure 6: held-out test MSE trajectory vs. budget spent ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
for ax, ds, title in zip(axes, ["synthetic", "diabetes"],
                          [f"Synthetic  (avg of {Q['synthetic']['n_repeats']} runs)",
                           f"Diabetes  (avg of {Q['diabetes']['n_repeats']} runs)"]):
    grid = np.array(Q[ds]["budget_grid_pct"])
    for m in methods:
        mean = np.array(Q[ds][m]["test_mse_mean"])
        std = np.array(Q[ds][m]["test_mse_std"])
        ax.plot(grid, mean, label=method_labels[m], color=colors[m],
                 linewidth=widths[m], linestyle=styles[m], zorder=zorders[m])
        ax.fill_between(grid, mean - std, mean + std, color=colors[m], alpha=0.10, zorder=zorders[m] - 0.5)
    ax.set_xlabel("% of deletion queue processed (compute budget spent)")
    ax.set_ylabel("Held-out test MSE (mean \u00b1 1 std over repeats)")
    ax.set_title(title, pad=10, fontsize=11.5)
    ax.set_xlim(5, 100)
    ax.grid(alpha=0.25, linewidth=0.6)
axes[0].legend(loc="upper right", frameon=False)
fig.suptitle("Held-out test error as the deletion queue is processed, by priority order",
             fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig6_priority_test_mse.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Figure 7: cumulative poisoned points removed (synthetic only) ----------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
grid = np.array(Q["synthetic"]["budget_grid_pct"])
for m in methods:
    mean = np.array(Q["synthetic"][m]["poison_removed_mean"])
    std = np.array(Q["synthetic"][m]["poison_removed_std"])
    ax.plot(grid, mean, label=method_labels[m], color=colors[m],
             linewidth=widths[m], linestyle=styles[m], zorder=zorders[m])
    ax.fill_between(grid, mean - std, mean + std, color=colors[m], alpha=0.10, zorder=zorders[m] - 0.5)
n_poison = Q["synthetic"]["n_poison_in_queue_mean"]
ax.axhline(n_poison, color="black", linewidth=0.9, linestyle=":")
ax.text(6, n_poison + 0.5, f"all {n_poison:.0f} poisoned points in queue", fontsize=9, color="#333333")
ax.set_xlabel("% of deletion queue processed (compute budget spent)")
ax.set_ylabel("Cumulative poisoned points removed (mean \u00b1 1 std)")
ax.set_title(f"Which priority order clears poisoned data fastest?\n(avg of {Q['synthetic']['n_repeats']} runs)", pad=10, fontsize=12)
ax.set_xlim(5, 100)
ax.grid(alpha=0.25, linewidth=0.6)
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("fig7_poison_clearance.png", dpi=200, bbox_inches="tight")
plt.close()

print("figures 5-7 written")
