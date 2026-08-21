import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

with open("results.json") as f:
    R = json.load(f)

datasets = ["synthetic", "diabetes", "large_synthetic_n2000_d26"]
labels = ["Synthetic\n(n=280, d=11)", "Diabetes\n(n=309, d=11)", "Large Synthetic\n(n=1400, d=26)"]
methods = ["sis", "cooks", "leverage", "naive"]
method_labels = ["SIS (recursive DFBETA)", "Cook's-style", "Leverage-only", "Naive residual"]
colors = ["#1B4F8C", "#E8871E", "#8E44AD", "#C0392B"]

# ---------- Figure 1: exact-match Pearson r vs true weight change ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(len(datasets))
width = 0.19
for i, (m, ml, c) in enumerate(zip(methods, method_labels, colors)):
    vals = [R[d]["baseline_vs_true_weight_change_pearson_r"][m] for d in datasets]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=ml, color=c, edgecolor="white", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Pearson r vs. exact  $\\|w - w_{-i}\\|^2$")
ax.set_title("Fidelity to the true leave-one-out weight change", pad=12)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
ax.legend(fontsize=9, loc="lower left", frameon=False)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig1_exact_match.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Figure 2: Spearman correlation with downstream test-MSE change ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
for i, (m, ml, c) in enumerate(zip(methods, method_labels, colors)):
    vals = [R[d]["spearman_vs_test_mse_change"][m] for d in datasets]
    ax.bar(x + (i - 1.5) * width, vals, width, label=ml, color=c, edgecolor="white", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Spearman \u03c1 vs. |\u0394 test MSE| on removal")
ax.set_title("Predicting downstream (unlearning-relevant) impact", pad=12)
ax.axhline(0, color="black", linewidth=0.8, zorder=0)
ax.legend(fontsize=9, loc="upper right", frameon=False)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig2_downstream_spearman.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Figure 3: efficiency / speedup ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
speedups = [R[d]["timing_sec"]["speedup_factor"] for d in datasets]
bars = ax.bar(labels, speedups, color="#1B4F8C", edgecolor="white", linewidth=0.6, width=0.55)
ax.set_ylabel("Speed-up factor  (brute-force LOO refit \u00f7 SIS)")
ax.set_title("Efficiency of the recursive SIS computation vs. brute-force LOO", pad=12, fontsize=12.5)
for b, v in zip(bars, speedups):
    ax.text(b.get_x() + b.get_width() / 2, v + max(speedups) * 0.02, f"{v:.0f}\u00d7",
            ha="center", fontsize=11, fontweight="bold", color="#1B4F8C")
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(speedups) * 1.18)
plt.tight_layout()
plt.savefig("fig3_speedup.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Figure 4: top-k outlier recovery on synthetic data ----------
fig, ax = plt.subplots(figsize=(7, 4.8))
topk = R["synthetic"]["topk_outlier_recovery"]
vals = [topk[m]["precision_at_k"] for m in methods]
bars = ax.bar(method_labels, vals, color=colors, edgecolor="white", linewidth=0.6, width=0.55)
ax.set_ylabel(f"Precision@k  (k = {topk['sis']['top_k']} injected outliers)")
ax.set_title("Recovering injected high-leverage / mislabeled points", pad=12, fontsize=12.5)
ax.set_ylim(0, 1.15)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
plt.setp(ax.get_xticklabels(), fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig4_outlier_recovery.png", dpi=200, bbox_inches="tight")
plt.close()

print("figures written")
