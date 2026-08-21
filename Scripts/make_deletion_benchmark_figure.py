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

with open("deletion_benchmark.json") as f:
    B = json.load(f)

datasets = ["synthetic", "diabetes", "large_synthetic_n2000_d26"]
labels = [f"Synthetic\n(n={B[d]['n_train']}, Q={B[d]['queue_size']})" for d in ["synthetic"]] + \
         [f"Diabetes\n(n={B[d]['n_train']}, Q={B[d]['queue_size']})" for d in ["diabetes"]] + \
         [f"Large Synthetic\n(n={B[d]['n_train']}, Q={B[d]['queue_size']})" for d in ["large_synthetic_n2000_d26"]]

deletion_only = [B[d]["pipeline_deletion_only_sec"] * 1000 for d in datasets]   # ms
sis_pipeline = [B[d]["pipeline_sis_plus_deletion_sec"] * 1000 for d in datasets]
brute_pipeline = [B[d]["pipeline_bruteforce_plus_deletion_sec"] * 1000 for d in datasets]

x = np.arange(len(datasets))
width = 0.26

fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - width, deletion_only, width, label="Deletion only (no scoring)", color="#7F8C8D", edgecolor="white", linewidth=0.6)
b2 = ax.bar(x, sis_pipeline, width, label="Deletion + SIS scoring", color="#1B4F8C", edgecolor="white", linewidth=0.6)
b3 = ax.bar(x + width, brute_pipeline, width, label="Deletion + brute-force scoring", color="#C0392B", edgecolor="white", linewidth=0.6)

ax.set_yscale("log")
ax.set_ylabel("Total pipeline wall-clock time, ms (log scale)")
ax.set_title("Overhead of scoring on top of the deletions themselves", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize=9, loc="upper left", frameon=False)
ax.grid(axis="y", which="both", alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)

for bars, vals in [(b1, deletion_only), (b2, sis_pipeline), (b3, brute_pipeline)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}", ha="center", fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig("fig8_deletion_overhead.png", dpi=200, bbox_inches="tight")
plt.close()

print("figure 8 written")

# ---- also print a clean summary table for pasting into the paper ----
print("\nDataset            | Delete-only (ms) | +SIS (ms) | SIS overhead | +Brute-force (ms) | BF overhead | SIS speedup on scoring")
for d in datasets:
    r = B[d]
    print(f"{d:19s} | {r['pipeline_deletion_only_sec']*1000:16.3f} | "
          f"{r['pipeline_sis_plus_deletion_sec']*1000:9.3f} | {r['sis_overhead_pct_of_deletion']:11.1f}% | "
          f"{r['pipeline_bruteforce_plus_deletion_sec']*1000:18.2f} | {r['bruteforce_overhead_pct_of_deletion']:10.0f}% | "
          f"{r['sis_vs_bruteforce_scoring_speedup']:8.0f}x")
