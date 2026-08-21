"""
Experiment: Prioritized Deletion Under a Compute Budget
---------------------------------------------------------
Motivating scenario: a stream of Q unlearning / deletion requests arrives
for an RLS model, but only a limited compute budget k < Q requests can be
exactly processed right now (each exact removal costs O(d^2), so k is
limited by latency/SLA, not by algorithmic difficulty). Which k requests
should be processed first?

We compare four priority orders for the SAME fixed queue of Q requests:
  - naive     : the naive residual score, e_i^2
  - leverage  : leverage-only score, h_i
  - cooks     : Cook's-distance-style score
  - sis       : the proposed Sample Influence Score (exact ||Delta w||^2)
  - random    : arrival order (no prioritization) -- the baseline you get
                without any importance score at all

For each ordering we process the queue sequentially using the EXACT online
downdate (RLS.unlearn), which is mathematically identical to refitting from
scratch without the removed samples, and after every processed request we
record:
  (a) held-out test MSE of the resulting model, and
  (b) "progress toward the fully-processed target": how much of the total
      required change ||w_0 - w_Q||^2 (from the model with all Q requests
      still present, to the model with all Q requests removed) has already
      been achieved after k steps. This directly measures whether a
      compute-limited system that stops after k of Q requests ends up
      close to -- or far from -- full compliance.

The queue itself mixes ordinary (benign) deletion requests with a subset of
deliberately harmful/poisoned points (on the synthetic dataset only, where
we know which points are which), simulating a realistic mixed workload:
some requests are routine privacy-driven erasure, others happen to be
consequential data-quality problems.
"""
import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rls_influence import RLS

rng = np.random.default_rng(11)


def make_synthetic(n=400, d=10, n_outliers=20, noise=0.3, seed=1):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    X = np.hstack([np.ones((n, 1)), X])
    w_true = r.normal(size=d + 1) * 1.5
    y = X @ w_true + r.normal(scale=noise, size=n)
    outlier_idx = r.choice(n, size=n_outliers, replace=False)
    X[outlier_idx, 1:] *= r.uniform(4, 7, size=(n_outliers, d))
    y[outlier_idx] += r.normal(scale=8.0, size=n_outliers)
    return X, y, outlier_idx


def priority_order(scores, queue, method):
    if method == "random":
        order = queue.copy()
        rng.shuffle(order)
        return order
    s = scores[method][queue]
    return queue[np.argsort(-s)]


def run_queue_experiment(name, model, X_test, y_test, queue, is_poison=None):
    scores = model.sample_influence_scores()
    w0, P0 = model.w.copy(), model.P.copy()

    # fully-processed target model: remove ALL queued samples
    target = model.copy()
    for i in queue:
        target.unlearn(i)
    w_target = target.w.copy()
    total_change = np.sum((w0 - w_target) ** 2)

    methods = ["sis", "cooks", "leverage", "naive", "random"]
    out = {"n_queue": int(len(queue)), "total_weight_change": float(total_change)}

    for m in methods:
        order = priority_order(scores, queue, m)
        working = model.copy()
        test_mse_traj = []
        progress_traj = []
        poison_removed_traj = []
        n_poison_removed = 0
        for step, i in enumerate(order, start=1):
            working.unlearn(i)
            pred = X_test @ working.w
            test_mse_traj.append(float(np.mean((y_test - pred) ** 2)))
            remaining_gap = np.sum((working.w - w_target) ** 2)
            progress = 1.0 - (remaining_gap / total_change if total_change > 1e-12 else 0.0)
            progress_traj.append(float(progress))
            if is_poison is not None:
                n_poison_removed += int(is_poison.get(i, False))
                poison_removed_traj.append(int(n_poison_removed))
        out[m] = {
            "test_mse_traj": test_mse_traj,
            "progress_traj": progress_traj,
        }
        if is_poison is not None:
            out[m]["poison_removed_traj"] = poison_removed_traj

    return out


GRID = np.linspace(0.05, 1.0, 20)  # common budget-fraction grid for averaging across repeats


def interp_to_grid(traj):
    x = np.linspace(1, len(traj), len(traj)) / len(traj)
    return np.interp(GRID, x, traj)


def run_queue_experiment_repeated(model, X_test, y_test, queue_builder, is_poison_builder=None, n_repeats=10):
    """Repeats the queue experiment n_repeats times (each with a freshly
    sampled queue / random order) and returns, for every method, the mean
    and std of the progress and test-MSE trajectories interpolated onto a
    common percent-of-budget grid -- this averages out the path-dependent
    noise of any single greedy removal order."""
    methods = ["sis", "cooks", "leverage", "naive", "random"]
    all_progress = {m: [] for m in methods}
    all_mse = {m: [] for m in methods}
    all_poison = {m: [] for m in methods}
    n_queue_vals, total_change_vals = [], []
    n_poison_in_queue = None

    for rep in range(n_repeats):
        queue, is_poison = queue_builder(rep)
        res = run_queue_experiment("rep", model, X_test, y_test, queue, is_poison=is_poison)
        n_queue_vals.append(res["n_queue"])
        total_change_vals.append(res["total_weight_change"])
        for m in methods:
            all_progress[m].append(interp_to_grid(res[m]["progress_traj"]))
            all_mse[m].append(interp_to_grid(res[m]["test_mse_traj"]))
            if is_poison is not None:
                all_poison[m].append(interp_to_grid(res[m]["poison_removed_traj"]))
        if is_poison is not None:
            n_poison_in_queue = sum(is_poison.values())

    out = {"n_queue_mean": float(np.mean(n_queue_vals)), "n_repeats": n_repeats,
           "budget_grid_pct": (GRID * 100).tolist()}
    for m in methods:
        P = np.array(all_progress[m])
        M = np.array(all_mse[m])
        out[m] = {
            "progress_mean": P.mean(axis=0).tolist(),
            "progress_std": P.std(axis=0).tolist(),
            "test_mse_mean": M.mean(axis=0).tolist(),
            "test_mse_std": M.std(axis=0).tolist(),
        }
        if all_poison[m]:
            Pn = np.array(all_poison[m])
            out[m]["poison_removed_mean"] = Pn.mean(axis=0).tolist()
            out[m]["poison_removed_std"] = Pn.std(axis=0).tolist()
    if n_poison_in_queue is not None:
        out["n_poison_in_queue_mean"] = float(n_poison_in_queue)
    return out


RESULTS = {}

# =====================================================================
# Synthetic: mixed queue of poisoned + benign deletion requests
# (repeated over multiple random benign-request samples + random orders)
# =====================================================================
X, y, outlier_idx_full = make_synthetic(n=400, d=10, n_outliers=20, noise=0.3, seed=1)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.3, random_state=0
)
is_outlier_train = np.isin(idx_train, outlier_idx_full)
n_train = X_train.shape[0]

model_syn = RLS(dim=X_train.shape[1], lam=1.0)
model_syn.fit_stream(X_train, y_train)

poison_positions = np.where(is_outlier_train)[0]          # all poisoned points in train
benign_positions = np.where(~is_outlier_train)[0]


def syn_queue_builder(rep):
    r = np.random.default_rng(1000 + rep)
    benign_sample = r.choice(benign_positions, size=40, replace=False)
    queue = np.concatenate([poison_positions, benign_sample])
    r.shuffle(queue)
    is_poison = {int(i): bool(is_outlier_train[i]) for i in queue}
    return queue, is_poison


res_syn = run_queue_experiment_repeated(model_syn, X_test, y_test, syn_queue_builder, n_repeats=15)
res_syn["n_poison_in_queue_mean"] = float(len(poison_positions))
res_syn["n_benign_in_queue_mean"] = 40.0
RESULTS["synthetic"] = res_syn

# =====================================================================
# Diabetes: no known poisoned points -- pure "progress toward compliance"
# and test-MSE-under-budget comparison on a realistic, unlabeled queue
# =====================================================================
data = load_diabetes()
Xd, yd = data.data, data.target
Xd = StandardScaler().fit_transform(Xd)
Xd = np.hstack([np.ones((Xd.shape[0], 1)), Xd])
yd = (yd - yd.mean()) / yd.std()
Xd_train, Xd_test, yd_train, yd_test, idxd_train, idxd_test = train_test_split(
    Xd, yd, np.arange(len(yd)), test_size=0.3, random_state=0
)
model_d = RLS(dim=Xd_train.shape[1], lam=1.0)
model_d.fit_stream(Xd_train, yd_train)
n_train_d = Xd_train.shape[0]


def diabetes_queue_builder(rep):
    r = np.random.default_rng(2000 + rep)
    queue = r.choice(n_train_d, size=60, replace=False)
    return queue, None


res_d = run_queue_experiment_repeated(model_d, Xd_test, yd_test, diabetes_queue_builder, n_repeats=15)
RESULTS["diabetes"] = res_d

with open("queue_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

# quick console summary
for ds in RESULTS:
    print(f"\n=== {ds} ===  n_queue~{RESULTS[ds]['n_queue_mean']:.0f}  (avg over {RESULTS[ds]['n_repeats']} repeats)")
    grid = np.array(RESULTS[ds]["budget_grid_pct"])
    k25 = int(np.argmin(np.abs(grid - 25)))
    for m in ["sis", "cooks", "leverage", "naive", "random"]:
        prog_mean = RESULTS[ds][m]["progress_mean"]
        mse_mean = RESULTS[ds][m]["test_mse_mean"]
        print(f"  {m:9s}  progress@25%budget={prog_mean[k25]*100:5.1f}%   test_mse@25%budget={mse_mean[k25]:.4f}   test_mse@100%={mse_mean[-1]:.4f}")

