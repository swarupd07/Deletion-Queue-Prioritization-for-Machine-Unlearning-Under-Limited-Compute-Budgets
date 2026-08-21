"""
Benchmark: Deletion-only vs. SIS-scoring + Deletion cost
-----------------------------------------------------------
Section 6.5 already shows that computing SIS for every sample is orders of
magnitude cheaper than a brute-force leave-one-out sweep. This benchmark
asks a complementary, more operationally direct question: once you're
already committed to performing Q exact deletions (via unlearn()), how much
EXTRA wall-clock time does also computing the SIS prioritization score cost,
on top of the deletions themselves?

We measure, per dataset:
  t_delete       : time to sequentially unlearn() a fixed queue of Q samples,
                    with NO scoring at all (the "deletion-only" baseline --
                    e.g. FIFO / arrival-order processing with no priority
                    signal).
  t_sis_score     : time to compute SIS for every sample in the training set
                    once (a single vectorized O(n d^2) pass), which is what
                    a system would do once, live, to obtain a priority order
                    before processing the queue.
  t_cooks_score   : time to compute the classical Cook's-distance-style
                    score on its own (same dominant O(n d^2) cost as SIS,
                    since both need X @ P; timed as its own standalone
                    computation rather than reusing SIS's timing).
  t_bruteforce_score : the only alternative way to obtain an equally exact
                    priority signal without SIS -- refit the model once per
                    training sample (O(n) refits of O(n d^2) each) and rank
                    by the resulting weight change. This is the same
                    quantity used as the "brute-force LOO" baseline in
                    Section 6.5, recomputed here for internal consistency
                    with this benchmark's own timing run.

From these we report the ADDED OVERHEAD of scoring, as a percentage of the
deletion-only cost, for SIS, Cook's-style, and the brute-force alternative,
and the resulting total pipeline time (score once + delete the queue) for
each.
"""
import json
import time

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from rls_influence import RLS

rng = np.random.default_rng(99)


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


def benchmark_dataset(name, X_train, y_train, lam=1.0, queue_size=60, n_reps=7):
    n_train, d = X_train.shape
    Q = min(queue_size, n_train)

    t_delete_reps, t_sis_reps, t_cooks_reps, t_brute_reps = [], [], [], []

    for rep in range(n_reps):
        # fresh model each rep so repeated unlearn() calls don't compound
        model = RLS(dim=d, lam=lam)
        model.fit_stream(X_train, y_train)

        queue = rng.choice(n_train, size=Q, replace=False)

        # ---- deletion-only baseline: unlearn() the queue, no scoring ----
        work = model.copy()
        t0 = time.perf_counter()
        for i in queue:
            work.unlearn(i)
        t_delete = time.perf_counter() - t0
        t_delete_reps.append(t_delete)

        # ---- SIS scoring cost: one vectorized pass over all n_train ----
        t0 = time.perf_counter()
        _ = model.sample_influence_scores()
        t_sis = time.perf_counter() - t0
        t_sis_reps.append(t_sis)

        # ---- Cook's-style scoring cost: same dominant X@P cost as SIS, but
        #      timed as its own standalone computation (see cooks_only_scores
        #      docstring) rather than reusing sample_influence_scores' timing.
        t0 = time.perf_counter()
        _ = model.cooks_only_scores()
        t_cooks = time.perf_counter() - t0
        t_cooks_reps.append(t_cooks)

        # ---- brute-force scoring cost: refit once per sample, subset-timed
        #      and extrapolated to all n_train (n_train full refits is too
        #      slow to run every rep at larger n; time a subset and scale) --
        check = rng.choice(n_train, size=min(15, n_train), replace=False)
        t0 = time.perf_counter()
        for i in check:
            model.refit_without(i)
        t_brute_subset = time.perf_counter() - t0
        t_brute_full_est = (t_brute_subset / len(check)) * n_train
        t_brute_reps.append(t_brute_full_est)

    t_delete_med = float(np.median(t_delete_reps))
    t_sis_med = float(np.median(t_sis_reps))
    t_cooks_med = float(np.median(t_cooks_reps))
    t_brute_med = float(np.median(t_brute_reps))

    return {
        "n_train": int(n_train),
        "queue_size": int(Q),
        "n_reps": n_reps,
        "t_delete_only_sec": t_delete_med,
        "t_sis_score_sec": t_sis_med,
        "t_cooks_score_sec": t_cooks_med,
        "t_bruteforce_score_sec": t_brute_med,
        "sis_overhead_pct_of_deletion": 100.0 * t_sis_med / t_delete_med,
        "cooks_overhead_pct_of_deletion": 100.0 * t_cooks_med / t_delete_med,
        "bruteforce_overhead_pct_of_deletion": 100.0 * t_brute_med / t_delete_med,
        "pipeline_deletion_only_sec": t_delete_med,
        "pipeline_sis_plus_deletion_sec": t_sis_med + t_delete_med,
        "pipeline_cooks_plus_deletion_sec": t_cooks_med + t_delete_med,
        "pipeline_bruteforce_plus_deletion_sec": t_brute_med + t_delete_med,
        "sis_vs_bruteforce_scoring_speedup": t_brute_med / t_sis_med,
        "sis_vs_cooks_scoring_ratio": t_sis_med / t_cooks_med,
    }


RESULTS = {}

# Synthetic (matches Section 5.1 / 6.6 sizing)
X, y, outlier_idx = make_synthetic(n=400, d=10, n_outliers=20, noise=0.3, seed=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
RESULTS["synthetic"] = benchmark_dataset("synthetic", X_train, y_train, lam=1.0, queue_size=57)

# Diabetes
data = load_diabetes()
Xd, yd = data.data, data.target
Xd = StandardScaler().fit_transform(Xd)
Xd = np.hstack([np.ones((Xd.shape[0], 1)), Xd])
yd = (yd - yd.mean()) / yd.std()
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.3, random_state=0)
RESULTS["diabetes"] = benchmark_dataset("diabetes", Xd_train, yd_train, lam=1.0, queue_size=60)

# Large synthetic (matches Section 5.1 sizing)
Xl, yl, outlier_idx_l = make_synthetic(n=2000, d=25, n_outliers=60, noise=0.4, seed=3)
Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.3, random_state=0)
RESULTS["large_synthetic_n2000_d26"] = benchmark_dataset(
    "large_synthetic_n2000_d26", Xl_train, yl_train, lam=1.0, queue_size=60
)

with open("deletion_benchmark.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
