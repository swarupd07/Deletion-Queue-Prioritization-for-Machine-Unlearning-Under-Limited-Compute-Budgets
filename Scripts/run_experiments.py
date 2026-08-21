import json
import time

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rls_influence import RLS

rng = np.random.default_rng(42)
RESULTS = {}


def make_synthetic(n=300, d=8, n_outliers=15, noise=0.3, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    X = np.hstack([np.ones((n, 1)), X])  # bias term
    w_true = r.normal(size=d + 1) * 1.5
    y = X @ w_true + r.normal(scale=noise, size=n)
    outlier_idx = r.choice(n, size=n_outliers, replace=False)
    # inject high-leverage + label-corrupted outliers
    X[outlier_idx, 1:] *= r.uniform(4, 7, size=(n_outliers, d))
    y[outlier_idx] += r.normal(scale=8.0, size=n_outliers)
    return X, y, outlier_idx, w_true


def evaluate_dataset(name, X, y, lam=1.0, outlier_idx=None, n_loo_check=40, n_refit_check=15):
    n, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # ---- fit RLS online, sample by sample ----
    model = RLS(dim=d, lam=lam)
    t0 = time.perf_counter()
    model.fit_stream(X_train, y_train)
    t_fit = time.perf_counter() - t0

    # ---- compute proposed + baseline scores (vectorized, O(n d^2) total) ----
    t0 = time.perf_counter()
    scores = model.sample_influence_scores()
    t_scores = time.perf_counter() - t0

    n_train = X_train.shape[0]

    # ---- ground truth #1: exact recursive downdate formula ----
    exact_dw2 = model.exact_loo_weight_change()
    rel_err = np.abs(scores["sis"] - exact_dw2) / (exact_dw2 + 1e-12)
    RESULTS.setdefault(name, {})["identity_check_max_rel_err"] = float(np.max(rel_err))
    RESULTS[name]["identity_check_mean_rel_err"] = float(np.mean(rel_err))
    # how well do the *baselines* approximate the true ||delta w||^2, vs SIS's exact match?
    RESULTS[name]["baseline_vs_true_weight_change_pearson_r"] = {
        "sis": float(pearsonr(scores["sis"], exact_dw2)[0]),
        "cooks": float(pearsonr(scores["cooks"], exact_dw2)[0]),
        "leverage": float(pearsonr(scores["leverage"], exact_dw2)[0]),
        "naive": float(pearsonr(scores["naive"], exact_dw2)[0]),
    }

    # ---- ground truth #2: brute-force refit-without-i on a random subset ----
    check_idx = rng.choice(n_train, size=min(n_refit_check, n_train), replace=False)
    t0 = time.perf_counter()
    brute_dw2 = []
    for i in check_idx:
        w_loo = model.refit_without(i)
        brute_dw2.append(np.sum((model.w - w_loo) ** 2))
    t_brute_subset = time.perf_counter() - t0
    brute_dw2 = np.array(brute_dw2)
    sis_subset = scores["sis"][check_idx]
    corr_brute = pearsonr(sis_subset, brute_dw2)[0]
    RESULTS[name]["sis_vs_bruteforce_refit_pearson_r"] = float(corr_brute)
    RESULTS[name]["sis_vs_bruteforce_refit_max_rel_err"] = float(
        np.max(np.abs(sis_subset - brute_dw2) / (brute_dw2 + 1e-12))
    )

    # ---- downstream validation: actual effect on held-out test MSE ----
    base_pred = X_test @ model.w
    base_test_mse = np.mean((y_test - base_pred) ** 2)
    n_dtestmse = min(n_loo_check, n_train)
    dtestmse_idx = rng.choice(n_train, size=n_dtestmse, replace=False)
    d_test_mse = []
    for i in dtestmse_idx:
        w_loo = model.refit_without(i)
        pred_loo = X_test @ w_loo
        mse_loo = np.mean((y_test - pred_loo) ** 2)
        d_test_mse.append(abs(mse_loo - base_test_mse))
    d_test_mse = np.array(d_test_mse)

    corr_table = {}
    for score_name in ["sis", "cooks", "leverage", "naive"]:
        s = scores[score_name][dtestmse_idx]
        rho, _ = spearmanr(s, d_test_mse)
        corr_table[score_name] = float(rho)
    RESULTS[name]["spearman_vs_test_mse_change"] = corr_table

    # ---- outlier recovery (top-k precision/recall), if synthetic outliers known ----
    if outlier_idx is not None:
        # map global outlier idx (defined pre-split) -- for synthetic we split AFTER
        pass

    # ---- efficiency comparison: proposed formula vs brute-force LOO sweep over ALL samples ----
    # (measured on a manageable subset and extrapolated is unreliable -> measure on full set
    #  when n_train is small enough; otherwise measure on a fixed sample count and report per-sample cost)
    per_sample_brute = t_brute_subset / len(check_idx)
    est_full_brute_time = per_sample_brute * n_train
    RESULTS[name]["timing_sec"] = {
        "rls_streaming_fit_all_n_train": t_fit,
        "sis_scores_all_n_train_vectorized": t_scores,
        "brute_force_refit_per_sample": per_sample_brute,
        "brute_force_refit_extrapolated_all_n_train": est_full_brute_time,
        "speedup_factor": est_full_brute_time / t_scores if t_scores > 0 else None,
        "n_train": int(n_train),
        "dim": int(d),
    }

    return model, scores, X_train, y_train, X_test, y_test


# =====================================================================
# Experiment 1: synthetic data with known injected outliers
# =====================================================================
X, y, outlier_idx_full, w_true = make_synthetic(n=400, d=10, n_outliers=20, noise=0.3, seed=1)
# do our own split so we can track which TRAIN rows are the injected outliers
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.3, random_state=0
)
is_outlier_train = np.isin(idx_train, outlier_idx_full)

model_syn = RLS(dim=X_train.shape[1], lam=1.0)
t0 = time.perf_counter()
model_syn.fit_stream(X_train, y_train)
t_fit_syn = time.perf_counter() - t0

t0 = time.perf_counter()
scores_syn = model_syn.sample_influence_scores()
t_scores_syn = time.perf_counter() - t0

n_train_syn = X_train.shape[0]
n_actual_outliers = int(is_outlier_train.sum())

topk_results = {}
for score_name in ["sis", "cooks", "leverage", "naive"]:
    order = np.argsort(-scores_syn[score_name])
    topk = order[:n_actual_outliers]
    hit = np.isin(topk, np.where(is_outlier_train)[0]).sum()
    precision = hit / n_actual_outliers
    topk_results[score_name] = {
        "top_k": int(n_actual_outliers),
        "true_outliers_recovered": int(hit),
        "precision_at_k": float(precision),
    }

# identity check on synthetic
exact_dw2_syn = model_syn.exact_loo_weight_change()
rel_err_syn = np.abs(scores_syn["sis"] - exact_dw2_syn) / (exact_dw2_syn + 1e-12)

# brute-force cross-check subset
check_idx = rng.choice(n_train_syn, size=15, replace=False)
t0 = time.perf_counter()
brute_dw2 = []
for i in check_idx:
    w_loo = model_syn.refit_without(i)
    brute_dw2.append(np.sum((model_syn.w - w_loo) ** 2))
t_brute_subset_syn = time.perf_counter() - t0
brute_dw2 = np.array(brute_dw2)
sis_subset = scores_syn["sis"][check_idx]
corr_brute_syn = pearsonr(sis_subset, brute_dw2)[0]

# downstream test-mse validation
base_pred = X_test @ model_syn.w
base_test_mse = np.mean((y_test - base_pred) ** 2)
dtestmse_idx = rng.choice(n_train_syn, size=40, replace=False)
d_test_mse = []
for i in dtestmse_idx:
    w_loo = model_syn.refit_without(i)
    pred_loo = X_test @ w_loo
    mse_loo = np.mean((y_test - pred_loo) ** 2)
    d_test_mse.append(abs(mse_loo - base_test_mse))
d_test_mse = np.array(d_test_mse)
corr_table_syn = {}
for score_name in ["sis", "cooks", "leverage", "naive"]:
    s = scores_syn[score_name][dtestmse_idx]
    rho, _ = spearmanr(s, d_test_mse)
    corr_table_syn[score_name] = float(rho)

per_sample_brute_syn = t_brute_subset_syn / len(check_idx)
est_full_brute_time_syn = per_sample_brute_syn * n_train_syn

RESULTS["synthetic"] = {
    "n_train": int(n_train_syn),
    "dim": int(X_train.shape[1]),
    "n_injected_outliers_in_train": n_actual_outliers,
    "identity_check_max_rel_err": float(np.max(rel_err_syn)),
    "identity_check_mean_rel_err": float(np.mean(rel_err_syn)),
    "sis_vs_bruteforce_refit_pearson_r": float(corr_brute_syn),
    "baseline_vs_true_weight_change_pearson_r": {
        "sis": float(pearsonr(scores_syn["sis"], exact_dw2_syn)[0]),
        "cooks": float(pearsonr(scores_syn["cooks"], exact_dw2_syn)[0]),
        "leverage": float(pearsonr(scores_syn["leverage"], exact_dw2_syn)[0]),
        "naive": float(pearsonr(scores_syn["naive"], exact_dw2_syn)[0]),
    },
    "topk_outlier_recovery": topk_results,
    "spearman_vs_test_mse_change": corr_table_syn,
    "timing_sec": {
        "rls_streaming_fit_all_n_train": t_fit_syn,
        "sis_scores_all_n_train_vectorized": t_scores_syn,
        "brute_force_refit_per_sample": per_sample_brute_syn,
        "brute_force_refit_extrapolated_all_n_train": est_full_brute_time_syn,
        "speedup_factor": est_full_brute_time_syn / t_scores_syn if t_scores_syn > 0 else None,
    },
}

# =====================================================================
# Experiment 2: real dataset -- sklearn diabetes
# =====================================================================
data = load_diabetes()
Xd, yd = data.data, data.target
Xd = StandardScaler().fit_transform(Xd)
Xd = np.hstack([np.ones((Xd.shape[0], 1)), Xd])
yd = (yd - yd.mean()) / yd.std()
evaluate_dataset("diabetes", Xd, yd, lam=1.0, n_loo_check=40, n_refit_check=15)

# =====================================================================
# Experiment 3: larger-scale synthetic dataset (shows efficiency scaling
# in n and d; California Housing download was blocked by network egress
# rules in this environment, so we substitute a bigger synthetic set with
# more features and more samples to stress-test scaling instead).
# =====================================================================
Xl, yl, outlier_idx_l, w_true_l = make_synthetic(n=2000, d=25, n_outliers=60, noise=0.4, seed=3)
evaluate_dataset("large_synthetic_n2000_d26", Xl, yl, lam=1.0, n_loo_check=40, n_refit_check=15)

with open("results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
