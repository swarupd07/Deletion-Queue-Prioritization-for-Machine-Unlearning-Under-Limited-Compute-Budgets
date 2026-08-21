"""
Sample Influence Metrics for Recursive Least Squares (RLS)
------------------------------------------------------------
Implements:
  1. Standard RLS recursion (Sherman-Morrison update) for ridge regression.
  2. The proposed Sample Influence Score (SIS): an EXACT, O(d^2)-per-sample
     estimate of how much removing a given training sample would change the
     learned weight vector, computed entirely from quantities RLS already
     maintains (the precision matrix P_t and the running weights w_t) --
     i.e. with NO refitting.
  3. Three baselines: naive squared residual, leverage-only, and a
     Cook's-distance-style score.
  4. Exact leave-one-out (LOO) downdate + full refit, used as ground truth.

Author: experiment code accompanying the paper draft.
"""

import numpy as np


class RLS:
    """Recursive Least Squares (ridge-regularized) estimator.

    Maintains:
        w_t : current weight vector estimate
        P_t : (X_t^T X_t + lambda*I)^{-1}, the running precision matrix
    via the Sherman-Morrison rank-1 update, processing one sample at a time.
    """

    def __init__(self, dim, lam=1.0):
        self.dim = dim
        self.lam = lam
        self.P = np.eye(dim) / lam
        self.w = np.zeros(dim)
        self.n_seen = 0
        # store history needed to compute per-sample influence after the fact
        self.X_hist = []
        self.y_hist = []

    def update(self, x, y):
        """One RLS step with a single sample (x in R^d, y scalar)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        Px = self.P @ x
        denom = 1.0 + x @ Px
        gain = Px / denom
        err = y - x @ self.w
        self.w = self.w + gain * err
        self.P = self.P - np.outer(gain, Px)
        self.n_seen += 1
        self.X_hist.append(x)
        self.y_hist.append(y)

    def fit_stream(self, X, y):
        for xi, yi in zip(X, y):
            self.update(xi, yi)
        return self

    # ------------------------------------------------------------------
    # Sample Influence Score (SIS) -- the paper's main contribution
    # ------------------------------------------------------------------
    def sample_influence_scores(self):
        """
        Compute, for every sample seen so far, an EXACT closed-form estimate
        of || w_full - w_{-i} ||^2  (squared change in the weight vector if
        sample i were removed / unlearned from the current model), using
        only the final P and w already maintained by RLS.

        Derivation (ridge LOO downdate):
            h_i   = x_i^T P x_i                      (leverage)
            e_i   = y_i - x_i^T w                     (residual, full fit)
            w_{-i} = w - P x_i e_i / (1 - h_i)
        =>  || w - w_{-i} ||^2 = e_i^2 * || P x_i ||^2 / (1 - h_i)^2

        This requires only a matrix-vector product per sample (already
        computed once during RLS updates), i.e. O(d^2) per sample --
        versus O(d^3) for a fresh matrix inversion or O(n d^2) for a
        brute-force refit-based LOO sweep.

        Returns a dict of arrays: sis, leverage, residual, cooks, naive
        """
        X = np.array(self.X_hist)
        y = np.array(self.y_hist)
        n = X.shape[0]

        Px_all = X @ self.P                      # (n, d), row i = P x_i  (P symmetric)
        h = np.einsum('ij,ij->i', X, Px_all)      # leverage h_i = x_i^T P x_i
        resid = y - X @ self.w                    # e_i
        Px_norm_sq = np.einsum('ij,ij->i', Px_all, Px_all)  # ||P x_i||^2

        denom = (1.0 - h)
        # guard against near-singular (h_i -> 1) samples
        denom_safe = np.where(np.abs(denom) < 1e-8, 1e-8, denom)

        sis = (resid ** 2) * Px_norm_sq / (denom_safe ** 2)          # proposed exact score
        cooks = (resid ** 2) * h / (denom_safe ** 2)                 # Cook's-distance-style baseline
        leverage_only = h.copy()                                     # leverage-only baseline
        naive = resid ** 2                                           # naive residual-only baseline

        return {
            "sis": sis,
            "leverage": leverage_only,
            "residual": resid,
            "naive": naive,
            "cooks": cooks,
            "h": h,
        }

    def cooks_only_scores(self):
        """
        Compute ONLY the Cook's-distance-style score, e_i^2 * h_i / (1-h_i)^2,
        without the ||P x_i||^2 term SIS additionally needs. Used to time
        Cook's-style scoring on its own (rather than reusing the timing of
        sample_influence_scores(), which computes SIS, Cook's, leverage, and
        naive all together in one pass and would understate the marginal
        cost of "just Cook's" as a standalone pipeline choice).

        Both scores still share the dominant cost, Px_all = X @ P (O(n d^2)),
        since leverage h_i = x_i^T P x_i requires it regardless; the only
        work this skips relative to sample_influence_scores() is the O(n d)
        ||P x_i||^2 reduction that SIS alone needs. Returns the cooks array.
        """
        X = np.array(self.X_hist)
        y = np.array(self.y_hist)
        Px_all = X @ self.P
        h = np.einsum('ij,ij->i', X, Px_all)
        resid = y - X @ self.w
        denom = (1.0 - h)
        denom_safe = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        cooks = (resid ** 2) * h / (denom_safe ** 2)
        return cooks

    def exact_loo_weight_change(self):
        """Ground truth: exact ||w - w_{-i}||^2 for every i via the closed-form
        downdate (mathematically exact for ridge regression). Used to verify
        the SIS formula is correct, independent of the derivation above."""
        X = np.array(self.X_hist)
        y = np.array(self.y_hist)
        n = X.shape[0]
        out = np.zeros(n)
        for i in range(n):
            xi, yi = X[i], y[i]
            Pxi = self.P @ xi
            hi = xi @ Pxi
            denom = 1.0 - hi
            if abs(denom) < 1e-8:
                denom = 1e-8
            ei = yi - xi @ self.w
            dw = Pxi * ei / denom
            out[i] = dw @ dw
        return out

    def refit_without(self, i):
        """Brute-force: refit ridge regression on all samples except i,
        from scratch (closed-form normal equations). Used as an
        independent, non-recursive ground truth."""
        X = np.array(self.X_hist)
        y = np.array(self.y_hist)
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        Xr, yr = X[mask], y[mask]
        A = Xr.T @ Xr + self.lam * np.eye(self.dim)
        b = Xr.T @ yr
        w_loo = np.linalg.solve(A, b)
        return w_loo

    # ------------------------------------------------------------------
    # Exact online unlearning (the downdate applied for real)
    # ------------------------------------------------------------------
    def unlearn(self, i):
        """Exactly remove sample i's contribution from the CURRENT (w, P),
        mutating them in place. This is the same closed-form downdate used
        to derive SIS, now actually applied rather than just scored. After
        this call, (self.w, self.P) are mathematically identical to what
        you would get by refitting ridge regression from scratch on every
        previously-seen sample except i. Cost: O(d^2), independent of n.
        """
        x = self.X_hist[i]
        y = self.y_hist[i]
        Px = self.P @ x
        h = x @ Px
        denom = 1.0 - h
        if abs(denom) < 1e-8:
            denom = 1e-8
        e = y - x @ self.w
        self.w = self.w - Px * e / denom
        self.P = self.P + np.outer(Px, Px) / denom

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)
