"""Reference statistics (fitted on training folds only) for the baseline monitors."""
import numpy as np
from common import *
NB = 20


class Reference:
    def fit(self, X, schema):
        self.schema = schema
        self.view_of = schema.view.to_numpy()
        self.groups = {v: np.where(self.view_of == v)[0] for v in VIEWS + ["struct"]}
        with np.errstate(all="ignore"):
            self.lo = np.nanpercentile(X, 0.05, axis=0); self.hi = np.nanpercentile(X, 99.95, axis=0)
        self.lo = np.where(np.isnan(self.lo), -np.inf, self.lo); self.hi = np.where(np.isnan(self.hi), np.inf, self.hi)
        self.edges, nb = [], []
        for j in range(X.shape[1]):
            v = X[:, j]; v = v[np.isfinite(v)]
            e = np.unique(np.quantile(v, np.linspace(0, 1, NB + 1)[1:-1])) if len(v) > 5 else np.array([])
            self.edges.append(e); nb.append(len(e) + 2)       # bins 0..len(e), plus NaN bin
        self.nb = np.array(nb); self.W = int(self.nb.max())
        B = self.bins(X); self.ref_p = self.counts(B) / len(X)
        self.miss = {v: np.isnan(X[:, ix]).mean() for v, ix in self.groups.items()}
        self.prep = Prep().fit(X)
        rng = np.random.default_rng(SEED); sub = rng.choice(len(X), min(6000, len(X)), replace=False)
        Z = self.prep.transform(X[sub]); self.mu = Z.mean(0)
        U, s, Vt = np.linalg.svd(Z - self.mu, full_matrices=False)
        k = 30; self.comp = Vt[:k].T; self.sd = s[:k] / np.sqrt(len(sub) - 1)
        return self

    def bins(self, X):
        B = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges):
            b = np.searchsorted(e, X[:, j], side="right"); b[np.isnan(X[:, j])] = len(e) + 1; B[:, j] = b
        return B

    def counts(self, B):
        F = B.shape[1]; idx = (np.arange(F)[None, :] * self.W + B).ravel()
        return np.bincount(idx, minlength=F * self.W).reshape(F, self.W).astype(float)

    def per_record(self, X):
        """Everything the monitors need, per record."""
        out = dict(bins=self.bins(X))
        oor = (X < self.lo) | (X > self.hi)
        fin = np.isfinite(X)
        out["oor"] = np.stack([(oor[:, ix]).sum(1) / np.maximum(fin[:, ix].sum(1), 1) for v, ix in self.groups.items()], 1).astype(np.float32)
        out["miss"] = np.stack([np.isnan(X[:, ix]).mean(1) for v, ix in self.groups.items()], 1).astype(np.float32)
        out["pcs"] = (((self.prep.transform(X) - self.mu) @ self.comp) / self.sd).astype(np.float32)
        return out

    def psi(self, B):
        p = np.maximum(self.counts(B) / len(B), 1e-4); q = np.maximum(self.ref_p, 1e-4)
        mask = np.arange(self.W)[None, :] < self.nb[:, None]
        return float(((((p - q) * np.log(p / q)) * mask).sum(1)).mean())
