"""A7: quality-aware late fusion (view completeness x plausibility, with validation fallback) and fault-aware abstention.
All design constants are prespecified: eligibility q>=0.5, plausibility penalty 10x out-of-range fraction, abstention budget 5% on clean fold 9.
Per-view Platt calibration and the abstention threshold are fitted on fold 9 (clean) only; fold 10 is evaluation."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common import *
from ops import Ctx, prepare_ctx, _convert, MaskViewOp
from faults2 import FX, PipelineX
D = f"{ROOT}/data"; OUT = f"{ROOT}/results/ext"
ds = build_dataset(D); m = ds.meta; X = ds.X.to_numpy(); y = m.y.to_numpy(); f = m.fold.to_numpy(); cols = list(ds.X.columns)
models = pickle.load(open(f"{ROOT}/results/models/targets_f1-8.pkl", "rb"))
ref = pickle.load(open(f"{ROOT}/results/models/reference.pkl", "rb"))
G = [ref.groups[v] for v in VIEWS]; cref = np.array([1 - ref.miss[v] for v in VIEWS])
Q_MIN, PEN = 0.5, 10.0

def qa(Xc):
    c = np.stack([np.isfinite(Xc[:, ix]).mean(1) for ix in G], 1)
    fin = np.stack([np.isfinite(Xc[:, ix]).sum(1) for ix in G], 1)
    oor = np.stack([(((Xc[:, ix] < ref.lo[ix]) | (Xc[:, ix] > ref.hi[ix])).sum(1)) for ix in G], 1) / np.maximum(fin, 1)
    return np.minimum(1, c / cref) * (1 - np.minimum(1, PEN * oor))

def stream(fold, s, extra=None):
    ev = f == fold; ctx = prepare_ctx(Ctx(ds.schema, m[ev], SEED)); ctx.cols_arr = np.array(cols)
    raw = Raw(X[ev].copy(), cols)
    if s != "clean" and FX[s]["tags"]: raw = _convert(raw, ctx, **FX[s]["tags"])
    if extra is not None: raw, _ = extra.apply(raw, ctx)
    return raw, y[ev]

def parts(T, s, raw):
    pipe = PipelineX(T, None if s == "clean" else s); Xc = pipe.canonical(raw)
    _, (P, A) = T.logit(Xc, return_parts=True); return P, A, qa(Xc), Xc

class Fuser:
    def __init__(self, T):
        self.T = T; raw, yy = stream(9, "clean"); P, A, Q, _ = parts(T, "clean", raw)
        self.cal = [LogisticRegression(C=1e4).fit(P[i][:, None], yy) for i in range(3)]
        Zc = self.zc(P); self.delta = np.quantile(self.disagree(Zc, np.ones_like(A, bool)), 0.95)
    def zc(self, P): return np.stack([self.cal[i].coef_[0, 0] * P[i] + self.cal[i].intercept_[0] for i in range(3)])
    def disagree(self, Zc, E):
        hi = np.where(E, Zc, -np.inf).max(0); lo = np.where(E, Zc, np.inf).min(0)
        return np.where(E.sum(0) >= 2, hi - lo, 0.0)
    def variants(self, P, A, Q):
        Zc = self.zc(P); out = {}
        wL = A.astype(float); out["L"] = np.where(wL.sum(0) > 0, (wL * P).sum(0) / np.maximum(wL.sum(0), 1), np.nan)
        out["Lcal"] = np.where(wL.sum(0) > 0, (wL * Zc).sum(0) / np.maximum(wL.sum(0), 1), np.nan)
        E = A & (Q.T >= Q_MIN); w = E * Q.T; s = w.sum(0)
        zq = np.where(s > 0, (w * Zc).sum(0) / np.maximum(s, 1e-9), np.nan); out["QL"] = zq
        dis = self.disagree(Zc, E); out["QL+abst"] = np.where(dis > self.delta, np.nan, zq); out["_dis"] = dis
        return out

def metrics(z, yy, thr=0.5):
    cov = np.isfinite(z); r = dict(coverage=cov.mean())
    if cov.sum() > 20 and 0 < yy[cov].sum() < cov.sum():
        p = 1 / (1 + np.exp(-z[cov])); yc = yy[cov]
        r.update(auc=roc_auc_score(yc, p), auprc=average_precision_score(yc, p), brier=brier_score_loss(yc, p), err=((p > thr) != yc).mean(),
                 miss_rate=((p <= thr) & (yc == 1)).sum() / max(yc.sum(), 1), fa_rate=((p > thr) & (yc == 0)).sum() / max((yc == 0).sum(), 1))
        b = np.minimum((p * 10).astype(int), 9); r["ece"] = sum(abs(yc[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())
    return r

if __name__ == "__main__":
    rows = []; viewmask = {v: MaskViewOp(v) for v in VIEWS}
    for tn in ("L-HGB", "L-LR"):
        T = models[tn]; Fu = Fuser(T)
        for s in ["clean"] + [k for k in FX if not k.startswith("ALTUNITS")]:
            raw, yy = stream(10, s); P, A, Q, _ = parts(T, s, raw); V = Fu.variants(P, A, Q)
            for vn, z in V.items():
                if vn.startswith("_"): continue
                rows.append(dict(target=tn, scenario=s, variant=vn, **metrics(z, yy)))
        # view-dropout fragility of each fusion variant on clean data (violation of |dp|<=0.05)
        raw, yy = stream(10, "clean"); P0, A0, Q0, _ = parts(T, "clean", raw); V0 = Fu.variants(P0, A0, Q0)
        for v in VIEWS:
            raw2, _ = stream(10, "clean", viewmask[v]); P1, A1, Q1, _ = parts(T, "clean", raw2); V1 = Fu.variants(P1, A1, Q1)
            for vn in ("L", "Lcal", "QL", "QL+abst"):
                a, b = 1 / (1 + np.exp(-V0[vn])), 1 / (1 + np.exp(-V1[vn])); ok = np.isfinite(a) & np.isfinite(b)
                rows.append(dict(target=tn, scenario="MASK_VIEW_" + v, variant=vn, viol_dropview=float((np.abs(a - b)[ok] > .05).mean()), coverage=float(ok.mean())))
    pd.DataFrame(rows).to_csv(f"{OUT}/fusion_qa.csv", index=False); print("done")
