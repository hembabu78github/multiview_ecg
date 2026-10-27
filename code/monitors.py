"""A1/A2/A8: batch-level monitors compared at an equal 5% batch false-alarm budget.
Reference data: training folds (drift/range/missingness) and half R of fold 9 (model-based references).
Null distribution / thresholds: half C of fold 9.  Evaluation: fold 10 (never used for calibration)."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys, pickle, numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
ALPHA = 0.05
OUT = f"{ROOT}/results/ext"
Z9 = np.load(f"{OUT}/scen_fold9.npz", allow_pickle=True); Z10 = np.load(f"{OUT}/scen_fold10.npz", allow_pickle=True)
meta = pickle.load(open(f"{OUT}/scen_meta.pkl", "rb")); ops, kinds = meta["ops"], meta["kinds"]
ref = pickle.load(open(f"{ROOT}/results/models/reference.pkl", "rb"))
TARGETS = ["E-LR", "E-HGB", "L-LR", "L-HGB", "S-HGB"]
TAU = 0.05
nview = len(ref.groups)
LF = [j for j, o in enumerate(ops) if not o.startswith('TERRITORY')]   # label-free operators only (territory ops need the true MI label)

def viol(P1, p0):
    d = P1 - p0[:, None]; V = np.empty(d.shape, bool)
    for j, k in enumerate(kinds):
        V[:, j] = np.abs(d[:, j]) > TAU if k == "invariance" else (d[:, j] > TAU if k == "nonincrease" else d[:, j] < -TAU)
    return V

# split fold 9 by patient into reference half R and calibration half C
rng = np.random.default_rng(20261025)
pid9 = Z9["pid"]; up = np.unique(pid9); Rp = set(rng.choice(up, len(up) // 2, replace=False))
isR = np.array([p in Rp for p in pid9]); R_ix, C_ix = np.where(isR)[0], np.where(~isR)[0]
M9, M10 = Z9["M"].astype(float), Z10["M"].astype(float)

class Src:
    """per-scenario record-level arrays for one fold"""
    def __init__(self, Z, M, s):
        self.Z, self.M, self.s = Z, M, s
        self.bins, self.oor, self.miss, self.pcs = (Z[f"{s}|mon|{k}"] for k in ("bins", "oor", "miss", "pcs"))
    def tgt(self, t):
        p0, P1 = self.Z[f"{self.s}|{t}|p0"].astype(float), self.Z[f"{self.s}|{t}|P1"].astype(float)
        return p0, viol(P1, p0)

# ---- references (clean pipeline, fold 9): from half R, half C and the full fold
REFS = {}
c9 = Src(Z9, M9, "clean")
for t in TARGETS:
    p0, V = c9.tgt(t)
    REFS[t] = {}
    for nm, ix in (("R", R_ix), ("C", C_ix), ("full", np.arange(len(p0)))):
        REFS[t][nm] = dict(p=p0[ix], auc=roc_auc_score(Z9["y"][ix], p0[ix]),
                           r=(V[ix] * M9[ix]).sum(0) / np.maximum(M9[ix].sum(0), 1))
ref_miss = np.array([ref.miss[v] for v in ref.groups])

def batch_stats(src, y, t, idx, which="full"):
    """idx: K x n matrix of record indices. returns dict monitor -> K array (higher = more alarming)"""
    K, n = idx.shape; out = {}
    # range validator: a record fails if >10% of the observed features of any ECG view lie outside the training 0.05-99.95 percentile range
    out["SCHEMA"] = (src.oor[idx][:, :, :3] > 0.10).any(2).mean(1)
    out["MISS"] = np.abs(src.miss[idx].mean(1) - ref_miss).max(1)
    out["PSI"] = np.array([ref.psi(src.bins[i]) for i in idx])
    out["HOT"] = n * (src.pcs[idx].mean(1) ** 2).sum(1)
    if t is not None:
        p0, V = src.tgt(t)
        out["PRED"] = np.array([ks_2samp(p0[i], REFS[t][which]["p"]).statistic for i in idx])
        Mi = src.M[idx]; rate = (V[idx] * Mi).sum(1) / np.maximum(Mi.sum(1), 1)
        out["MR"] = (rate - REFS[t][which]["r"])[:, LF].max(1)
        a = []
        for i in idx:
            yy = y[i]; a.append(REFS[t][which]["auc"] - roc_auc_score(yy, p0[i]) if 0 < yy.sum() < len(yy) else 0.0)
        out["ACC"] = np.array(a)
    return out

def draw(rng, pool, K, n): return rng.choice(pool, size=(K, n), replace=True)

def run(n, K, seed=1):
    rng = np.random.default_rng([20261025, n, seed])
    yC = Z9["y"]; thr = {}
    N9 = len(yC); allix = np.arange(N9)
    # training-referenced monitors: null from the full fold 9
    st_free = {t: None for t in TARGETS}
    base = batch_stats(c9, yC, None, draw(rng, allix, 2000, n))
    # SCHEMA: rare-event statistic -> Poisson null with smoothed failure rate from fold 9
    from scipy.stats import poisson
    p_fail = ((c9.oor[:, :3] > 0.10).any(1).sum() + 0.5) / (N9 + 1)
    base["SCHEMA"] = np.array([poisson.ppf(1 - ALPHA, n * p_fail) / n])  # only its 95th pct is used below
    for t in TARGETS:
        # model-referenced monitors: cross-fitted null (reference from one half, batches from the other), pooled
        sR = batch_stats(c9, yC, t, draw(rng, C_ix, 1000, n), "R"); sC = batch_stats(c9, yC, t, draw(rng, R_ix, 1000, n), "C")
        st = {k: (np.concatenate([sR[k], sC[k]]) if k in ("PRED", "MR", "ACC") else base[k]) for k in sR}
        thr[t] = {k: np.quantile(v, 1 - ALPHA) for k, v in st.items()}
        thr[t]["SCHEMA"] = float(poisson.ppf(1 - ALPHA, n * p_fail) / n)
        for name, mem in (("STACK", ["SCHEMA", "MISS", "PSI", "HOT", "PRED"]), ("MR+VAL", ["MR", "SCHEMA", "MISS"]), ("ALL", ["MR", "SCHEMA", "MISS", "PSI", "HOT", "PRED"])):
            thr[t][name] = {k: (float(poisson.ppf(1 - ALPHA / len(mem), n * p_fail) / n) if k == "SCHEMA" else np.quantile(st[k], 1 - ALPHA / len(mem))) for k in mem}
    def flags(st, t):
        f = {k: st[k] > thr[t][k] for k in st}
        for name in ("STACK", "MR+VAL", "ALL"): f[name] = np.any([st[k] > thr[t][name][k] for k in thr[t][name]], axis=0)
        return f
    rows_det, rows_fa = [], []
    yT = Z10["y"]
    # clean + alt-units-ok + shifted-population false alarms
    age, sex, dev = Z10["age"], Z10["sex"], Z10["device"]
    pops = {"all": np.ones(len(yT), bool), "age>=65": age >= 65, "age<50": age < 50, "female": sex == 0, "male": sex == 1}
    top = pd.Series(dev).value_counts().index[0]; pops["device_top"] = dev == top; pops["device_other"] = dev != top
    for sname in ["clean", "ALTUNITS_OK_AMP", "ALTUNITS_OK_TIME", "ALTUNITS_OK_ANTHRO"]:
        src = Src(Z10, M10, sname)
        for pn, mk in pops.items():
            if sname != "clean" and pn != "all": continue
            pool = np.where(mk)[0]; idx = draw(rng, pool, K, n)
            for t in TARGETS:
                for k, v in flags(batch_stats(src, yT, t, idx), t).items():
                    rows_fa.append(dict(n=n, scenario=sname, pop=pn, target=t, monitor=k, rate=float(v.mean())))
    for s in meta["scen"]:
        if s == "clean" or s.startswith("ALTUNITS"): continue
        src = Src(Z10, M10, s); idx = draw(rng, np.arange(len(yT)), K, n)
        for t in TARGETS:
            for k, v in flags(batch_stats(src, yT, t, idx), t).items():
                rows_det.append(dict(n=n, scenario=s, family=meta["FX"][s]["family"], group=meta["FX"][s]["group"], sev=meta["FX"][s]["sev"],
                                     target=t, monitor=k, det=float(v.mean())))
    import json
    json.dump({t: {k: (v if not isinstance(v, dict) else {a: float(b) for a, b in v.items()}) for k, v in d.items()} for t, d in thr.items()}, open(f"{OUT}/monitor_thresholds_n{n}.json", "w"), default=float, indent=1)
    return pd.DataFrame(rows_det), pd.DataFrame(rows_fa)

if __name__ == "__main__":
    allD, allF = [], []
    for n, K in ((200, 500), (50, 200), (100, 200), (400, 200)):
        d, f = run(n, K); allD.append(d); allF.append(f); print("n", n, "done", flush=True)
    pd.concat(allD).to_csv(f"{OUT}/monitor_det.csv", index=False); pd.concat(allF).to_csv(f"{OUT}/monitor_fa.csv", index=False)
