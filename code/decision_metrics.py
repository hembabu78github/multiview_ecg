"""A5: calibration and decision metrics before/after faults (fold 10), with patient-cluster bootstrap CIs for the change vs clean.
Threshold sets: default 0.5 and decision-curve thresholds 0.1/0.2/0.3/0.5. The clinician-chosen threshold (review form Q3) is added when returned."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from run_experiments import boot_weights
OUT = f"{ROOT}/results/ext"
Z = np.load(f"{OUT}/scen_fold10.npz", allow_pickle=True); meta = pickle.load(open(f"{OUT}/scen_meta.pkl", "rb"))
y = Z["y"].astype(float); pid = Z["pid"]; B = 500
W = boot_weights(pid, B, np.random.default_rng(777)); Wn = W.sum(1)
PTS = (0.1, 0.2, 0.3, 0.5)
def stats(p, w=None):
    w = np.ones_like(y) if w is None else w; n = w.sum(); out = {}
    out["brier"] = (w * (p - y) ** 2).sum() / n
    for pt in PTS:
        tp = (w * ((p >= pt) & (y == 1))).sum(); fp = (w * ((p >= pt) & (y == 0))).sum()
        out[f"nb@{pt}"] = tp / n - fp / n * pt / (1 - pt)
    t = 0.5; P = (w * (y == 1)).sum(); N = (w * (y == 0)).sum()
    out["sens@0.5"] = (w * ((p >= t) & (y == 1))).sum() / P; out["spec@0.5"] = (w * ((p < t) & (y == 0))).sum() / N
    return out
def boot_delta(pf, pc):
    d = {}
    base = {k: v for k, v in stats(pc).items()}; fault = stats(pf)
    bs = [ (stats(pf, W[b]), stats(pc, W[b])) for b in range(B)]
    for k in fault:
        dd = np.array([a[k] - c[k] for a, c in bs]); d[k] = fault[k] - base[k]; d[k + "_lo"], d[k + "_hi"] = np.percentile(dd, [2.5, 97.5])
    return d
rows = []
for t in ["E-LR", "E-HGB", "L-LR", "L-HGB", "S-HGB"]:
    pc = Z[f"clean|{t}|p0"].astype(float)
    for s in meta["scen"]:
        pf = Z[f"{s}|{t}|p0"].astype(float)
        r = dict(target=t, scenario=s, auroc=roc_auc_score(y, pf), auprc=average_precision_score(y, pf), **{f"{k}": v for k, v in stats(pf).items()})
        bins = np.minimum((pf * 10).astype(int), 9); r["ece"] = sum(abs(y[bins == k].mean() - pf[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
        if s != "clean": r.update({"d_" + k: v for k, v in boot_delta(pf, pc).items()})
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/decision_metrics.csv", index=False); print("done")
