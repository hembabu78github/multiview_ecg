"""Exploratory: sensitivity of the cost analysis to the missed-MI : false-alarm cost ratio and to the action threshold."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pickle, numpy as np, pandas as pd
from run_experiments import boot_weights
OUT = f"{ROOT}/results/ext"
Z = np.load(f"{OUT}/scen_fold10.npz", allow_pickle=True); meta = pickle.load(open(f"{OUT}/scen_meta.pkl", "rb"))
y = Z["y"].astype(float); B = 300
W = boot_weights(Z["pid"], B, np.random.default_rng(31337)); one = np.ones_like(y)
def cost(p, t, w, r):
    n = w.sum(); fn = (w * ((p < t) & (y == 1))).sum(); fp = (w * ((p >= t) & (y == 0))).sum(); return (r * fn + fp) / n * 1000
rows = []
for t in ["E-LR", "E-HGB", "L-LR", "L-HGB", "S-HGB"]:
    pc = Z[f"clean|{t}|p0"].astype(float)
    for thr in (0.05, 0.10, 0.15, 0.30):
        for r in (1, 5, 10, 20, 50):
            c0 = cost(pc, thr, one, r); bc = np.array([cost(pc, thr, W[b], r) for b in range(B)]); ninc = ndec = 0; n = 0
            for s in meta["scen"]:
                if s == "clean" or s.startswith("ALTUNITS_OK"): continue
                pf = Z[f"{s}|{t}|p0"].astype(float); d = np.array([cost(pf, thr, W[b], r) for b in range(B)]) - bc
                lo, hi = np.percentile(d, [2.5, 97.5]); n += 1; ninc += lo > 0; ndec += hi < 0
            rows.append(dict(target=t, thr=thr, ratio=r, clean_cost=c0, n_scen=n, n_cost_up=int(ninc), n_cost_down=int(ndec)))
pd.DataFrame(rows).to_csv(f"{OUT}/cost_ratio_sensitivity.csv", index=False)
