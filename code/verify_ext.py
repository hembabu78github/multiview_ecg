"""Independent re-computation of extended results from saved per-record arrays (no project modules except the saved reference object for
feature ranges). Different RNG and a plain-loop implementation. python verify_ext.py"""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import json, pickle, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
E = f"{ROOT}/results/ext"
Z = np.load(f"{E}/scen_fold10.npz", allow_pickle=True); meta = pickle.load(open(f"{E}/scen_meta.pkl", "rb"))
det = pd.read_csv(f"{E}/monitor_det.csv"); det = det[det.n == 200]
dm = pd.read_csv(f"{E}/decision_metrics.csv"); thr = json.load(open(f"{E}/monitor_thresholds_n200.json"))
ops, kinds = meta["ops"], meta["kinds"]; LF = [j for j, o in enumerate(ops) if not o.startswith("TERRITORY")]
y = Z["y"]; M = Z["M"].astype(float); fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail)
    if not ok: fails.append(name)
# 1. decision metrics
for t in ["E-LR", "E-HGB", "L-LR", "L-HGB", "S-HGB"]:
    for s in ("clean", "SHIFTCOL_12sl", "GAIN_0.5", "ACT_F2_TIME"):
        p = Z[f"{s}|{t}|p0"].astype(float); r = dm[(dm.target == t) & (dm.scenario == s)].iloc[0]
        check(f"AUROC/Brier {t} {s}", abs(roc_auc_score(y, p) - r.auroc) < 1e-9 and abs(brier_score_loss(y, p) - r.brier) < 1e-9)
# 2. independent MR / ACC monitor detection rates (fold-9 references recomputed from fold 9 clean arrays)
Z9 = np.load(f"{E}/scen_fold9.npz", allow_pickle=True); y9 = Z9["y"]; M9 = Z9["M"].astype(float)
def viol(P1, p0):
    out = np.zeros(P1.shape, bool)
    for j, k in enumerate(kinds):
        d = P1[:, j] - p0
        out[:, j] = (np.abs(d) > .05) if k == "invariance" else ((d > .05) if k == "nonincrease" else (d < -.05))
    return out
rng = np.random.default_rng(424242); K = 400; n = 200; worst = 0
for t in ["E-HGB", "L-LR"]:
    p9 = Z9[f"clean|{t}|p0"].astype(float); V9 = viol(Z9[f"clean|{t}|P1"].astype(float), p9)
    r_ref = (V9 * M9).sum(0) / np.maximum(M9.sum(0), 1); auc_ref = roc_auc_score(y9, p9)
    for s in ["F1_AMP_UNIT_IGNORED", "F4_SWAP_V1_V2", "F6_MISSING_ZERO_FILLED", "MISS_50", "SHIFTCOL_12sl", "ACT_F2_TIME", "TRUNC_30"]:
        p = Z[f"{s}|{t}|p0"].astype(float); V = viol(Z[f"{s}|{t}|P1"].astype(float), p)
        hit_mr = hit_acc = 0
        for _ in range(K):
            i = rng.integers(0, len(y), n)
            rate = (V[i] * M[i]).sum(0) / np.maximum(M[i].sum(0), 1)
            hit_mr += (rate - r_ref)[LF].max() > thr[t]["MR"]
            yy = y[i]; hit_acc += (auc_ref - roc_auc_score(yy, p[i]) > thr[t]["ACC"]) if 0 < yy.sum() < n else 0
        for mon, hit in (("MR", hit_mr), ("ACC", hit_acc)):
            rep = det[(det.scenario == s) & (det.target == t) & (det.monitor == mon)].det.iloc[0]
            worst = max(worst, abs(hit / K - rep))
check("independent MR/ACC monitor detection rates agree with reported (|diff| <= 0.09)", worst <= 0.09, f"worst {worst:.3f}")
# 3. dormant faults silent on canonical-unit stream; correct pipeline benign on alt-unit streams
for t in ["E-LR", "E-HGB", "L-LR", "L-HGB", "S-HGB"]:
    p0 = Z[f"clean|{t}|p0"].astype(float)
    for s in ("F1_AMP_UNIT_IGNORED", "F2_TIME_UNIT_IGNORED", "F3_ANTHRO_UNIT_IGNORED"):
        check(f"{t} {s} dormant", np.abs(Z[f"{s}|{t}|p0"].astype(float) - p0).max() < 1e-6)
    for s in ("ALTUNITS_OK_AMP", "ALTUNITS_OK_TIME", "ALTUNITS_OK_ANTHRO"):
        d = np.abs(Z[f"{s}|{t}|p0"].astype(float) - p0)
        check(f"{t} {s} benign (<=0.5% of records move >0.05)", (d > .05).mean() <= .005, f"{(d > .05).sum()} recs")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}"); sys.exit(1 if fails else 0)
