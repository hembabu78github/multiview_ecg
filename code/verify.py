"""Independent re-computation of the headline numbers from the saved per-record probabilities (probs.npz).
Uses only numpy/pandas/sklearn (no project code) and a different bootstrap implementation + seed.
python verify.py test
"""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

split = sys.argv[1] if len(sys.argv) > 1 else "test"
R = f"{ROOT}/results/{split}"
Z = np.load(f"{R}/probs.npz", allow_pickle=True)
info = json.load(open(f"{R}/run_info.json"))
y, pid = Z["y"], Z["pid"]
perf = pd.read_csv(f"{R}/perf.csv")
clean = pd.read_csv(f"{R}/clean_mr.csv")
summ = pd.read_csv(f"{R}/fault_summary_gated.csv")
ops = list(pd.read_csv(f"{R}/clean_mr.csv").op.unique())
kinds = clean.drop_duplicates("op").set_index("op").kind.to_dict()
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail)
    if not ok:
        fails.append(name)


# 1. AUROC point estimates
for t in perf.target.unique():
    p0 = Z[f"{t}|clean|p0"]
    a = roc_auc_score(y, p0)
    rep = perf[(perf.target == t) & (perf.fault == "clean")].auc.iloc[0]
    check(f"AUROC {t}", abs(a - rep) < 1e-9, f"{a:.4f}")

# 2. violation rates recomputed by brute force for every target x op at tau=.05 (point estimates)
mx = 0
meta_terr = None
for t in perf.target.unique():
    p0, P1 = Z[f"{t}|clean|p0"], Z[f"{t}|clean|P1"]
    for j, op in enumerate(ops):
        d = P1[:, j] - p0
        k = kinds[op]
        v = np.abs(d) > .05 if k == "invariance" else (d > .05 if k == "nonincrease" else d < -.05)
        row = clean[(clean.target == t) & (clean.op == op) & (clean.tau == .05)].iloc[0]
        if op.startswith("TERRITORY"):
            # applicable rows = rows whose probability changed (masking applied) among positives; compare via n_applicable
            m = row.n_applicable
            check(f"{t} {op} applicable count plausible", 0 < m <= (y == 1).sum(), str(int(m)))
            continue
        mx = max(mx, abs(v.mean() - row.rate))
check("violation rates (all non-territory ops) match brute force (float32 tolerance)", mx < 1e-6, f"max diff {mx:.2e}")

# 3. invariants that must hold by construction
for t in perf.target.unique():
    P1 = Z[f"{t}|clean|P1"]; p0 = Z[f"{t}|clean|p0"]
    for op in ("UNIT_AMP_uV", "UNIT_TIME_s", "UNIT_ANTHRO_in_lb"):
        j = ops.index(op)
        d = np.abs(P1[:, j] - p0)
        if t.endswith("LR"):
            check(f"{t} {op} exactly invariant under correct harmonizer", d.max() < 1e-9, f"max {d.max():.1e}")
        else:
            # tree splits can sit exactly on a data value; 1-ulp round-off in the unit round-trip may flip a few records
            check(f"{t} {op} invariant up to float round-off (<=0.5% of records move >0.05)", (d > .05).mean() <= .005,
                  f"{(d > .05).sum()} records")
    check(f"{t} probabilities in [0,1] and finite", np.isfinite(p0).all() and (p0 >= 0).all() and (p0 <= 1).all())

# 4. independent cluster bootstrap of dAUROC for each fault (different RNG and implementation)
rng = np.random.default_rng(987654321)
u, inv = np.unique(pid, return_inverse=True)
by_cluster = [np.where(inv == k)[0] for k in range(len(u))]
B = 400
idxs = [np.concatenate([by_cluster[k] for k in rng.integers(0, len(u), len(u))]) for _ in range(B)]
mism = 0
for _, r in summ.iterrows():
    p0, pf = Z[f"{r.target}|clean|p0"], Z[f"{r.target}|{r.fault}|p0"]
    d = np.array([roc_auc_score(y[i], pf[i]) - roc_auc_score(y[i], p0[i]) for i in idxs])
    hi = np.percentile(d, 97.5)
    acc_det = hi < -0.01
    if acc_det != bool(r.acc_detect):
        # allow only borderline disagreements (within 0.003 of the threshold)
        if abs(hi + 0.01) > 0.003:
            mism += 1
            print("   accuracy-detection disagreement", r.target, r.fault, round(hi, 4))
check("accuracy-only detection reproduced by independent bootstrap", mism == 0, f"{mism} non-borderline disagreements")

# 5. faulty pipelines: a fault must not change predictions on the *unmodified* data for unit faults
for t in perf.target.unique():
    p0 = Z[f"{t}|clean|p0"]
    for fl in ("F1_AMP_UNIT_IGNORED", "F2_TIME_UNIT_IGNORED", "F3_ANTHRO_UNIT_IGNORED"):
        check(f"{t} {fl} silent on canonical-unit data", np.abs(Z[f"{t}|{fl}|p0"] - p0).max() < 1e-9)

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
