"""Experiment runner.  python run_experiments.py --split val|test [--B 2000]

Design is frozen before the test split is touched: models are fitted on folds 1-8 only; fold 9 is for development,
fold 10 is the final evaluation. No hyper-parameter search is performed.
Outputs (results/<split>/): clean_mr.csv, fault_ops.csv, fault_summary.csv, benign.csv, perf.csv, probs.npz
"""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import argparse, os, time, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import *
from ops import Ctx, prepare_ctx, all_ops, violation

D = f"{ROOT}/data"
TAUS = [0.02, 0.05, 0.10]
TAU0 = 0.05                      # primary tolerance
ALPHA = 0.05
N_BENIGN = 5


def boot_weights(cluster, B, rng):
    """Patient-cluster bootstrap: B x n matrix of record weights."""
    u, inv = np.unique(cluster, return_inverse=True)
    k = len(u)
    cnt = rng.multinomial(k, np.full(k, 1.0 / k), size=B)   # B x k cluster draw counts
    return cnt[:, inv].astype(np.float32)


def rate_ci(W, v, m, alpha=0.05):
    """Weighted violation rate among applicable rows with percentile CI."""
    num, den = W @ (v * m), W @ m
    r = num / np.maximum(den, 1e-9)
    pt = (v * m).sum() / max(m.sum(), 1)
    return pt, np.percentile(r, 100 * alpha / 2), np.percentile(r, 100 * (1 - alpha / 2))


def auc_w(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    out = f"{ROOT}/results/{a.split}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    ds = build_dataset(D)
    m = ds.meta
    X = ds.X.to_numpy()
    y, f, pid = m.y.to_numpy(), m.fold.to_numpy(), m.patient_id.to_numpy()
    tr = f <= 8
    ev = (f == 9) if a.split == "val" else (f == 10)
    assert not (set(pid[tr]) & set(pid[ev])), "patient leakage between train and eval"
    cols = list(ds.X.columns)
    rng = np.random.default_rng(SEED + (0 if a.split == "val" else 1))
    ctx = prepare_ctx(Ctx(ds.schema, m, SEED))
    ops = all_ops()
    Xe, ye, pe = X[ev], y[ev], pid[ev]
    ctx_e = prepare_ctx(Ctx(ds.schema, m.loc[ev] if False else m[ev], SEED))
    ne = ev.sum()
    W = boot_weights(pe, a.B, rng)
    print(f"split={a.split} n_train={tr.sum()} n_eval={ne} pos={ye.sum()} patients={len(set(pe))} B={a.B}", flush=True)

    raw = Raw(Xe.copy(), cols)
    # Apply each operator once to the raw eval batch (operators act on source-system values)
    t_ops = []
    for op in ops:
        r2, mask = op.apply(raw, ctx_e)
        t_ops.append((op, r2, mask.astype(np.float32)))

    def op_probs(pipe):
        p0 = pipe.proba(raw)
        P1 = np.stack([pipe.proba(r2) for _, r2, _ in t_ops], axis=1)
        return p0, P1

    def viol_matrix(p0, P1, tau):
        V = np.zeros_like(P1, dtype=np.float32)
        for j, (op, _, _) in enumerate(t_ops):
            V[:, j] = violation(op.kind, p0, P1[:, j], tau)
        return V

    M = np.stack([mk for _, _, mk in t_ops], axis=1)
    op_names = [o.name for o, _, _ in t_ops]
    op_fam = [o.family for o, _, _ in t_ops]
    op_kind = [o.kind for o, _, _ in t_ops]

    perf, clean_rows, fault_rows, summ_rows, benign_rows, benign_ops = [], [], [], [], [], []
    probs = {}
    models = {}
    for tn in TARGETS:
        t1 = time.time()
        T = Target(tn, ds.schema).fit(X[tr], y[tr])
        models[tn] = T
        pipe = Pipeline(T, None)
        p0, P1 = op_probs(pipe)
        probs[f"{tn}|clean|p0"], probs[f"{tn}|clean|P1"] = p0, P1
        auc0 = roc_auc_score(ye, p0)
        aw = np.array([auc_w(ye, p0, w) for w in W])
        perf.append(dict(target=tn, fault="clean", auc=auc0, lo=np.percentile(aw, 2.5), hi=np.percentile(aw, 97.5),
                         acc=((p0 > .5) == ye).mean()))
        Vc = {tau: viol_matrix(p0, P1, tau) for tau in TAUS}
        for tau in TAUS:
            for j in range(len(t_ops)):
                r, lo, hi = rate_ci(W, Vc[tau][:, j], M[:, j])
                clean_rows.append(dict(target=tn, op=op_names[j], family=op_fam[j], kind=op_kind[j], tau=tau,
                                       n_applicable=int(M[:, j].sum()), rate=r, lo=lo, hi=hi))
        # ---------------- injected faults
        for fn in FAULTS:
            pf = Pipeline(T, fn)
            q0, Q1 = op_probs(pf)
            probs[f"{tn}|{fn}|p0"], probs[f"{tn}|{fn}|P1"] = q0, Q1
            aucf = roc_auc_score(ye, q0)
            dA = np.array([auc_w(ye, q0, w) - auc_w(ye, p0, w) for w in W])
            perf.append(dict(target=tn, fault=fn, auc=aucf, lo=np.nan, hi=np.nan, acc=((q0 > .5) == ye).mean()))
            Vf = {tau: viol_matrix(q0, Q1, tau) for tau in TAUS}
            flagged = []
            for tau in TAUS:
                for j in range(len(t_ops)):
                    dv = Vf[tau][:, j] - Vc[tau][:, j]
                    num_f, den = W @ (Vf[tau][:, j] * M[:, j]), W @ M[:, j]
                    num_c = W @ (Vc[tau][:, j] * M[:, j])
                    dr = (num_f - num_c) / np.maximum(den, 1e-9)
                    pt = ((Vf[tau][:, j] - Vc[tau][:, j]) * M[:, j]).sum() / max(M[:, j].sum(), 1)
                    lo_b = np.percentile(dr, 100 * ALPHA / len(t_ops))     # one-sided, Bonferroni over operators
                    det = bool(lo_b > 0)
                    if tau == TAU0 and det:
                        flagged.append(op_names[j])
                    fault_rows.append(dict(target=tn, fault=fn, op=op_names[j], family=op_fam[j], tau=tau,
                                           rate_clean=((Vc[tau][:, j] * M[:, j]).sum() / max(M[:, j].sum(), 1)),
                                           rate_fault=((Vf[tau][:, j] * M[:, j]).sum() / max(M[:, j].sum(), 1)),
                                           delta=pt, lo_bonf=lo_b, detected=det))
            dA_pt = aucf - auc0
            acc_det = bool(np.percentile(dA, 97.5) < -0.01)        # CI of change in AUROC entirely below -0.01
            summ_rows.append(dict(target=tn, fault=fn, auc_clean=auc0, auc_fault=aucf, dAUC=dA_pt,
                                  dAUC_lo=np.percentile(dA, 2.5), dAUC_hi=np.percentile(dA, 97.5),
                                  acc_detect=acc_det, mr_detect=len(flagged) > 0, n_ops_flagged=len(flagged),
                                  ops_flagged=";".join(flagged)))
        # ---------------- benign retrains (false-alarm control): patient-cluster resample of the training set
        tr_ix = np.where(tr)[0]
        for b in range(N_BENIGN):
            rb = np.random.default_rng([SEED, 77, b])
            upid = np.unique(pid[tr_ix])
            draw = rb.choice(upid, size=len(upid), replace=True)
            cnt = pd.Series(draw).value_counts()
            idx = np.concatenate([np.repeat(tr_ix[pid[tr_ix] == q], c) for q, c in cnt.items()])
            Tb = Target(tn, ds.schema).fit(X[idx], y[idx])
            pb = Pipeline(Tb, None)
            b0, B1 = op_probs(pb)
            aucb = roc_auc_score(ye, b0)
            Vb = viol_matrix(b0, B1, TAU0)
            flagged = []
            for j in range(len(t_ops)):
                num_b, den = W @ (Vb[:, j] * M[:, j]), W @ M[:, j]
                num_c = W @ (Vc[TAU0][:, j] * M[:, j])
                dr = (num_b - num_c) / np.maximum(den, 1e-9)
                lo_b = np.percentile(dr, 100 * ALPHA / len(t_ops))
                dpt = ((Vb[:, j] - Vc[TAU0][:, j]) * M[:, j]).sum() / max(M[:, j].sum(), 1)
                benign_ops.append(dict(target=tn, replicate=b, op=op_names[j], family=op_fam[j], delta=dpt, lo_bonf=lo_b))
                if lo_b > 0:
                    flagged.append(op_names[j])
            dA = np.array([auc_w(ye, b0, w) - auc_w(ye, p0, w) for w in W])
            benign_rows.append(dict(target=tn, replicate=b, auc=aucb, dAUC=aucb - auc0,
                                    acc_detect=bool(np.percentile(dA, 97.5) < -0.01),
                                    mr_detect=len(flagged) > 0, ops_flagged=";".join(flagged)))
        print(f"  {tn} done in {time.time()-t1:.0f}s  AUROC {auc0:.4f}", flush=True)

    pd.DataFrame(perf).to_csv(f"{out}/perf.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(f"{out}/clean_mr.csv", index=False)
    pd.DataFrame(fault_rows).to_csv(f"{out}/fault_ops.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(f"{out}/fault_summary.csv", index=False)
    pd.DataFrame(benign_rows).to_csv(f"{out}/benign.csv", index=False)
    pd.DataFrame(benign_ops).to_csv(f"{out}/benign_ops.csv", index=False)
    np.savez_compressed(f"{out}/probs.npz", y=ye, pid=pe, ecg_id=m.index.to_numpy()[ev], **probs)
    json.dump(dict(split=a.split, n_eval=int(ne), n_pos=int(ye.sum()), n_patients=int(len(set(pe))), B=a.B,
                   seed=SEED, tau_primary=TAU0, taus=TAUS, alpha=ALPHA, n_ops=len(ops), runtime_s=time.time() - t0),
              open(f"{out}/run_info.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
