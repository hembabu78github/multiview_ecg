"""A4: large model-update false-alarm study. 50 patient-bootstrap retrains + 50 80%-patient-subsample retrains per target.
Each retrained model is compared with the original on the test fold (stale-reference scenario)."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import *
from ops import Ctx, prepare_ctx, all_ops, violation
D = f"{ROOT}/data"; OUT = f"{ROOT}/results/ext"
NR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ds = build_dataset(D); m = ds.meta; X = ds.X.to_numpy(); y = m.y.to_numpy(); f = m.fold.to_numpy(); pid = m.patient_id.to_numpy()
tr_ix = np.where(f <= 8)[0]; ev = f == 10
ctx = prepare_ctx(Ctx(ds.schema, m[ev], SEED)); ops = all_ops()
raw = Raw(X[ev].copy(), list(ds.X.columns)); ye = y[ev]
T_ops = [op.apply(raw, ctx) for op in ops]; M = np.stack([mk for _, mk in T_ops], 1).astype(float)
names = [o.name for o in ops]
def rates(pipe):
    p0 = pipe.proba(raw); P1 = np.stack([pipe.proba(r) for r, _ in T_ops], 1)
    V = np.stack([violation(o.kind, p0, P1[:, j], .05) for j, o in enumerate(ops)], 1).astype(float)
    return p0, (V * M).sum(0) / np.maximum(M.sum(0), 1)
import pickle
orig = pickle.load(open(f"{ROOT}/results/models/targets_f1-8.pkl", "rb"))
rows = []; t0 = time.time()
upid = np.unique(pid[tr_ix])
for tn in ["S-HGB","L-HGB"]:
    p0o, r_o = rates(Pipeline(orig[tn], None)); a_o = roc_auc_score(ye, p0o)
    for b in range(NR):
        rb = np.random.default_rng([SEED, 4242, b])
        if b < NR // 2:
            draw = rb.choice(upid, size=len(upid), replace=True); kind = "bootstrap"
        else:
            draw = rb.choice(upid, size=int(.8 * len(upid)), replace=False); kind = "subsample80"
        cnt = pd.Series(draw).value_counts()
        by = pd.Series(np.arange(len(tr_ix)), index=pid[tr_ix]).groupby(level=0).apply(list)
        idx = np.concatenate([np.repeat(tr_ix[by[q]], c) for q, c in cnt.items()])
        Tb = Target(tn, ds.schema).fit(X[idx], y[idx])
        p0, r = rates(Pipeline(Tb, None))
        d = r - r_o
        rows.append(dict(target=tn, rep=b, kind=kind, auc=roc_auc_score(ye, p0), dauc=roc_auc_score(ye, p0) - a_o,
                         max_delta=d.max(), max_abs_delta=np.abs(d).max(), argmax=names[int(np.argmax(d))],
                         **{"d_" + n: d[j] for j, n in enumerate(names)}))
    pd.DataFrame(rows).to_csv(f"{OUT}/benign_100_b.csv", index=False)
    print(tn, "done", round(time.time() - t0), "s", flush=True)
