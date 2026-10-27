"""A10: CPU cost of inference vs monitoring (process CPU time, single thread). Batch = 200 records, median of 5 repeats."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import time, pickle, numpy as np, pandas as pd
from common import *
from ops import Ctx, prepare_ctx, all_ops
from monitor_ref import Reference
D = f"{ROOT}/data"; OUT = f"{ROOT}/results/ext"
ds = build_dataset(D); m = ds.meta; X = ds.X.to_numpy(); f = m.fold.to_numpy(); cols = list(ds.X.columns)
models = pickle.load(open(f"{ROOT}/results/models/targets_f1-8.pkl", "rb"))
ref = pickle.load(open(f"{ROOT}/results/models/reference.pkl", "rb"))
ev = np.where(f == 10)[0][:200]; ctx = prepare_ctx(Ctx(ds.schema, m.iloc[ev], SEED)); ops = all_ops(); raw = Raw(X[ev].copy(), cols)
def cpu(fn, rep=5):
    ts = []
    for _ in range(rep):
        t = time.process_time(); fn(); ts.append(time.process_time() - t)
    return float(np.median(ts))
rows = []
t_ops = cpu(lambda: [o.apply(raw, ctx) for o in ops])
T_ops = [o.apply(raw, ctx) for o in ops]
t_schema = cpu(lambda: ref.per_record(X[ev])); B = ref.bins(X[ev])
t_psi = cpu(lambda: ref.psi(B))
for tn, T in models.items():
    pipe = Pipeline(T, None)
    t_inf = cpu(lambda: pipe.proba(raw))
    t_mr = cpu(lambda: [pipe.proba(r) for r, _ in T_ops], 3)
    rows.append(dict(target=tn, n=200, inference_s=t_inf, mr_model_queries_s=t_mr, mr_operator_s=t_ops, mr_total_s=t_mr + t_ops,
                     validators_s=t_schema + t_psi, mr_over_inference=(t_mr + t_ops) / t_inf, per_record_ms_inference=1e3 * t_inf / 200,
                     per_record_ms_mr=1e3 * (t_mr + t_ops) / 200))
pd.DataFrame(rows).to_csv(f"{OUT}/cost_timing.csv", index=False); print(pd.DataFrame(rows).round(4).to_string())
