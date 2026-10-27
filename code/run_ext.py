"""Run every scenario (clean + 36 fault scenarios) through the 5 frozen models and the 25 operators.
Saves per-record probabilities and monitor inputs for folds 9 (calibration) and 10 (evaluation)."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys, time, pickle, numpy as np
from common import *
from ops import Ctx, prepare_ctx, all_ops, _convert
from faults2 import FX, PipelineX
from monitor_ref import Reference
D = f"{ROOT}/data"; OUT = f"{ROOT}/results/ext"
ds = build_dataset(D); m = ds.meta; X = ds.X.to_numpy(); y = m.y.to_numpy(); f = m.fold.to_numpy()
models = pickle.load(open(f"{ROOT}/results/models/targets_f1-8.pkl", "rb"))
ref = Reference().fit(X[f <= 8], ds.schema); pickle.dump(ref, open(f"{ROOT}/results/models/reference.pkl", "wb"))
ops = all_ops(); cols = list(ds.X.columns)
scen = ["clean"] + list(FX)
FX["clean"] = dict(spec={}, tags=None, family="clean", group="clean", sev=None)
for fold in (9, 10):
    ev = f == fold; mm = m[ev]
    ctx = prepare_ctx(Ctx(ds.schema, mm, SEED)); ctx.cols_arr = np.array(cols)
    store = dict(y=y[ev], pid=mm.patient_id.to_numpy(), age=mm.age.to_numpy(), sex=mm.sex.to_numpy(), site=mm.site.to_numpy(),
                 device=mm.device.astype(str).to_numpy())
    t0 = time.time()
    for s in (scen if fold == 10 else ["clean"]):
        raw = Raw(X[ev].copy(), cols)
        if FX[s]["tags"]:
            from ops import _convert
            raw = _convert(raw, ctx, **FX[s]["tags"])
        T_ops = [op.apply(raw, ctx) for op in ops]
        if s == "clean": store["M"] = np.stack([mk for _, mk in T_ops], 1)
        for tn, T in models.items():
            pipe = PipelineX(T, None if s == "clean" else s)
            store[f"{s}|{tn}|p0"] = pipe.proba(raw).astype(np.float32)
            store[f"{s}|{tn}|P1"] = np.stack([pipe.proba(r) for r, _ in T_ops], 1).astype(np.float32)
            if tn == "L-HGB":
                z, parts = pipe.logit(raw, parts=True); store[f"{s}|parts"] = parts[0].astype(np.float32); store[f"{s}|avail"] = parts[1]
        rec = ref.per_record(pipe.canonical(raw))
        for k, v in rec.items(): store[f"{s}|mon|{k}"] = v
        print(fold, s, round(time.time() - t0), "s", flush=True)
    np.savez_compressed(f"{OUT}/scen_fold{fold}.npz", **store)
pickle.dump(dict(ops=[o.name for o in ops], kinds=[o.kind for o in ops], fam=[o.family for o in ops],
                 scen=scen, FX={k: {kk: vv for kk, vv in v.items() if kk != "spec"} for k, v in FX.items()}), open(f"{OUT}/scen_meta.pkl", "wb"))
