"""Sensitivity analysis: compare monitors at MATCHED *realised* clean false-alarm rates.
Thresholds are set on clean fold-10 batches (draw A), then false alarms are measured on an independent clean draw B from the same pool
and detection on all fault scenarios. This is an oracle-matched comparison (calibration data and test records share a fold);
the primary analysis (monitors.py) calibrates on fold 9 only. n=200 records per batch."""
import numpy as np, pandas as pd
import monitors as M
from monitors import *
n, Kc, Ke, Kd = 200, 1000, 500, 300
rng = np.random.default_rng(20261026)
yT = Z10["y"]; N = len(yT); allix = np.arange(N)
clean10 = Src(Z10, M10, "clean")
# split fold-10 records by patient into calibration half (A) and evaluation half (B) for the clean draws, to keep A and B independent
pid = Z10["pid"]; up = np.unique(pid); A_p = set(rng.choice(up, len(up)//2, replace=False))
isA = np.array([p in A_p for p in pid]); A_ix = B_ix = allix   # same fold-10 pool, independent batch draws (halves gave pool-specific shifts: batches resampled from ~580 records share their mean)
UN = {"STACK": ["SCHEMA", "MISS", "PSI", "HOT", "PRED"], "MR+VAL": ["MR", "SCHEMA", "MISS"], "ALL": ["MR", "SCHEMA", "MISS", "PSI", "HOT", "PRED"]}
rows_det, rows_fa, rows_gap = [], [], []
for t in TARGETS:
    # reference from the clean fold-9 pipeline (as in the primary analysis); calibration on clean fold-10 half A
    cal = batch_stats(clean10, yT, t, draw(rng, A_ix, Kc, n), "full")
    thr = {k: np.quantile(v, 1 - ALPHA) for k, v in cal.items()}
    thrU = {u: {k: np.quantile(cal[k], 1 - ALPHA/len(m)) for k in m} for u, m in UN.items()}
    def flags(st):
        f = {k: st[k] > thr[k] for k in st}
        for u in UN: f[u] = np.any([st[k] > thrU[u][k] for k in UN[u]], axis=0)
        return f
    fa = flags(batch_stats(clean10, yT, t, draw(rng, B_ix, Ke, n), "full"))
    for k, v in fa.items(): rows_fa.append(dict(target=t, monitor=k, rate=float(v.mean())))
    for s in M.meta["scen"]:
        if s == "clean" or s.startswith("ALTUNITS"): continue
        src = Src(Z10, M10, s)
        for k, v in flags(batch_stats(src, yT, t, draw(rng, B_ix, Kd, n), "full")).items():
            rows_det.append(dict(scenario=s, family=M.meta["FX"][s]["family"], group=M.meta["FX"][s]["group"], target=t, monitor=k, det=float(v.mean())))
    print(t, "done", flush=True)
OUT2 = M.OUT
pd.DataFrame(rows_det).to_csv(f"{OUT2}/monitor_det_matched.csv", index=False); pd.DataFrame(rows_fa).to_csv(f"{OUT2}/monitor_fa_matched.csv", index=False)
