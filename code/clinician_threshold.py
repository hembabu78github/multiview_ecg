"""Clinician-specified operating points (review form): threshold 0.10 and 0.15, missed-MI : false-alarm cost >= 20:1.
Outputs results/ext/clinician_threshold.csv: sens, spec, cost per 1000 records (20*FN+FP), change vs clean with cluster-bootstrap CI."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pickle, numpy as np, pandas as pd
from run_experiments import boot_weights
OUT=f"{ROOT}/results/ext"
Z=np.load(f"{OUT}/scen_fold10.npz",allow_pickle=True); meta=pickle.load(open(f"{OUT}/scen_meta.pkl","rb"))
y=Z["y"].astype(float); pid=Z["pid"]; B=500
W=boot_weights(pid,B,np.random.default_rng(2026))
def st(p,t,w):
    n=w.sum(); fn=(w*((p<t)&(y==1))).sum(); fp=(w*((p>=t)&(y==0))).sum(); P=(w*(y==1)).sum(); N=(w*(y==0)).sum()
    return dict(sens=1-fn/P, spec=1-fp/N, cost=(20*fn+fp)/n*1000, fn_per1000=fn/n*1000)
rows=[]
one=np.ones_like(y)
for t in ["E-LR","E-HGB","L-LR","L-HGB","S-HGB"]:
    pc=Z[f"clean|{t}|p0"].astype(float)
    for thr in (0.10,0.15):
        bc=[st(pc,thr,W[b]) for b in range(B)]
        for s in meta["scen"]:
            pf=Z[f"{s}|{t}|p0"].astype(float); r=dict(target=t,thr=thr,scenario=s,**st(pf,thr,one))
            if s!="clean":
                bf=[st(pf,thr,W[b]) for b in range(B)]
                for k in ("sens","cost"):
                    d=np.array([a[k]-c[k] for a,c in zip(bf,bc)]); r[f"d_{k}"]=r[k]-st(pc,thr,one)[k]; r[f"d_{k}_lo"],r[f"d_{k}_hi"]=np.percentile(d,[2.5,97.5])
            rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/clinician_threshold.csv",index=False)
