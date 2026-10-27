"""A7b: is fault-aware abstention better than plain confidence-based selective prediction at matched coverage?"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from fusion_qa import *
rows = []
for tn in ("L-HGB", "L-LR"):
    T = models[tn]; Fu = Fuser(T)
    for s in ["clean"] + [k for k in FX if not k.startswith("ALTUNITS")]:
        raw, yy = stream(10, s); P, A, Q, _ = parts(T, s, raw); V = Fu.variants(P, A, Q)
        zq = V["QL+abst"]; cov = np.isfinite(zq); c = cov.mean()
        if c == 0:
            rows.append(dict(target=tn, scenario=s, coverage=0.0, note="refused all")); continue
        zb = V["Lcal"]; ok = np.isfinite(zb)
        k = int(round(c * len(yy))); keep_conf = np.zeros(len(yy), bool); keep_conf[np.argsort(-np.abs(np.where(ok, zb, 0)))[:k]] = True
        rng = np.random.default_rng(1); keep_rand = np.zeros(len(yy), bool); keep_rand[rng.choice(len(yy), k, replace=False)] = True
        def ev(z, keep):
            p = 1 / (1 + np.exp(-z[keep])); yc = yy[keep]
            return dict(auc=roc_auc_score(yc, p), err=((p > .5) != yc).mean(), miss=((p <= .5) & (yc == 1)).sum() / max(yc.sum(), 1))
        a = ev(np.where(cov, zq, 0), cov); b = ev(zb, keep_conf); r = ev(zb, keep_rand); f0 = ev(zb, np.ones(len(yy), bool))
        rows.append(dict(target=tn, scenario=s, coverage=c, auc_abst=a["auc"], auc_conf=b["auc"], auc_rand=r["auc"], auc_full=f0["auc"],
                         err_abst=a["err"], err_conf=b["err"], err_full=f0["err"], miss_abst=a["miss"], miss_conf=b["miss"], miss_full=f0["miss"]))
pd.DataFrame(rows).to_csv(f"{OUT}/abstention_eval.csv", index=False); print("done")
