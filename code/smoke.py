import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import time, numpy as np
from sklearn.metrics import roc_auc_score
from common import *
D=f"{ROOT}/data"
t=time.time(); ds=build_dataset(D); print("built",ds.X.shape,round(time.time()-t,1),"s")
m=ds.meta; print(m.groupby(["fold","y"]).size().unstack().T)
print(ds.schema.groupby(["view","kind"]).size().unstack())
print("units",ds.schema.unit.value_counts(dropna=False).to_dict())
X=ds.X.to_numpy(); y=m.y.to_numpy(); f=m.fold.to_numpy()
tr,va=f<=8,f==9
for n in TARGETS:
    t=time.time(); T=Target(n,ds.schema).fit(X[tr],y[tr]); p=T.proba(X[va])
    print(n,"val AUROC %.4f"%roc_auc_score(y[va],p),"fit %.1fs"%(time.time()-t))
