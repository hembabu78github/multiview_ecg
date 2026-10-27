"""Fit the five frozen targets on folds 1-8 and pickle them (deterministic; identical to run_experiments.py)."""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pickle, sys
from common import *
D = f"{ROOT}/data"
ds = build_dataset(D); m = ds.meta
X = ds.X.to_numpy(); y = m.y.to_numpy(); f = m.fold.to_numpy()
out = {}
for tn in TARGETS:
    out[tn] = Target(tn, ds.schema).fit(X[f <= 8], y[f <= 8]); print(tn, flush=True)
pickle.dump(out, open(f"{ROOT}/results/models/targets_f1-8.pkl", "wb"))
