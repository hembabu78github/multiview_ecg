"""Apply the frozen decision rule and build the manuscript tables.  python analyze.py test
Frozen rule (set on the validation fold, applied unchanged to the test fold):
  MR detection of a fault  = exists operator with  (violation-rate increase >= GATE at tau=0.05)
                             AND (one-sided Bonferroni-adjusted bootstrap lower bound > 0)
  Accuracy-only detection  = 95% cluster-bootstrap CI of the change in AUROC lies entirely below -0.01
"""
import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys
import numpy as np, pandas as pd

GATE = 0.10
split = sys.argv[1] if len(sys.argv) > 1 else "test"
R = f"{ROOT}/results/{split}"
f = pd.read_csv(f"{R}/fault_ops.csv")
s = pd.read_csv(f"{R}/fault_summary.csv")
b = pd.read_csv(f"{R}/benign.csv")
bo = pd.read_csv(f"{R}/benign_ops.csv")
c = pd.read_csv(f"{R}/clean_mr.csv")
perf = pd.read_csv(f"{R}/perf.csv")

f5 = f[f.tau == 0.05].copy()
f5["flag"] = (f5.lo_bonf > 0) & (f5.delta >= GATE)
g = f5.groupby(["target", "fault"]).agg(n_flag=("flag", "sum"),
                                         ops=("op", lambda x: ";".join(x[f5.loc[x.index, "flag"]]))).reset_index()
s = s.drop(columns=["mr_detect", "n_ops_flagged", "ops_flagged"]).merge(g, on=["target", "fault"])
s["mr_detect"] = s.n_flag > 0
s.to_csv(f"{R}/fault_summary_gated.csv", index=False)

bo["flag"] = (bo.lo_bonf > 0) & (bo.delta >= GATE)
bg = bo.groupby(["target", "replicate"]).flag.any().reset_index().merge(b[["target", "replicate", "dAUC", "acc_detect"]],
                                                                        on=["target", "replicate"])
bg.to_csv(f"{R}/benign_gated.csv", index=False)

print(f"=== split={split}  GATE={GATE}")
print("\nAUROC (clean):")
print(perf[perf.fault == "clean"].round(4).to_string(index=False))
print("\nDetection by fault (5 targets):")
d = s.groupby("fault").agg(acc_detect=("acc_detect", "sum"), mr_detect=("mr_detect", "sum"), mean_dAUC=("dAUC", "mean"),
                           worst_dAUC=("dAUC", "min")).round(4)
print(d.to_string())
print("\nSilent faults (|dAUC CI| not below -0.01) detected by MR suite:",
      int(((~s.acc_detect) & s.mr_detect).sum()), "of", int((~s.acc_detect).sum()), "target-fault pairs")
print("Faults detected by accuracy but not MR:", int((s.acc_detect & ~s.mr_detect).sum()))
print("\nBenign retrains: MR false alarms %d / %d ; accuracy false alarms %d / %d ; max |dAUC| %.4f" % (
    bg.flag.sum(), len(bg), bg.acc_detect.sum(), len(bg), bg.dAUC.abs().max()))
print(bo[bo.flag][["target", "replicate", "op", "delta"]].round(3).to_string(index=False))
