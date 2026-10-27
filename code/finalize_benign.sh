#!/bin/bash
# waits for all five targets of the model-update study, merges, regenerates tables, runs cost timing
E=/home/claude/jbhi_ptbxl/results/ext
cd "$(dirname "$0")"
while true; do
  n=$(python3 - <<'PY'
import pandas as pd,os
E="/home/claude/jbhi_ptbxl/results/ext"; s=set()
for f in ("benign_100.csv","benign_100_b.csv"):
    if os.path.exists(f"{E}/{f}"): s|=set(pd.read_csv(f"{E}/{f}").target)
print(len(s))
PY
)
  [ "$n" -ge 5 ] && break
  # stop the primary job once it has finished the targets the second job does not cover (L-HGB / S-HGB duplicates)
  sleep 60
done
pkill -f "run_benign.py 100" ; sleep 1
cp $E/benign_100.csv $E/benign_100_a.csv
python3 - <<'PY'
import pandas as pd
E="/home/claude/jbhi_ptbxl/results/ext"
a=pd.read_csv(f"{E}/benign_100_a.csv"); b=pd.read_csv(f"{E}/benign_100_b.csv")
keep=["E-LR","E-HGB","L-LR"]; d=pd.concat([a[a.target.isin(keep)], b]); d.to_csv(f"{E}/benign_100.csv",index=False); print(d.groupby("target").size())
PY
OMP_NUM_THREADS=1 python3 make_tables2.py && OMP_NUM_THREADS=1 python3 make_ext_report.py && OMP_NUM_THREADS=1 python3 cost_timing.py
echo FINALIZED
