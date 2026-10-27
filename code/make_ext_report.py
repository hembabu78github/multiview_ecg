import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import beta
E = f"{ROOT}/results/ext"; P = f"{ROOT}/paper"
d = pd.read_csv(f"{E}/monitor_det.csv"); fa = pd.read_csv(f"{E}/monitor_fa.csv")
MON = ["ACC", "PRED", "SCHEMA", "MISS", "PSI", "HOT", "MR", "STACK", "MR+VAL", "ALL"]
NAMES = {"ACC": "Accuracy (oracle labels)", "PRED": "Prediction shift", "SCHEMA": "Range validator", "MISS": "Missingness", "PSI": "Drift (PSI)", "HOT": "Drift (PCA T²)",
         "MR": "Metamorphic (MR)", "STACK": "Conventional stack", "MR+VAL": "MR + validator + missingness", "ALL": "All monitors"}
# detection by group
d2 = d[d.n == 200]
byg = d2.groupby(["group", "monitor"]).det.mean().unstack()[MON]
byg.to_csv(f"{E}/tab_det_by_group.csv")
byf = d2.groupby(["family", "monitor"]).det.mean().unstack()[MON]; byf.to_csv(f"{E}/tab_det_by_family.csv")
# false alarm
f2 = fa[fa.n == 200]
fa_tab = f2.groupby(["scenario", "pop", "monitor"]).rate.mean().unstack()[MON]; fa_tab.to_csv(f"{E}/tab_fa.csv")
# alarm precision at fault prevalence
fpr = f2[(f2.scenario == "clean") & (f2["pop"] == "all")].groupby("monitor").rate.mean()
rows = []
for g in ["dev", "heldout", "joint", "active"]:
    tpr = byg.loc[g]
    for pi in (0.05, 0.2):
        rows.append(dict(group=g, prevalence=pi, **{m: tpr[m] * pi / (tpr[m] * pi + fpr[m] * (1 - pi) + 1e-12) for m in MON}))
pd.DataFrame(rows).to_csv(f"{E}/tab_precision.csv", index=False)
# detection vs batch size (delay)
dn = d.groupby(["n", "monitor"]).det.mean().unstack()[MON]; dn.to_csv(f"{E}/tab_det_by_n.csv")
# severity curves
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
fams = [("gain", ["GAIN_0.5", "F7_AMP_GAIN_80PCT", "GAIN_1.2", "GAIN_1.5"], [0.5, 0.8, 1.2, 1.5], "amplitude gain"),
        ("missing", ["MISS_5", "MISS_20", "MISS_50"], [5, 20, 50], "random missing (%)"),
        ("corrupt", ["CORRUPT_5", "CORRUPT_20", "CORRUPT_50"], [5, 20, 50], "value corruption (%)"),
        ("partial", ["PARTIAL_25", "PARTIAL_50", "F9_ECGDELI_JOIN_LOST"], [25, 50, 100], "ECGDeli rows lost (%)")]
show = ["ACC", "PRED", "SCHEMA", "PSI", "MR"]; sty = dict(ACC="k:", PRED="C1-", SCHEMA="C2-", PSI="C4-", MR="C3-")
for ax, (fam, sc, xs, lab) in zip(axes, fams):
    for m in show:
        y = [d2[(d2.scenario == s) & (d2.monitor == m)].det.mean() for s in sc]
        ax.plot(xs, y, sty[m], marker="o", ms=2.5, lw=1.2, label=NAMES[m])
    ax.set_xlabel(lab, fontsize=6.5); ax.tick_params(labelsize=6); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=.3)
axes[0].set_ylabel("detection rate", fontsize=6.5); axes[0].legend(fontsize=4.8, loc="center left", frameon=False)
plt.tight_layout(); plt.savefig(f"{P}/fig_severity.pdf"); plt.savefig(f"{P}/fig_severity.png", dpi=200)
# heatmap monitor x fault scenario
x = d2.groupby(["scenario", "monitor"]).det.mean().unstack()[["ACC", "PRED", "SCHEMA", "MISS", "PSI", "HOT", "MR"]]
grp = d2.drop_duplicates("scenario").set_index("scenario").group
order = [s for g in ["dev", "joint", "heldout", "active"] for s in x.index if grp[s] == g]
x = x.loc[order]
fig, ax = plt.subplots(figsize=(3.5, 6.2)); im = ax.imshow(x.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(x.shape[1])); ax.set_xticklabels(x.columns, rotation=45, ha="right", fontsize=6); ax.set_yticks(range(len(x))); ax.set_yticklabels(x.index, fontsize=4.8)
for i in range(x.shape[0]):
    for j in range(x.shape[1]): ax.text(j, i, f"{x.values[i,j]*100:.0f}", ha="center", va="center", fontsize=4, color="w" if x.values[i, j] < .6 else "k")
plt.colorbar(im, ax=ax, fraction=.05, pad=.02).ax.tick_params(labelsize=6); plt.tight_layout(); plt.savefig(f"{P}/fig_monitors.pdf"); plt.savefig(f"{P}/fig_monitors.png", dpi=200)
# benign retrains (model update) -- alarm if max_delta >= 0.10 (full-fold point rule)
b = pd.read_csv(f"{E}/benign_100.csv")
def wilson(k, n, a=.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi
rows = []
for (t, kd), g in list(b.groupby(["target", "kind"])) + [((t, "all"), g) for t, g in b.groupby("target")]:
    k = int((g.max_delta >= 0.10).sum()); lo, hi = wilson(k, len(g))
    rows.append(dict(target=t, kind=kd, n=len(g), alarms=k, rate=k / len(g), lo=lo, hi=hi, median_max_delta=g.max_delta.median(), p95_max_delta=g.max_delta.quantile(.95), max_abs_dauc=g.dauc.abs().max()))
pd.DataFrame(rows).to_csv(f"{E}/tab_model_update_fa.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
print(byg.round(2).to_string()); print(fa_tab.round(3).to_string()); print(dn.round(2).to_string())
