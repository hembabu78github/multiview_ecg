import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=f"{ROOT}/results/test"; P=f"{ROOT}/paper"
perf=pd.read_csv(f"{R}/perf.csv"); c=pd.read_csv(f"{R}/clean_mr.csv"); s=pd.read_csv(f"{R}/fault_summary_gated.csv")
order=["E-LR","E-HGB","L-LR","L-HGB","S-HGB"]
# Table 1
t=perf[perf.fault=="clean"].set_index("target").loc[order]
with open(f"{P}/t1.tex","w") as f:
    for k in order: f.write(f"{k} & {t.loc[k,'auc']:.3f} ({t.loc[k,'lo']:.3f}--{t.loc[k,'hi']:.3f}) & {t.loc[k,'acc']:.3f} \\\\\n")
# Table 2
names={"F1_AMP_UNIT_IGNORED":"F1 amplitude unit ignored","F2_TIME_UNIT_IGNORED":"F2 time unit ignored","F3_ANTHRO_UNIT_IGNORED":"F3 height/weight unit ignored",
"F4_SWAP_V1_V2":"F4 leads V1/V2 swapped","F5_LIMB_LEADS_CYCLED":"F5 limb leads I/II/III cycled","F6_MISSING_ZERO_FILLED":"F6 missing values zero-filled",
"F7_AMP_GAIN_80PCT":"F7 amplitude gain $\\times$0.8","F8_SEX_CODE_FLIPPED":"F8 sex code flipped","F9_ECGDELI_JOIN_LOST":"F9 ECGDeli view lost"}
with open(f"{P}/t2.tex","w") as f:
    for k,n in names.items():
        d=s[s.fault==k]
        f.write(f"{n} & {d.dAUC.mean():+.4f} & {d.dAUC.min():+.4f} & {int(d.acc_detect.sum())}/5 & {int(d.mr_detect.sum())}/5 \\\\\n")
# Table 3 selected ops
c5=c[c.tau==0.05]
sel=[("UNIT_AMP_uV","unit change (mV$\\to\\mu$V)"),("UNIT_TIME_s","unit change (ms$\\to$s)"),("AGE_PLUS_1Y","age $+1$ y"),("NOISE_x1","feature noise $\\times$1"),
("NOISE_x2","feature noise $\\times$2"),("MASK_LEAD_II","drop lead II"),("MASK_LEAD_V3","drop lead V3"),("MASK_VIEW_12sl","drop 12SL view"),("MASK_VIEW_unig","drop Uni-G view"),
("TERRITORY_MASK","mask infarct-territory leads"),("TERRITORY_CONTROL","mask unrelated leads (control)"),("ST_ELEV_+0.15mV_2LEADS","ST $+0.15$ mV, 2 leads")]
with open(f"{P}/t3.tex","w") as f:
    for op,lab in sel:
        cells=[]
        for tg in order:
            r=c5[(c5.target==tg)&(c5.op==op)].iloc[0]
            cells.append(f"{100*r.rate:.1f}")
        f.write(f"{lab} & "+" & ".join(cells)+" \\\\\n")
# Figure: heatmap
piv=c5.pivot(index="op",columns="target",values="rate")[order]
fam=lambda o:(0 if o.startswith("UNIT") else 1 if o.startswith("AGE") else 2 if o.startswith("NOISE") else 3 if o.startswith("MASK_VIEW") else 4 if o.startswith("MASK_LEAD") else 5)
ops=sorted(piv.index,key=lambda o:(fam(o),o)); piv=piv.loc[ops]
fig,ax=plt.subplots(figsize=(3.4,4.6)); im=ax.imshow(100*piv.values,cmap="viridis",aspect="auto",vmin=0,vmax=40)
ax.set_xticks(range(5)); ax.set_xticklabels(order,rotation=45,ha="right",fontsize=6); ax.set_yticks(range(len(ops))); ax.set_yticklabels([o.replace("ST_ELEV_+0.15mV_2LEADS","ST_ELEV") for o in ops],fontsize=5)
for i in range(len(ops)):
    for j in range(5):
        v=100*piv.values[i,j]; ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=4.5,color="w" if v<25 else "k")
cb=plt.colorbar(im,ax=ax,fraction=.05,pad=.02); cb.ax.tick_params(labelsize=6); cb.set_label("violation rate, % (clipped at 40)",fontsize=6)
plt.tight_layout(); plt.savefig(f"{P}/fig_heat.pdf"); plt.savefig(f"{P}/fig_heat.png",dpi=200)
# key numbers
b=pd.read_csv(f"{R}/benign_gated.csv"); bo=pd.read_csv(f"{R}/benign_ops.csv")
print("benign MR false alarm",b.flag.sum(),len(b),"max|dAUC|",b.dAUC.abs().max().round(4))
sil=s[~s.acc_detect]; print("accuracy-silent pairs",len(sil),"MR-detected",int(sil.mr_detect.sum()))
print(s.groupby("fault")[["acc_detect","mr_detect"]].sum().T.to_string())
# tau sensitivity of the headline rates
for tau in (0.02,0.05,0.10):
    x=c[(c.tau==tau)&(c.op=="MASK_VIEW_unig")].set_index("target").loc[order]; print("tau",tau,"unig mask",(100*x.rate).round(1).to_dict())
x=c[(c.tau==0.05)&(c.op.isin(["MASK_VIEW_unig","TERRITORY_MASK","TERRITORY_CONTROL","NOISE_x2"]))]
print(x[["target","op","n_applicable","rate","lo","hi"]].round(3).to_string())
