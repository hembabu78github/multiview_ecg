import sys, re, ast, pandas as pd, numpy as np, collections
D = sys.argv[1]
db = pd.read_csv(f"{D}/ptbxl_database.csv", index_col="ecg_id")
st = pd.read_csv(f"{D}/scp_statements.csv", index_col=0)
fd = pd.read_csv(f"{D}/feature_description.csv").dropna(subset=["id"])
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
units = dict(zip(fd["id"], fd["unit"]))
views = {n: pd.read_csv(f"{D}/{n}_features.csv", low_memory=False).set_index("ecg_id") for n in ["12sl","ecgdeli","unig"]}
for n, v in views.items():
    cols = list(v.columns)
    suffix = collections.Counter()
    unit_cov = collections.Counter(); unknown = []
    for c in cols:
        base = re.sub(r"_(iqr|count)$", "", c)
        m = re.match(r"^(.*)_(%s)$" % "|".join(map(re.escape, LEADS)), base)
        fid = (m.group(1) + "_X") if m else base
        u = units.get(fid)
        if u is None: unknown.append(c)
        unit_cov[u] += 1
        suffix["lead" if m else "global"] += 1
    print(n, dict(suffix), "unit coverage", dict(unit_cov), "unknown e.g.", unknown[:8])
# labels
db["scp"] = db.scp_codes.apply(ast.literal_eval)
vals = [v for d in db.scp for v in d.values()]
print("likelihood value dist:", pd.Series(vals).describe().round(1).to_dict(), "zeros:", sum(v == 0 for v in vals))
diag = st[st.diagnostic == 1]
def sup(d, minl):
    return frozenset(diag.loc[k, "diagnostic_class"] for k, v in d.items() if k in diag.index and v >= minl)
for minl in (0, 1, 50):
    s = db.scp.apply(lambda d: sup(d, minl))
    print("minL", minl, "NORM-only", int((s == frozenset({"NORM"})).sum()), "MI-only", int((s == frozenset({"MI"})).sum()))
# fold/patient leakage
g = db.groupby("patient_id").strat_fold.nunique()
print("patients spanning >1 fold:", int((g > 1).sum()), "of", len(g))
print("records of those patients:", int(db.patient_id.isin(g[g > 1].index).sum()))
# MI subclass distribution among MI-only
s0 = db.scp.apply(lambda d: sup(d, 0))
mi = db[s0 == frozenset({"MI"})]
sub = collections.Counter()
for d in mi.scp:
    sub[tuple(sorted({diag.loc[k, "diagnostic_subclass"] for k in d if k in diag.index and diag.loc[k, "diagnostic_class"] == "MI"}))] += 1
print("MI-only subclass combos:", sub.most_common(8))
print("fold of MI-only/NORM-only:", pd.crosstab(db.strat_fold[(s0 == frozenset({"MI"})) | (s0 == frozenset({"NORM"}))], s0[(s0 == frozenset({"MI"})) | (s0 == frozenset({"NORM"}))].apply(lambda x: list(x)[0])).T.to_dict())
