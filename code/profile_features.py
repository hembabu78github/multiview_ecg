import sys, pandas as pd, numpy as np
D = sys.argv[1]
db = pd.read_csv(f"{D}/ptbxl_database.csv", index_col="ecg_id")
print("metadata", db.shape)
views = {}
for name in ["12sl", "ecgdeli", "unig"]:
    f = pd.read_csv(f"{D}/{name}_features.csv", low_memory=False)
    print(f"\n== {name}: shape {f.shape}")
    print("first cols:", list(f.columns[:6]))
    idcol = "ecg_id" if "ecg_id" in f.columns else f.columns[0]
    print("id column:", idcol, "unique ids", f[idcol].nunique(), "dupes", int(f[idcol].duplicated().sum()))
    f = f.set_index(idcol)
    views[name] = f
    num = f.select_dtypes("number")
    print("numeric cols", num.shape[1], "non-numeric cols", f.shape[1] - num.shape[1])
    miss = num.isna().mean()
    print("missing: median col %.3f, cols>50%% missing: %d, cols 100%% missing: %d" % (miss.median(), (miss > .5).sum(), (miss == 1).sum()))
    print("rows with any non-null feature:", int(num.notna().any(axis=1).sum()))
    print("example col names:", list(num.columns[:10]))
ids = [set(v.index) for v in views.values()]
print("\nids in all three views:", len(set.intersection(*ids)))
for a in views:
    for b in views:
        if a < b:
            print(a, b, "overlap", len(set(views[a].index) & set(views[b].index)))
print("ids in metadata:", len(set(db.index)), " covered by 12sl:", len(set(db.index) & set(views['12sl'].index)),
      " ecgdeli:", len(set(db.index) & set(views['ecgdeli'].index)), " unig:", len(set(db.index) & set(views['unig'].index)))
