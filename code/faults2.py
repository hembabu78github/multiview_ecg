"""Extended fault engine (A3): severities, joint faults, and held-out fault families.
HELD-OUT families (stale, truncation, misalignment, column shift, value corruption) were specified AFTER the 25-operator suite
and the 10-point gate were frozen; the operator suite is not changed for them."""
import re
import numpy as np
from common import *

ALT_UNITS = dict(amp="uV", time="s", height="in", weight="lb")


def S(**k): return k

# name -> (spec, family, severity, group, stream_tags)    group: dev | heldout | joint | active
FX = {}
def add(name, spec, family, sev=None, group="dev", tags=None):
    FX[name] = dict(spec=spec, family=family, sev=sev, group=group, tags=tags)

for k, v in FAULTS.items():                     # the nine original faults (canonical-unit stream; F1-F3 are dormant)
    add(k, dict(v), k.split("_")[0], None, "dev")
for g in (0.5, 1.2, 1.5): add(f"GAIN_{g}", S(gain=g), "gain", g, "dev")
for p in (5, 20, 50): add(f"MISS_{p}", S(miss=p / 100), "missing", p, "dev")
for p in (25, 50): add(f"PARTIAL_{p}", S(partial=p / 100), "partial_view", p, "dev")
add("JOINT_F4_F7", S(perm=("V1", "V2"), gain=0.8), "joint", 2, "joint")
add("JOINT_F9_F6", S(dropview="ecgdeli", zerofill=True), "joint", 2, "joint")
add("JOINT_F5_F7_F6", S(cycle=("I", "II", "III"), gain=0.8, zerofill=True), "joint", 3, "joint")
add("STALE_12sl_30", S(stale=("12sl", .30)), "stale", 30, "heldout")
add("TRUNC_30", S(trunc=.30), "truncated", 30, "heldout")
add("MISALIGN_ecgdeli", S(misalign="ecgdeli"), "misaligned", None, "heldout")
add("MISALIGN_unig", S(misalign="unig"), "misaligned", None, "heldout")
add("SHIFTCOL_12sl", S(shiftcol="12sl"), "col_shift", None, "heldout")
add("SHIFTCOL_ecgdeli", S(shiftcol="ecgdeli"), "col_shift", None, "heldout")
for p in (5, 20, 50): add(f"CORRUPT_{p}", S(corrupt=p / 100), "corrupt", p, "heldout")
# active unit faults: the incoming stream itself uses non-canonical units and the pipeline ignores the tag
add("ACT_F1_AMP", dict(ignore=("amp",)), "active_unit", None, "active", dict(amp="uV"))
add("ACT_F2_TIME", dict(ignore=("time",)), "active_unit", None, "active", dict(time="s"))
add("ACT_F3_ANTHRO", dict(ignore=("height", "weight")), "active_unit", None, "active", dict(height="in", weight="lb"))
# correct pipeline on alt-unit streams: must be benign
add("ALTUNITS_OK_AMP", {}, "alt_units_ok", None, "ok", dict(amp="uV"))
add("ALTUNITS_OK_TIME", {}, "alt_units_ok", None, "ok", dict(time="s"))
add("ALTUNITS_OK_ANTHRO", {}, "alt_units_ok", None, "ok", dict(height="in", weight="lb"))


class PipelineX(Pipeline):
    def __init__(self, target, name):
        self.t, self.fault, self.name = target, name, name
        sc = target.schema
        spec = FX[name]["spec"] if name else {}
        self.spec = spec
        self.h = Harmonizer(sc, ignore=spec.get("ignore", ()))
        self.cols = list(sc.index); self.col_pos = {c: i for i, c in enumerate(self.cols)}
        self.perm = self._perm(sc, spec)
        self.gain_ix = np.where(sc.unit.isin(["mV", "mV*ms"]).to_numpy() & (sc.view != "struct").to_numpy())[0]
        self.drop_ix = np.where((sc.view == spec["dropview"]).to_numpy())[0] if "dropview" in spec else None
        self.view_ix_all = {v: np.where((sc.view == v).to_numpy())[0] for v in VIEWS}
        self.nonstruct = np.where((sc.view != "struct").to_numpy())[0]

    def canonical(self, raw):
        return self._base(raw)

    def _base(self, raw):
        sp = self.spec
        X = self.h(raw)
        n = len(X)
        rng = np.random.default_rng([SEED, sum(map(ord, self.name or "")), n])
        if "perm" in sp or "cycle" in sp: X = X[:, self.perm]
        if "gain" in sp: X[:, self.gain_ix] *= sp["gain"]
        if sp.get("sexflip"): X[:, self.col_pos["struct::sex"]] = 1.0 - X[:, self.col_pos["struct::sex"]]
        if "shiftcol" in sp:
            ix = self.view_ix_all[sp["shiftcol"]]; X[:, ix] = X[:, np.roll(ix, 1)]
        if "misalign" in sp:
            ix = self.view_ix_all[sp["misalign"]]; X[:, ix] = np.roll(X[:, ix], 1, axis=0)
        if "stale" in sp:
            v, frac = sp["stale"]; ix = self.view_ix_all[v]
            rows = np.where(rng.random(n) < frac)[0]; rows = rows[rows > 0]
            X[np.ix_(rows, ix)] = X[np.ix_(rows - 1, ix)]
        if "trunc" in sp:
            for v in VIEWS:
                ix = self.view_ix_all[v]; k = int(np.ceil((1 - sp["trunc"]) * len(ix))); X[:, ix[k:]] = np.nan
        if "corrupt" in sp:
            ns = self.nonstruct; msk = rng.random((n, len(ns))) < sp["corrupt"]
            src = rng.integers(0, n, size=(n, len(ns)))
            Xn = X[:, ns]; Xn = np.where(msk, np.take_along_axis(Xn, src, axis=0), Xn); X[:, ns] = Xn
        if "miss" in sp:
            ns = self.nonstruct; msk = rng.random((n, len(ns))) < sp["miss"]
            Xn = X[:, ns]; Xn[msk] = np.nan; X[:, ns] = Xn
        if "partial" in sp:
            rows = rng.random(n) < sp["partial"]; X[np.ix_(rows, self.view_ix_all["ecgdeli"])] = np.nan
        if "dropview" in sp: X[:, self.drop_ix] = np.nan
        if sp.get("zerofill"): X = np.where(np.isnan(X), 0.0, X)
        return X
