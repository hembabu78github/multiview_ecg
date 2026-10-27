"""Metamorphic operators over the multimodal feature pipeline.

Each operator maps a Raw batch (values + unit tags) to a transformed Raw batch plus a boolean mask of rows
where the relation applies, and states the relation it expects between p(x) and p(T(x)):
  invariance   |p' - p| <= tau
  nonincrease  p' - p <= tau          (probability must not rise by more than tau)
  nondecrease  p - p' <= tau          (probability must not fall by more than tau)
All relations are *candidate* relations derived from the PTB-XL+ feature definitions (units, lead names) and
general ECG conventions; reviewed by one independent cardiologist (verdicts in review/clinician_verdicts_deidentified.csv): rejected as invariances: loss of leads II, III, aVF, V2, V3, V4; conditional: NOISE x2, TERRITORY_CONTROL, ST elevation, loss of I, aVL, V1, V5, V6.
"""
import numpy as np
from common import LEADS, AMP_TO_CANON, TIME_TO_CANON, HEIGHT_TO_CANON, WEIGHT_TO_CANON

TERRITORY_LEADS = {"IMI": ["II", "III", "aVF"], "AMI": ["V1", "V2", "V3", "V4"]}
ST_PAIRS = [("II", "III"), ("III", "aVF"), ("V2", "V3"), ("V3", "V4"), ("V5", "V6"), ("I", "aVL")]
ST_BASES = {"ST_Amp", "ST_Amp116", "ST_Amp18", "ST_AmpMax", "ST_AmpMin", "ST_Amp80ms", "ST_Amp28", "ST_Amp38",
            "ST_AmpAdj", "ST_Elev"}
NOISE_SIGMA = dict(ms=2.0, mV=0.01, area_rel=0.02, degrees=1.0)   # one 500 Hz sample / 10 uV / 2% / 1 degree
ST_SHIFT_MV = 0.15


class Ctx:
    def __init__(self, schema, meta, seed):
        self.schema, self.meta, self.seed = schema, meta, seed
        self.cols = list(schema.index)
        self.pos = {c: i for i, c in enumerate(self.cols)}
        self.unit = schema.unit.fillna("").to_numpy()
        self.view = schema.view.to_numpy()
        self.lead = schema.lead.to_numpy()
        self.kind = schema.kind.to_numpy()
        self.base = schema.base.to_numpy()

    def rng(self, tag):
        return np.random.default_rng([self.seed, abs(hash(tag)) % (2 ** 31)] if False else [self.seed, sum(map(ord, tag))])

    def cols_where(self, mask):
        return np.where(mask)[0]


def _convert(raw, ctx, **new_tags):
    """Re-express the batch in other units (exact conversion, no rounding)."""
    r = raw.copy()
    cur = r.tags
    for key, table in (("amp", AMP_TO_CANON), ("time", TIME_TO_CANON), ("height", HEIGHT_TO_CANON), ("weight", WEIGHT_TO_CANON)):
        if key not in new_tags:
            continue
        to = new_tags[key]
        f_old, f_new = table[cur[key]], table[to]
        ratio = f_old / f_new            # raw_new = raw_old * ratio
        if key == "amp":
            groups = [(ctx.unit == "mV", ratio), (ctx.unit == "mV*ms", ratio), (ctx.unit == "mV/ms", ratio)]
        elif key == "time":
            groups = [(ctx.unit == "ms", ratio), (ctx.unit == "mV*ms", ratio), (ctx.unit == "mV/ms", 1.0 / ratio)]
        elif key == "height":
            groups = [(ctx.cols_arr == "struct::height", ratio)]
        else:
            groups = [(ctx.cols_arr == "struct::weight", ratio)]
        for m, f in groups:
            ix = np.where(m & (ctx.view != "struct" if key in ("amp", "time") else True))[0]
            if len(ix):
                r.X[:, ix] = r.X[:, ix] * f
        r.tags[key] = to
    return r


class Op:
    kind = "invariance"
    family = ""
    name = ""

    def apply(self, raw, ctx):
        raise NotImplementedError


class UnitOp(Op):
    family = "UNIT"

    def __init__(self, key, to):
        self.key, self.to = key, to
        self.name = f"UNIT_{key.upper()}_{to}"

    def apply(self, raw, ctx):
        if not hasattr(ctx, "cols_arr"):
            ctx.cols_arr = np.array(ctx.cols)
        return _convert(raw, ctx, **{self.key: self.to}), np.ones(len(raw.X), bool)


class AnthroUnitOp(Op):
    family = "UNIT"
    name = "UNIT_ANTHRO_in_lb"

    def apply(self, raw, ctx):
        if not hasattr(ctx, "cols_arr"):
            ctx.cols_arr = np.array(ctx.cols)
        r = _convert(raw, ctx, height="in")
        r = _convert(r, ctx, weight="lb")
        return r, np.ones(len(raw.X), bool)


class NoiseOp(Op):
    family = "NOISE"

    def __init__(self, scale):
        self.scale = scale
        self.name = f"NOISE_x{scale:g}"

    def apply(self, raw, ctx):
        r = raw.copy()
        rng = ctx.rng(self.name)
        val = (ctx.kind == "value") & (ctx.view != "struct")
        # reference scales are defined in canonical units (ms, mV) and converted to the stream's current unit tags
        t_ms = NOISE_SIGMA["ms"] / TIME_TO_CANON[raw.tags["time"]]
        t_mV = NOISE_SIGMA["mV"] / AMP_TO_CANON[raw.tags["amp"]]
        for unit, sig, rel in (("ms", t_ms, False), ("mV", t_mV, False),
                               ("degrees", NOISE_SIGMA["degrees"], False), ("mV*ms", NOISE_SIGMA["area_rel"], True)):
            ix = np.where(val & (ctx.unit == unit))[0]
            if not len(ix):
                continue
            z = rng.standard_normal((len(r.X), len(ix)))
            if rel:
                r.X[:, ix] = r.X[:, ix] * (1 + self.scale * sig * z)
            else:
                r.X[:, ix] = r.X[:, ix] + self.scale * sig * z
        return r, np.ones(len(r.X), bool)


class AgeOp(Op):
    family = "STRUCT"
    name = "AGE_PLUS_1Y"

    def apply(self, raw, ctx):
        r = raw.copy()
        r.X[:, ctx.pos["struct::age"]] += 1.0
        return r, np.ones(len(r.X), bool)


class MaskViewOp(Op):
    family = "MASK_VIEW"

    def __init__(self, view):
        self.view_name = view
        self.name = f"MASK_VIEW_{view}"

    def apply(self, raw, ctx):
        r = raw.copy()
        r.X[:, ctx.view == self.view_name] = np.nan
        return r, np.ones(len(r.X), bool)


class MaskLeadOp(Op):
    family = "MASK_LEAD"

    def __init__(self, lead):
        self.lead_name = lead
        self.name = f"MASK_LEAD_{lead}"

    def apply(self, raw, ctx):
        r = raw.copy()
        r.X[:, ctx.lead == self.lead_name] = np.nan
        return r, np.ones(len(r.X), bool)


class TerritoryMaskOp(Op):
    """For MI records with a single labelled territory (inferior or anterior), mask that territory's leads
    (or, for the control, the same number of other leads). Relation: probability of MI must not rise."""
    family = "TERRITORY"
    kind = "nonincrease"

    def __init__(self, control=False):
        self.control = control
        self.name = "TERRITORY_CONTROL" if control else "TERRITORY_MASK"

    def apply(self, raw, ctx):
        r = raw.copy()
        terr = ctx.meta["terr"].to_numpy()
        y = ctx.meta["y"].to_numpy()
        rng = ctx.rng(self.name)
        mask = np.zeros(len(r.X), bool)
        for i in range(len(r.X)):
            if y[i] == 1 and terr[i] in TERRITORY_LEADS:
                leads = TERRITORY_LEADS[terr[i]]
                if self.control:
                    pool = [l for l in LEADS if l not in leads and l != "aVR"]
                    leads = list(rng.choice(pool, size=len(leads), replace=False))
                r.X[i, np.isin(ctx.lead, leads)] = np.nan
                mask[i] = True
        return r, mask


class StElevationOp(Op):
    """Add 0.15 mV to ST-level features in two contiguous leads (all views). Relation: P(MI) must not fall."""
    family = "ST_ELEVATION"
    kind = "nondecrease"
    name = "ST_ELEV_+0.15mV_2LEADS"

    def apply(self, raw, ctx):
        r = raw.copy()
        rng = ctx.rng(self.name)
        pick = rng.integers(0, len(ST_PAIRS), size=len(r.X))
        isst = np.isin(ctx.base_noLead, list(ST_BASES)) & (ctx.kind == "value") & (ctx.unit == "mV") & (ctx.view != "struct")
        for k, (a, b) in enumerate(ST_PAIRS):
            rows = np.where(pick == k)[0]
            ix = np.where(isst & np.isin(ctx.lead, [a, b]))[0]
            if len(rows) and len(ix):
                r.X[np.ix_(rows, ix)] += ST_SHIFT_MV / AMP_TO_CANON[raw.tags["amp"]]
        return r, np.ones(len(r.X), bool)


def prepare_ctx(ctx):
    ctx.cols_arr = np.array(ctx.cols)
    ctx.base_noLead = np.array([b[: -len(l) - 1] if isinstance(l, str) and b.endswith("_" + l) else b
                                for b, l in zip(ctx.base, ctx.lead)])
    return ctx


def all_ops():
    ops = [UnitOp("amp", "uV"), UnitOp("time", "s"), AnthroUnitOp(),
           NoiseOp(0.5), NoiseOp(1.0), NoiseOp(2.0), AgeOp()]
    ops += [MaskViewOp(v) for v in ("12sl", "ecgdeli", "unig")]
    ops += [MaskLeadOp(l) for l in LEADS]
    ops += [TerritoryMaskOp(False), TerritoryMaskOp(True), StElevationOp()]
    return ops


def violation(kind, p0, p1, tau):
    d = p1 - p0
    if kind == "invariance":
        return np.abs(d) > tau
    if kind == "nonincrease":
        return d > tau
    return d < -tau
