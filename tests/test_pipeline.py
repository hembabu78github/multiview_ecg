import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys, numpy as np, pytest
sys.path.insert(0, f"{ROOT}/code")
from common import *
from ops import *
from run_experiments import boot_weights, rate_ci

D = f"{ROOT}/data"


@pytest.fixture(scope="module")
def ds():
    return build_dataset(D)


@pytest.fixture(scope="module")
def ctx(ds):
    return prepare_ctx(Ctx(ds.schema, ds.meta, SEED))


def test_no_patient_spans_folds(ds):
    g = ds.meta.groupby("patient_id").fold.nunique()
    assert (g == 1).all()


def test_no_label_or_text_columns_in_features(ds):
    for c in ds.X.columns:
        assert c.split("::")[0] in ("12sl", "ecgdeli", "unig", "struct")
        assert "report" not in c and "scp" not in c and c != "y"


def test_cohort_is_pure(ds):
    assert set(ds.meta.y.unique()) == {0, 1}
    assert (ds.meta.loc[ds.meta.y == 0, "terr"] == "").all()


def test_every_value_column_has_a_unit(ds):
    s = ds.schema
    assert s.unit.isna().sum() == 0


def test_unit_ops_are_exact_inverse_under_correct_harmonizer(ds, ctx):
    raw = to_raw(ds, ds.X.index[:300])
    h = Harmonizer(ds.schema)
    base = h(raw)
    for op in [UnitOp("amp", "uV"), UnitOp("time", "s"), AnthroUnitOp()]:
        r2, mask = op.apply(raw, ctx)
        assert mask.all()
        np.testing.assert_allclose(h(r2), base, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("fault,op", [("F1_AMP_UNIT_IGNORED", UnitOp("amp", "uV")), ("F2_TIME_UNIT_IGNORED", UnitOp("time", "s"))])
def test_unit_faults_change_canonical_values_only_when_units_change(ds, ctx, fault, op):
    raw = to_raw(ds, ds.X.index[:300])
    clean, bad = Pipeline_h(ds, None), Pipeline_h(ds, fault)
    np.testing.assert_allclose(clean(raw), bad(raw), equal_nan=True)         # canonical units: fault invisible
    r2, _ = op.apply(raw, ctx)
    assert not np.allclose(clean(r2), bad(r2), equal_nan=True)               # non-canonical units: fault visible


def Pipeline_h(ds, fault):
    class T:  # minimal stand-in exposing what Pipeline.canonical needs
        schema = ds.schema
    p = Pipeline(T, fault)
    return p.canonical


def test_lead_swap_is_involution_and_permutes_correct_columns(ds):
    class T:
        schema = ds.schema
    p = Pipeline(T, "F4_SWAP_V1_V2")
    perm = p.perm
    np.testing.assert_array_equal(perm[perm], np.arange(len(perm)))
    cols = list(ds.schema.index)
    i = cols.index("12sl::QRS_Amp_V1") if "12sl::QRS_Amp_V1" in cols else None
    if i is not None:
        assert cols[perm[i]] == "12sl::QRS_Amp_V2"
    moved = {ds.schema.lead.iloc[i] for i in range(len(perm)) if perm[i] != i}
    assert moved == {"V1", "V2"}


def test_limb_cycle_has_order_three(ds):
    class T:
        schema = ds.schema
    perm = Pipeline(T, "F5_LIMB_LEADS_CYCLED").perm
    np.testing.assert_array_equal(perm[perm[perm]], np.arange(len(perm)))


def test_ops_deterministic_and_do_not_mutate_input(ds, ctx):
    raw = to_raw(ds, ds.X.index[:200])
    orig = raw.X.copy()
    for op in all_ops():
        r1, m1 = op.apply(raw, ctx) if False else op.apply(raw, Ctx_for(ds, ds.X.index[:200]))
        r2, m2 = op.apply(raw, Ctx_for(ds, ds.X.index[:200]))
        np.testing.assert_array_equal(r1.X, r2.X)
        np.testing.assert_array_equal(raw.X, orig)


def Ctx_for(ds, idx):
    return prepare_ctx(Ctx(ds.schema, ds.meta.loc[idx], SEED))


def test_territory_mask_only_on_mi_with_territory(ds):
    idx = ds.X.index[:2000]
    c = Ctx_for(ds, idx)
    raw = to_raw(ds, idx)
    r2, mask = TerritoryMaskOp().apply(raw, c)
    m = ds.meta.loc[idx]
    assert not mask[(m.y == 0).to_numpy()].any()
    assert mask.sum() > 0
    changed = np.isnan(r2.X).sum(1) > np.isnan(raw.X).sum(1)
    assert (changed == mask).all()


def test_st_elevation_touches_only_st_amplitude_columns(ds):
    idx = ds.X.index[:300]
    c = Ctx_for(ds, idx)
    raw = to_raw(ds, idx)
    r2, _ = StElevationOp().apply(raw, c)
    diff = ~np.isclose(r2.X, raw.X, equal_nan=True)
    cols = np.where(diff.any(0))[0]
    assert len(cols) > 0
    for j in cols:
        assert ds.schema.unit.iloc[j] == "mV" and ds.schema.base.iloc[j].split("_")[0] == "ST"


def test_violation_semantics():
    p0, p1 = np.array([.5, .5, .5]), np.array([.56, .44, .5])
    assert violation("invariance", p0, p1, .05).tolist() == [True, True, False]
    assert violation("nonincrease", p0, p1, .05).tolist() == [True, False, False]
    assert violation("nondecrease", p0, p1, .05).tolist() == [False, True, False]


def test_bootstrap_weights_and_ci():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(100), 3)
    W = boot_weights(cl, 400, rng)
    assert W.shape == (400, 300)
    # every record of a cluster shares one weight, and each replicate draws exactly k clusters
    assert (W[:, ::3] == W[:, 1::3]).all()
    assert np.allclose(W[:, ::3].sum(1), 100)
    v = (rng.random(300) < 0.2).astype(float); m = np.ones(300)
    pt, lo, hi = rate_ci(W, v, m)
    assert lo <= pt <= hi


@pytest.mark.parametrize("op", [NoiseOp(1.0), StElevationOp()])
def test_operators_are_unit_tag_aware(ds, ctx, op):
    """The same physical perturbation must result whether the stream is in mV/ms or uV/s."""
    idx = ds.X.index[:300]
    c = Ctx_for(ds, idx)
    raw = to_raw(ds, idx)
    h = Harmonizer(ds.schema)
    a, _ = op.apply(raw, c)
    alt, _ = UnitOp("amp", "uV").apply(raw, c)
    alt, _ = UnitOp("time", "s").apply(alt, c)
    b, _ = op.apply(alt, c)
    np.testing.assert_allclose(h(a), h(b), rtol=1e-8, atol=1e-8, equal_nan=True)
