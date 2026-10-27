import os
ROOT = os.environ.get("JBHI_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import sys, numpy as np, pytest
sys.path.insert(0, f"{ROOT}/code")
from common import *
from ops import *
from faults2 import FX, PipelineX

D = f"{ROOT}/data"


@pytest.fixture(scope="module")
def ds():
    return build_dataset(D)


@pytest.fixture(scope="module")
def setup(ds):
    class T: schema = ds.schema
    raw = to_raw(ds, ds.X.index[:500])
    return T, raw, Harmonizer(ds.schema)(raw)


def canon(T, raw, name): return PipelineX(T, name)._base(raw)


def test_misalign_is_row_shift_of_one_view(ds, setup):
    T, raw, base = setup; X = canon(T, raw, "MISALIGN_ecgdeli")
    ix = np.where(ds.schema.view.to_numpy() == "ecgdeli")[0]; other = np.setdiff1d(np.arange(base.shape[1]), ix)
    np.testing.assert_array_equal(X[:, other], base[:, other])
    np.testing.assert_array_equal(X[:, ix], np.roll(base[:, ix], 1, axis=0))


def test_shiftcol_is_permutation_of_view_columns(ds, setup):
    T, raw, base = setup; X = canon(T, raw, "SHIFTCOL_12sl")
    ix = np.where(ds.schema.view.to_numpy() == "12sl")[0]
    np.testing.assert_array_equal(np.sort(np.nan_to_num(X[:, ix], nan=-9e9), 1), np.sort(np.nan_to_num(base[:, ix], nan=-9e9), 1))
    assert not np.array_equal(np.nan_to_num(X[:, ix]), np.nan_to_num(base[:, ix]))


def test_missing_and_corrupt_fractions(ds, setup):
    T, raw, base = setup; ns = np.where(ds.schema.view.to_numpy() != "struct")[0]
    base_nan = np.isnan(base[:, ns]).mean()
    for p in (5, 20, 50):
        X = canon(T, raw, f"MISS_{p}"); got = np.isnan(X[:, ns]).mean() - base_nan
        assert abs(got - (1 - base_nan) * p / 100) < 0.01
        Xc = canon(T, raw, f"CORRUPT_{p}"); changed = (~np.isclose(Xc[:, ns], base[:, ns], equal_nan=True)).mean()
        assert 0.3 * p / 100 < changed < 1.1 * p / 100
    # corruption swaps in same-feature values from other records, so the overall missing fraction is unchanged in expectation
    assert abs(np.isnan(canon(T, raw, "CORRUPT_20")[:, ns]).mean() - base_nan) < 0.01


def test_truncation_sets_trailing_columns_missing(ds, setup):
    T, raw, base = setup; X = canon(T, raw, "TRUNC_30")
    for v in VIEWS:
        ix = np.where(ds.schema.view.to_numpy() == v)[0]; k = int(np.ceil(0.7 * len(ix)))
        assert np.isnan(X[:, ix[k:]]).all()
        np.testing.assert_array_equal(X[:, ix[:k]], base[:, ix[:k]])


def test_stale_rows_copy_previous_record(ds, setup):
    T, raw, base = setup; X = canon(T, raw, "STALE_12sl_30")
    ix = np.where(ds.schema.view.to_numpy() == "12sl")[0]
    changed = np.where((~np.isclose(X[:, ix], base[:, ix], equal_nan=True)).any(1))[0]
    assert 0.15 < len(changed) / 500 < 0.45
    for r in changed[:20]: np.testing.assert_array_equal(X[r, ix], base[r - 1, ix])


def test_faults_are_deterministic(ds, setup):
    T, raw, _ = setup
    for n in ("MISS_20", "CORRUPT_20", "STALE_12sl_30", "PARTIAL_50"):
        np.testing.assert_array_equal(canon(T, raw, n), canon(T, raw, n))


def test_active_unit_faults_scale_canonical_values_and_correct_pipeline_does_not(ds, setup):
    T, raw, base = setup
    c = prepare_ctx(Ctx(ds.schema, ds.meta.loc[raw.cols and ds.X.index[:500]], SEED)); c.cols_arr = np.array(raw.cols)
    from ops import _convert
    alt = _convert(raw.copy(), c, amp="uV")
    ok = canon(T, alt, "ALTUNITS_OK_AMP"); np.testing.assert_allclose(ok, base, rtol=1e-8, atol=1e-8, equal_nan=True)
    bad = canon(T, alt, "ACT_F1_AMP"); ix = np.where(ds.schema.unit.to_numpy() == "mV")[0]
    j = ix[np.isfinite(base[:, ix]).all(0) & (np.abs(base[:, ix]).mean(0) > 0)][0]
    np.testing.assert_allclose(bad[:, j], base[:, j] * 1000, rtol=1e-8)
