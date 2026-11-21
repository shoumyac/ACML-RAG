import numpy as np
import pytest
from ncsrag import RiskBinnedCRC, clopper_pearson_upper, decertify_on_drift, marginal_threshold


def test_cp_zero_failures_matches_closed_form():
    # k = 0: upper bound is 1 - delta**(1/n)
    assert clopper_pearson_upper(0, 10, 0.10) == pytest.approx(1 - 0.10 ** (1 / 10), abs=1e-9)


def test_cp_edge_cases():
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(5, 5) == 1.0
    assert clopper_pearson_upper(3, 100) > 0.03


def test_marginal_threshold_abstains_when_everything_is_risky():
    s = np.linspace(0, 1, 200)
    assert marginal_threshold(s, np.ones(200, int), alpha=0.1) == -np.inf


def _world(n, rng, noisy_share):
    """Risk score is calibrated: P(hallucination | score) = score."""
    noisy = rng.random(n) < noisy_share
    s = np.where(noisy, rng.beta(4, 2, n), rng.beta(1.2, 6, n))
    y = (rng.random(n) < s).astype(int)
    return s, y


def test_riskbin_certifies_only_low_risk_bins():
    rng = np.random.default_rng(0)
    s, y = _world(6000, rng, 0.3)
    crc = RiskBinnedCRC(alpha=0.10, n_bins=10).fit(s, y)
    for c in crc.certificates_:
        assert c.certified == (c.upper_bound <= 0.10)
    # every certified bin has a low score range
    assert all(c.high < 0.5 for c in crc.certificates_ if c.certified)


def test_riskbin_stays_valid_under_bin_reweighting_but_marginal_does_not():
    rng = np.random.default_rng(1)
    viol_rb = viol_m = 0
    trials = 60
    for t in range(trials):
        cs, cy = _world(4000, rng, 0.2)          # clean-heavy calibration
        ds, dy = _world(4000, rng, 0.8)          # noise-heavy deployment (reweights bins)
        crc = RiskBinnedCRC(alpha=0.10, n_bins=10).fit(cs, cy)
        risk, _ = crc.evaluate(ds, dy)
        viol_rb += risk > 0.10
        thr = marginal_threshold(cs, cy, 0.10)
        m = ds <= thr
        viol_m += (dy[m].mean() if m.sum() else 0.0) > 0.10
    assert viol_rb / trials <= 0.10       # the guarantee
    assert viol_m / trials >= 0.5         # the failure it fixes


def test_answer_mask_and_summary():
    rng = np.random.default_rng(2)
    s, y = _world(3000, rng, 0.3)
    crc = RiskBinnedCRC().fit(s, y)
    mask = crc.answer_mask(s)
    assert mask.dtype == bool and 0 < mask.mean() < 1
    assert "certified" in crc.summary()


def test_monitor_flags_drift_only_in_shifted_bins():
    rng = np.random.default_rng(3)
    s, y = _world(5000, rng, 0.3)
    crc = RiskBinnedCRC().fit(s, y)
    cal_flag = rng.random(len(s)) < 0.10
    ds, _ = _world(5000, rng, 0.3)
    benign = rng.random(len(ds)) < 0.10
    shifted = rng.random(len(ds)) < 0.40
    assert len(decertify_on_drift(crc, s, cal_flag, ds, shifted)) > len(decertify_on_drift(crc, s, cal_flag, ds, benign))
