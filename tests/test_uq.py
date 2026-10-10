import numpy as np
import pytest

from mflow import thresholds as T
from mflow.uq import calibration, rankdata, spearman, verdict


def test_rankdata_handles_ties():
    assert rankdata([3, 1, 2, 2]).tolist() == [4.0, 1.0, 2.5, 2.5]


def test_spearman_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    b = a + 0.5 * b
    assert abs(spearman(a, b) - stats.spearmanr(a, b).statistic) < 1e-12


def test_calibrated_vs_overconfident():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.05, 0.5, 20000)
    delta = rng.normal(0, sigma)                       # errors drawn with exactly sigma
    good = calibration(delta, sigma, np.abs(delta), sigma)
    assert good["miscalibration_area"] < 0.02 and abs(good["sigma_ratio"] - 1) < 0.05
    assert good["spearman_atom"] > 0.3
    bad = calibration(3 * delta, sigma, np.abs(3 * delta), sigma)   # true errors 3x larger than claimed
    assert bad["miscalibration_area"] > 0.2 and bad["sigma_ratio"] > 2.5
    assert verdict(good)[0] in ("pass", "warn")


def test_threshold_override(tmp_path, monkeypatch):
    f = tmp_path / "t.yaml"
    f.write_text("md:\n  nve_drift_pass_eV_atom_ps: 1.0e-3\n")
    monkeypatch.setenv("MFLOW_THRESHOLDS", str(f))
    T.thresholds.cache_clear()
    try:
        md = T.section("md")
        assert md["nve_drift_pass_eV_atom_ps"] == 1e-3 and md["temperature_pass"] == 0.05   # merged, not replaced
    finally:
        T.thresholds.cache_clear()
