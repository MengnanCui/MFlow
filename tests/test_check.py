
import numpy as np

from mflow.report import Report
from mflow.train.check import check_frames, e0_residuals, split_frames


def by_name(rep):
    return {c["name"]: c for c in rep.checks}


def test_clean_data_passes(frames):
    rep = Report("check")
    check_frames(frames, "REF_", rep, "config_type", has_test=True)
    checks = by_name(rep)
    for name in ("labels", "finite labels", "force magnitude", "closest pair", "duplicates", "energy outliers"):
        assert checks[name]["status"] == "pass", (name, checks[name])


def test_every_planted_error_is_found(frames):
    frames[3].arrays["REF_forces"][0, 0] = np.nan
    frames[5] = frames[4].copy()
    frames[7].positions[1] = frames[7].positions[0] + [0.3, 0, 0]
    frames[9].info["REF_energy"] += 50.0
    frames[11].arrays["REF_forces"][2] = [80, 0, 0]
    rep = Report("check")
    flags = check_frames(frames, "REF_", rep, None, has_test=True)
    assert flags["nan"] == [3]
    assert flags["duplicate"] == [5]
    assert flags["close"] == [7]
    assert flags["e_outlier"] == [9]
    assert flags["f_warn"] == [11]
    assert rep.failed


def test_outlier_fit_is_robust(frames):
    without = e0_residuals(frames[1:], "REF_")
    frames[0].info["REF_energy"] += 100.0
    resid = e0_residuals(frames, "REF_")
    assert np.argmax(np.abs(resid)) == 0
    assert np.allclose(resid[1:], without, atol=1.0)   # same fit as if the outlier were not there


def test_split_is_stratified(frames):
    groups = np.array([a.info["config_type"] for a in frames])
    train, test = split_frames(frames, 0.2, groups, seed=1)
    assert len(train) + len(test) == len(frames)
    assert {a.info["config_type"] for a in test} == {"Cu", "Al", "CuAl"}


def test_force_breakdown_by_element(frames):
    from mflow.calc import compare
    for a in frames:
        a.info["p_energy"] = a.info["REF_energy"] + 0.01 * len(a)
        a.arrays["p_forces"] = a.arrays["REF_forces"] + np.where(a.numbers[:, None] == 29, 0.1, 0.0)
    errors, _ = compare(frames, "p_", "REF_", "none")
    by = errors["forces_by_element"]
    assert abs(by["Cu"]["RMSE"] - 100.0) < 1e-6 and by["Al"]["RMSE"] < 1e-9
