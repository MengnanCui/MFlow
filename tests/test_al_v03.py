"""v0.3 active learning: convergence levels, adaptive-lower, regions, committee options, coverage,
probes and the new explorers. Committees here are made of ASE calculators, so no MACE is needed
except in the end-to-end campaign at the bottom."""

import json
import shutil
from argparse import Namespace

import ase.io
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from conftest import labelled


class Scaled(Calculator):
    """EMT times a factor — a committee member that disagrees smoothly with plain EMT."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, factor, **kw):
        super().__init__(**kw)
        self.factor, self.emt = factor, EMT()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy()
        a.calc = self.emt
        e = a.get_potential_energy() * self.factor
        self.results = {"energy": e, "free_energy": e, "forces": a.get_forces() * self.factor}


def committee():
    return [EMT(), Scaled(1.05), Scaled(0.93)]


def cu(n=2, rattle=0.05, seed=0):
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(n)
    a.rattle(rattle, seed=seed)
    return a


def make_campaign(root, settings=None, state=None, ledger=()):
    import yaml
    root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "labelled").mkdir(parents=True, exist_ok=True)
    (root / "campaign.yaml").write_text(yaml.safe_dump(settings or {}))
    (root / "state.json").write_text(json.dumps(state or {"round": 0, "stage": "init", "models": []}))
    (root / "ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ledger))
    ase.io.write(root / "data" / "system.xyz", [cu(1, 0.0)])
    from mflow.al.campaign import Campaign
    return Campaign(root)


def sel(r, acc, cand, fail, n_sel=0, **kw):
    n = acc + cand + fail
    return {"step": "select", "round": r, "n_explored": n, "n_accurate": acc, "n_candidate": cand, "n_failed": fail,
            "candidate_fraction": cand / n, "n_selected": n_sel, "trust_mode": "fixed", **kw}


def imp(r, e, f, model="committee", accepted=50):
    return {"step": "import", "round": r, "accepted": accepted,
            "fresh_frame_error": {"model": model, "energy_RMSE": e, "force_RMSE": f, "n": 10}}


# ============================================================================ #
#  L0 / L1 / L2 / budget
# ============================================================================ #

def test_adaptive_lower_quantile_and_freezing():
    from mflow.al.convergence import adaptive_lo, lo_is_stable
    score = np.linspace(0.0, 1.0, 1001)                      # 0.000 .. 1.000
    lo = adaptive_lo(score, hi=0.5, numb=100, rate=0.01)      # 100 frames in [lo, 0.5)
    assert ((score >= lo) & (score < 0.5)).sum() == 100
    assert adaptive_lo(score, hi=0.5, numb=5, rate=0.2) == pytest.approx(0.3, abs=2e-3)   # rate wins: 200 frames
    assert adaptive_lo(np.array([0.9, 1.0]), hi=0.5, numb=10, rate=0.01) == 0.5            # nothing below hi
    assert lo_is_stable([0.20, 0.12, 0.10, 0.09], tol=0.05, n=2)
    assert not lo_is_stable([0.20, 0.12, 0.05], tol=0.05, n=2)                             # 0.07 jump
    assert not lo_is_stable([0.10, 0.10], tol=0.05, n=2)                                   # too short


def test_convergence_levels_consecutive_and_na(tmp_path):
    from mflow.al.convergence import evaluate
    stop = {"stop": {"candidate_fraction": 0.05, "dft_energy_meV_atom": 10, "dft_force_meV_A": 50}}
    led = [sel(1, 995, 4, 1, n_sel=4), imp(1, 3.0, 30.0),
           {"step": "train", "round": 1, "test_force_RMSE": 25.0}]
    camp = make_campaign(tmp_path / "a", stop, ledger=led)
    ev = evaluate(camp)
    rows = {name: ok for _, name, ok, _, _ in ev["rows"]}
    assert rows["accurate ratio"] is None                    # one classified round, two needed: n/a
    assert rows["candidate fraction"] and rows["first-seen energy"] and rows["first-seen forces"]
    assert rows["first-seen / test forces"] is True          # 30 / 25 = 1.2 <= 1.5
    assert rows["exploration MD finished"] is None
    assert rows["properties stable between rounds"] is None
    assert ev["verdict"] == "converged"                       # n/a never blocks; the v0.2 core rules hold

    led2 = [sel(1, 995, 4, 1), sel(2, 980, 20, 0, n_sel=20), imp(2, 3.0, 30.0)]
    ev = evaluate(make_campaign(tmp_path / "b", stop, ledger=led2))
    rows = {name: ok for _, name, ok, _, _ in ev["rows"]}
    assert rows["accurate ratio"] is False                    # 98 % in the second round
    assert rows["frames left after FPS"] is False             # 20 >= 10
    assert ev["verdict"] == "continue"


def test_budget_exhausted_is_not_converged(tmp_path):
    from mflow.al.convergence import evaluate
    settings = {"stop": {"candidate_fraction": 0.05, "dft_energy_meV_atom": 10, "dft_force_meV_A": 50,
                         "max_rounds": 2, "max_dft": 1000}}
    led = [sel(0, 0, 100, 0), imp(0, 30, 300, model="mpa", accepted=100), sel(1, 50, 50, 0), imp(1, 20, 200)]
    ev = evaluate(make_campaign(tmp_path / "c", settings, ledger=led))
    assert ev["verdict"] == "budget_exhausted"
    rows = {name: (ok, val) for lvl, name, ok, val, _ in ev["rows"] if lvl == "budget"}
    assert rows["max_rounds"][0] is False and rows["max_dft"][0] is True


def test_md_failures_block_and_adaptive_open_blocks(tmp_path):
    from mflow.al.convergence import evaluate
    stop = {"stop": {"candidate_fraction": 0.05, "dft_energy_meV_atom": 10, "dft_force_meV_A": 50},
            "select": {"trust": "adaptive-lower"}}
    led = [{"step": "explore", "round": 1, "counts": {"md_runs": 4, "md_failed": 1}},
           sel(1, 999, 1, 0), sel(2, 999, 1, 0), imp(2, 1.0, 10.0)]
    ev = evaluate(make_campaign(tmp_path / "d", stop, ledger=led))
    rows = {name: ok for _, name, ok, _, _ in ev["rows"]}
    assert rows["exploration MD finished"] is False
    assert rows["accurate ratio"] is False                    # adaptive window not frozen yet
    assert ev["verdict"] == "continue"


# ============================================================================ #
#  regions
# ============================================================================ #

def test_region_cfg_grows_and_regions_advance(tmp_path):
    from mflow.al.explore import advance_region, region_cfg
    md = {"temps": [300], "press": [1.0], "steps": 1000, "dump": 10}
    cfg, grid = region_cfg(md, {"name": "hot", "temps": [900, 1200], "grow": 2.0, "max_steps": 3000}, k=2)
    assert cfg["steps"] == 3000 and grid == [(900.0, 1.0), (1200.0, 1.0)]
    settings = {"stop": {"candidate_fraction": 0.05, "dft_energy_meV_atom": 10, "dft_force_meV_A": 50},
                "explore": {"md": {"regions": [{"name": "cold", "temps": [100]}, {"name": "hot", "temps": [900]}]}}}
    good = [sel(1, 999, 1, 0, region="cold"), sel(2, 999, 1, 0, region="cold"), imp(2, 1.0, 10.0)]
    camp = make_campaign(tmp_path / "r", settings, ledger=good)
    assert advance_region(camp)["name"] == "hot"
    assert camp.state["region_index"] == 1
    camp2 = make_campaign(tmp_path / "r2", settings, ledger=[sel(1, 900, 100, 0, region="cold")])
    assert advance_region(camp2)["name"] == "cold"
    from mflow.al.convergence import evaluate
    ev = evaluate(camp)
    assert ev["region"] == "hot" and ev["regions"]["cold"] is True and ev["regions"]["hot"] is False
    assert ev["verdict"] == "continue"                         # hot has not been classified yet


# ============================================================================ #
#  committee statistics, collapse, biased MD
# ============================================================================ #

def test_committee_stats_and_without_driver():
    from mflow.calc import committee_stats, evaluate_calculators
    F = np.zeros((3, 2, 3))
    F[1, 0, 0], F[2, 0, 0] = 1.0, -1.0                        # member 0 sits at the mean
    s = committee_stats([0.0, 0.1, -0.1], F)
    assert s["Fdev_max"] == pytest.approx(np.sqrt(2 / 3))
    assert s["Fdev_max_wo0"] == pytest.approx(1.0)             # members 1, 2 only
    frames = [cu(1, 0.1, seed=s_) for s_ in range(3)]
    evaluate_calculators(frames, committee(), "c_")
    assert all(a.info["c_Fdev_max"] > 0 and "c_Fdev_max_wo0" in a.info for a in frames)
    assert frames[0].arrays["c_Fdev"].shape == (4,)


def test_collapse_verdict():
    from mflow.uq import verdict
    status, msg = verdict({"spearman_atom": 0.8, "sigma_ratio": 5.0})
    assert status == "warn" and "COLLAPSED" in msg
    assert verdict({"spearman_atom": 0.8, "sigma_ratio": 1.1})[0] == "pass"
    assert verdict({"spearman_atom": 0.1, "sigma_ratio": 4.0})[0] == "fail"


def test_biased_committee_forces_are_minus_gradient():
    from mflow.al.bias import BiasedCommitteeCalculator
    a = cu(1, 0.15, seed=3)
    a.calc = BiasedCommitteeCalculator(committee(), kappa=2.0)
    F = a.get_forces()
    h, num = 1e-5, np.zeros_like(F)
    for i in range(len(a)):
        for k in range(3):
            for sgn in (1, -1):
                b = a.copy()
                b.positions[i, k] += sgn * h
                b.calc = BiasedCommitteeCalculator(committee(), kappa=2.0)
                num[i, k] -= sgn * b.get_potential_energy() / (2 * h)
    assert np.allclose(F, num, atol=1e-5)
    assert a.calc.results["sigma_E"] > 0


def test_ubmd_explores_where_members_disagree(tmp_path):
    from mflow.al.bias import BiasedCommitteeCalculator
    from mflow.al.explore import explore_ubmd
    cfg = {"kappa": 3.0, "temps": [300.0], "steps": 200, "dump": 20, "dt": 2.0}
    start = [cu(2, 0.0)]
    biased = explore_ubmd(start, cfg, [], tmp_path, 1, calcs=committee())
    plain = explore_ubmd(start, {**cfg, "kappa": 0.0}, [], tmp_path, 1, calcs=committee())
    def sig(frames):
        out = []
        for f in frames[3:]:
            f.calc = BiasedCommitteeCalculator(committee(), 0.0)
            f.get_potential_energy()
            out.append(f.calc.results["sigma_E"])
        return np.mean(out)
    assert len(biased) == len(plain) == 11              # step 0 included
    assert sig(biased) > sig(plain)
    assert explore_ubmd(start, cfg, ["one.model"], tmp_path, 1) == []      # needs two members


# ============================================================================ #
#  new explorers
# ============================================================================ #

def test_heatquench_and_rss_with_emt(tmp_path):
    from mflow.al.explore import explore_heatquench, explore_rss, random_cell
    stats = {}
    hq = explore_heatquench([cu(2, 0.0)], {"T_high": 1500.0, "T_low": 300.0, "steps_hot": 100, "steps_quench": 100,
                                           "dump": 20, "dt": 2.0}, "", tmp_path, 0, calc=EMT(), stats=stats)
    assert len(hq) == 10 and stats == {"md_runs": 1, "md_failed": 0}
    T = [a.info["mflow_T"] for a in hq]
    assert T[0] == 1500.0 and T[-1] == 300.0 and all(a.info["mflow_source"] == "heatquench" for a in hq)
    rng = np.random.default_rng(0)
    rc = random_cell(cu(2, 0.0), rng, 0.2, 0.75)
    assert rc is not None and len(rc) == 32
    d = rc.get_all_distances(mic=True)[np.triu_indices(32, 1)]
    assert d.min() > 0.75 * 2 * 1.32 - 1e-9
    rss = explore_rss([cu(1, 0.0)], {"n": 3, "vol_jitter": 0.2, "min_dist": 0.75, "fmax": 0.1, "steps": 60,
                                     "every": 10}, "", tmp_path, 0, calc=EMT())
    ends = [a for a in rss if "mflow_E_explore" in a.info]
    assert len(ends) == 3 and all(a.calc is None for a in rss)


def test_md_blew_up_detects_collapse():
    from mflow.al.explore import md_blew_up
    a = cu(1, 0.0)
    assert not md_blew_up(a)
    a.positions[1] = a.positions[0] + [0.2, 0, 0]
    assert md_blew_up(a)


def test_plumed_line_and_skiprun_rename(tmp_path):
    from mflow.lmp.inputs import MDParams, md_input
    from mflow.lmp.run import skiprun_input
    p = MDParams(species=["Cu"], pair_style="eam/alloy Cu.eam", steps=100, plumed=str(tmp_path / "plumed.dat"))
    text = md_input(p)
    assert f"fix             mflow_plumed all plumed plumedfile {tmp_path / 'plumed.dat'} outfile plumed.log" in text
    assert "outfile skiprun_plumed.log" in skiprun_input(p)
    assert "plumed" not in md_input(MDParams(species=["Cu"], pair_style="x", steps=10))


# ============================================================================ #
#  coverage and selection without a committee
# ============================================================================ #

def test_descriptor_ood_threshold():
    from mflow.al.coverage import descriptor_scores, nn_distance
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 4))
    near, far = rng.normal(size=(50, 4)) * 0.5, rng.normal(size=(5, 4)) + 8.0
    d, thr = descriptor_scores(np.vstack([near, far]), train, 0.99)
    assert (d[:50] > thr).mean() < 0.1 and (d[50:] > thr).all()
    assert nn_distance(np.zeros((1, 2)), np.array([[3.0, 4.0], [1.0, 0.0]]))[0] == pytest.approx(1.0)


def test_quests_flags_new_environments():
    pytest.importorskip("quests")
    from mflow.al.coverage import quests_scores
    train = [cu(2, 0.02, seed=s) for s in range(4)]
    new = [cu(2, 0.02, seed=10), bulk("Cu", "bcc", a=3.6).repeat(3)]
    new[1].set_cell(new[1].cell * 0.85, scale_atoms=True)
    q = quests_scores(new, train)
    assert q["dH_max"][1] > 0 > q["dH_max"][0]
    assert q["dH_pos_fraction"] == pytest.approx(0.5) and q["entropy_train"] > 0


def test_selection_without_committee_uses_descriptor_distance(tmp_path, monkeypatch):
    import mflow.al.select as S
    train = [cu(1, 0.02, seed=s) for s in range(30)]
    for a in train:
        a.info["REF_energy"] = 0.0
    ase.io.write(tmp_path / "train.xyz", train)
    camp = make_campaign(tmp_path / "s", {"select": {"max_per_round": 3}},
                         state={"round": 1, "stage": "explored", "models": ["single.model"],
                                "dataset": str(tmp_path / "train.xyz")})
    feats = lambda frames, model: np.array([[np.abs(a.positions - a.get_scaled_positions() @ a.cell).sum(),
                                             a.get_volume()] for a in frames])
    monkeypatch.setattr(S, "descriptors", feats)
    frames = [cu(1, 0.02, seed=100 + s) for s in range(10)] + [cu(1, 0.4, seed=200 + s) for s in range(5)]
    for a in frames[-5:]:
        a.set_cell(a.cell * 1.1, scale_atoms=True)
    selected, stats = S.run_selection(camp, 1, frames)
    assert stats["uncertainty"] == "descriptor"
    assert stats["n_candidate"] >= 5 and stats["n_selected"] == 3
    assert stats["ood_fraction"] >= 5 / 15
    assert all(a.info["mflow_dev"] > stats["trust_lo"] for a in selected)
    assert (camp.root / "round_01" / "select" / "coverage.json").exists()


# ============================================================================ #
#  probes
# ============================================================================ #

def test_probes_stable_and_member_spread(tmp_path):
    from mflow.al.probes import compare, spread, step_probe
    camp = make_campaign(tmp_path / "p", {"probes": [{"kind": "eos", "index": 0, "npts": 7},
                                                     {"kind": "dimer", "elements": ["Cu"], "dr": 0.05}]},
                         state={"round": 1, "stage": "trained", "models": ["m0", "m1"]})
    first = step_probe(camp, 1, calc_factory=lambda m: EMT())
    assert first["stable"] is None and first["spread_ok"] is True
    assert set(first["values"]) >= {"eos0:V0", "eos0:B0", "dimer:Cu-Cu:r_eq", "dimer:Cu-Cu:wall"}
    second = step_probe(camp, 2, calc_factory=lambda m: EMT())
    assert second["stable"] is True
    tol = second["tolerance"]
    rows = compare({"eos0:V0": 10.0, "eos0:B0": 100.0}, {"eos0:V0": 10.2, "eos0:B0": 102.0}, tol)
    assert [r["ok"] for r in rows] == [False, True]           # V0 moved 2 % (> 0.5 %), B0 2 % (< 5 %)
    ok, msg = spread({"a": {"eos0:V0": 10.0}, "b": {"eos0:V0": 10.3}}, tol)
    assert ok is False and "eos0:V0" in msg
    probes = [x for x in camp.ledger() if x["step"] == "probe"]
    assert [p["stable"] for p in probes] == [None, True]


# ============================================================================ #
#  end to end: single model, descriptor selection, regions, probes, new explorers
# ============================================================================ #

pytest.importorskip("mace")


@pytest.mark.skipif(not shutil.which("mace_run_train"), reason="mace_run_train not on PATH")
def test_single_model_campaign_with_regions_and_probes(tmp_path, monkeypatch):
    from mflow.al.campaign import Campaign, cmd_al_init
    from mflow.al.commands import cmd_al_import, cmd_al_run, cmd_al_status
    from mflow.train.run import train_one
    from test_al import emt_label

    monkeypatch.chdir(tmp_path)
    ase.io.write("data.xyz", labelled(30, seed=1))
    ase.io.write("test.xyz", labelled(6, seed=2))
    ase.io.write("system.xyz", labelled(3, seed=3)[:1])
    base = Namespace(input="data.xyz", index=":", name="base", outdir="base", test=None, valid=None, ref_prefix=None,
                     group=None, mode="scratch", foundation=None, head=None, pt=None, epochs=2, batch=8,
                     dtype="float64", device="cpu", cueq="off", seed=1, dry=False, resume=False, force=True,
                     no_plot=True, set=["num_channels=8", "max_L=0", "r_max=4.0"], committee=1, bagging=None)
    assert train_one(base) == 0
    assert cmd_al_init(Namespace(name="camp", system="system.xyz", data="data.xyz", test="test.xyz",
                                 foundation=str(tmp_path / "base" / "base.model"), mode="finetune",
                                 committee=1, engine="ase", force=False)) == 0
    with open("camp/campaign.yaml", "a") as fh:
        fh.write("train: {committee: 1, epochs: 1, dtype: float64, device: cpu}\nseed: {n_select: 4}\n"
                 "explore:\n  md:\n    temps: [300]\n    steps: 20\n    dump: 10\n"
                 "    regions: [{name: cold, temps: [200], grow: 2.0}, {name: hot, temps: [800]}]\n"
                 "  perturb: {n: 2}\n"
                 "  heatquench: {enabled: true, T_high: 1200, T_low: 300, steps_hot: 20, steps_quench: 20, dump: 10}\n"
                 "  rss: {enabled: true, n: 2, steps: 20, every: 10}\n"
                 "select: {max_per_round: 4, uncertainty: descriptor}\n"
                 "stop: {max_rounds: 5}\n"
                 "probes: [{kind: eos, index: 0, npts: 5}]\n")
    run = Namespace(name="camp", lmp=None, force=False)
    assert cmd_al_run(run) == 0                                    # round 0 → export
    emt_label("camp/round_00/export/candidates.extxyz", "lab0.xyz")
    assert cmd_al_import(Namespace(name="camp", round=None, tflow=None, labelled="lab0.xyz")) == 0
    assert cmd_al_run(run) == 0                                    # train (1 model) → probe → explore → select
    camp = Campaign("camp")
    led = camp.ledger()
    assert camp.state["round"] == 1 and len(camp.committee()) == 1
    tr = next(x for x in led if x["step"] == "train")
    assert tr["test_force_RMSE"] is not None
    se = [x for x in led if x["step"] == "select"][-1]
    assert se["uncertainty"] == "descriptor" and se["region"] == "cold" and "ood_fraction" in se
    ex = [x for x in led if x["step"] == "explore"][-1]
    assert ex["counts"]["heatquench"] > 0 and ex["counts"]["rss"] > 0 and ex["counts"]["md_runs"] >= 2
    assert any(x["step"] == "probe" for x in led)
    assert cmd_al_status(Namespace(name="camp")) == 0
    status = json.loads((tmp_path / "camp" / "status.json").read_text())
    assert status["verdict"] in ("continue", "budget_exhausted") and status["region"] == "cold"


@pytest.mark.skipif(not shutil.which("mace_run_train"), reason="mace_run_train not on PATH")
def test_committee_campaign_mixed_foundations_adaptive_drivers_ubmd(tmp_path, monkeypatch):
    from mflow.al.campaign import Campaign, cmd_al_init
    from mflow.al.commands import cmd_al_import, cmd_al_run
    from mflow.train.run import train_one
    from test_al import emt_label

    monkeypatch.chdir(tmp_path)
    ase.io.write("data.xyz", labelled(30, seed=1))
    ase.io.write("test.xyz", labelled(6, seed=2))
    ase.io.write("system.xyz", labelled(3, seed=3)[:1])
    for name, seed in (("base", 1), ("base2", 2)):
        args = Namespace(input="data.xyz", index=":", name=name, outdir=name, test=None, valid=None, ref_prefix=None,
                         group=None, mode="scratch", foundation=None, head=None, pt=None, epochs=2, batch=8,
                         dtype="float64", device="cpu", cueq="off", seed=seed, dry=False, resume=False, force=True,
                         no_plot=True, set=["num_channels=8", "max_L=0", "r_max=4.0"], committee=1, bagging=None)
        assert train_one(args) == 0
    b1, b2 = str(tmp_path / "base" / "base.model"), str(tmp_path / "base2" / "base2.model")
    assert cmd_al_init(Namespace(name="camp", system="system.xyz", data="data.xyz", test="test.xyz", foundation=b1,
                                 mode="finetune", committee=2, engine="ase", force=False)) == 0
    with open("camp/campaign.yaml", "a") as fh:
        fh.write(f"train: {{committee: 2, epochs: 1, dtype: float64, device: cpu, foundations: [{b1}, {b2}]}}\n"
                 "seed: {n_select: 4}\n"
                 "explore:\n  md: {temps: [300], steps: 20, dump: 10, drivers: all}\n  perturb: {n: 2}\n"
                 "  ubmd: {enabled: true, kappa: 0.5, temps: [300], steps: 20, dump: 10}\n"
                 "select: {max_per_round: 4, trust: adaptive-lower}\n")
    run = Namespace(name="camp", lmp=None, force=False)
    assert cmd_al_run(run) == 0
    emt_label("camp/round_00/export/candidates.extxyz", "lab0.xyz")
    assert cmd_al_import(Namespace(name="camp", round=None, tflow=None, labelled="lab0.xyz")) == 0
    assert cmd_al_run(run) == 0
    camp = Campaign("camp")
    cj = json.loads((camp.root / "round_01" / "train" / "committee.json").read_text())
    assert [m["foundation"] for m in cj["members"]] == [b1, b2]
    led = camp.ledger()
    ex = [x for x in led if x["step"] == "explore"][-1]
    assert ex["counts"]["ubmd"] > 0 and ex["counts"]["md_runs"] == 2         # two drivers × one temperature
    se = [x for x in led if x["step"] == "select"][-1]
    assert se["uncertainty"] == "committee" and se["trust_mode"] == "adaptive-lower"
    assert se["n_candidate"] > 0
    explored = ase.io.read(camp.root / "round_01" / "explore" / "explored.xyz", ":")
    assert {a.info.get("mflow_driver") for a in explored if a.info.get("mflow_source") == "md"} == {0, 1}
