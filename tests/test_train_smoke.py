"""Two epochs of real mace_run_train on CPU (~20 s). Skipped without mace-torch."""

import json
import shutil
from argparse import Namespace

import ase.io
import pytest

from conftest import labelled

mace = pytest.importorskip("mace")


@pytest.mark.skipif(not shutil.which("mace_run_train"), reason="mace_run_train not on PATH")
def test_scratch_training_end_to_end(tmp_path):
    from mflow.train.run import cmd_train
    ase.io.write(tmp_path / "train.xyz", labelled(40, seed=1))
    ase.io.write(tmp_path / "test.xyz", labelled(9, seed=2))
    args = Namespace(input=str(tmp_path / "train.xyz"), index=":", name="smoke", outdir=str(tmp_path / "run"),
                     test=str(tmp_path / "test.xyz"), valid=None, ref_prefix=None, group="config_type",
                     mode="scratch", foundation=None, head=None, pt=None, epochs=2, batch=8, dtype="float64",
                     device="cpu", cueq="off", seed=1, dry=False, resume=False, force=True, no_plot=True,
                     set=["num_channels=8", "max_L=0", "r_max=4.0"])
    assert cmd_train(args) == 0
    report = json.loads((tmp_path / "run" / "smoke_train_report.json").read_text())
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["mace_run_train"] == "pass" and names["model file"] == "pass"
    assert report["curves"]["last_epoch"] == 1
    assert "errors" in report["test"]
