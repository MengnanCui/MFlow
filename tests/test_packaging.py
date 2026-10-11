"""The `mflow` command as a user meets it: overview, config, doctor, skills, tools, completion, packaging."""

import subprocess
import sys
from pathlib import Path

import pytest

from mflow.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A clean user: own HOME and config, no MFLOW_* overrides."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    for key in ("MFLOW_CONFIG", "MFLOW_MODEL", "MFLOW_LMP", "MFLOW_SKILLS_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_no_arguments_prints_the_overview(home, capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "mflow doctor" in out and "mflow md" in out and "mflow al" in out


def test_config_set_get_and_env_override(home, monkeypatch):
    from mflow import config
    assert config.lookup("model")[1].startswith("default")
    model = home / "cu.model"
    model.write_text("x")
    assert main(["config", "set", "model", str(model)]) == 0
    assert config.lookup("model") == (str(model), "config file")
    assert (home / ".config" / "mflow" / "config.yaml").exists()
    monkeypatch.setenv("MFLOW_MODEL", "omat")
    assert config.lookup("model") == ("omat", "env MFLOW_MODEL")
    monkeypatch.delenv("MFLOW_MODEL")
    assert main(["config", "unset", "model"]) == 0
    assert config.lookup("model")[1].startswith("default")
    assert main(["config", "set", "nonsense", "1"]) == 2


def test_lmp_from_config(home, monkeypatch):
    from mflow.lmp.env import find_lmp
    fake = home / "my_lmp"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    main(["config", "set", "lmp", str(fake)])
    monkeypatch.setenv("PATH", "/nonexistent")
    assert find_lmp() == str(fake.resolve())


def test_doctor_runs_and_reports(home, capsys):
    code = main(["doctor", "-json", str(home / "doctor.json")])
    assert code in (0, 2)
    import json
    rep = json.loads((home / "doctor.json").read_text())
    names = {c["name"] for c in rep["checks"]}
    assert {"package ase", "default model", "LAMMPS", "agent skills"} <= names


def test_skills_install_link_and_copy(home):
    from mflow.skills import installed_state, shipped
    assert len(shipped()) == 12
    assert main(["skills", "install"]) == 0
    dest = home / ".agents" / "skills"
    assert all((dest / s.name).is_symlink() for s in shipped())
    assert installed_state()[1] == {"total": 12, "installed": 12}
    assert main(["skills", "install"]) == 0                      # idempotent
    other = home / "pi"
    assert main(["skills", "install", "-dest", str(other), "-copy"]) == 0
    assert (other / "mflow-md" / "SKILL.md").is_file() and not (other / "mflow-md").is_symlink()


def test_tool_registry_and_izo(home, capsys):
    assert main(["tool"]) == 0
    assert "izo-formation" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["tool", "izo-formation", "-h"])
    assert exc.value.code == 0
    assert main(["tool", "nope"]) == 2
    shim = subprocess.run([sys.executable, str(ROOT / "tools" / "izo_formation.py"), "-h"],
                          capture_output=True, text=True)
    assert shim.returncode == 0 and "izo-formation" in shim.stdout


def test_missing_input_is_one_line(home, capsys):
    assert main(["check", "-in", "nothere.xyz", "-noplot"]) == 2
    assert "file not found" in capsys.readouterr().out


def test_completion_script(home):
    from mflow.completion import bash_script, table
    words = table()
    assert "md" in words[""] and "-protocol" in words["md"]
    assert "init" in words["al"] and "-system" in words["al init"]
    assert "eos" in words["validate"] and "-strain" in words["validate eos"]
    script = bash_script()
    assert "complete -o default -F _mflow_complete mflow" in script
    probe = subprocess.run(["bash", "-c", script + '\nCOMP_WORDS=(mflow md -pro); COMP_CWORD=2; _mflow_complete; '
                                                    'echo "${COMPREPLY[@]}"'], capture_output=True, text=True)
    assert probe.stdout.strip() == "-protocol"


def test_pyproject_lists_every_subpackage():
    tomllib = pytest.importorskip("tomllib")                     # python >= 3.11
    from setuptools import find_packages
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    listed = set(cfg["packages"])
    found = set(find_packages(str(ROOT), include=["mflow", "mflow.*"]))
    assert found <= listed, f"add to pyproject [tool.setuptools] packages: {sorted(found - listed)}"
    assert cfg["package-dir"]["mflow._skills"] == "skills"
