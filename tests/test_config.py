import pytest

from mflow.train import config as C

BASE = dict(name="t", train_file="/x/train.xyz", valid_file=None, test_file=None, head=None, pt_file=None,
            e0s=None, has_stress=True, device="cpu", dtype="float32", cueq=False, seed=1, batch=4,
            epochs=None, work_dir="/x")


@pytest.mark.parametrize("mode,lr,ema", [("finetune", 1e-3, 0.999), ("multihead", 1e-4, 0.9999),
                                         ("lora", 1e-2, 0.99)])
def test_mode_defaults(mode, lr, ema):
    cfg = C.build_config(mode=mode, foundation="medium-mpa-0", **BASE)
    assert cfg["lr"] == lr and cfg["ema_decay"] == ema
    assert cfg["E0s"] == "estimated" and cfg["stress_weight"] == 1.0
    assert cfg["multiheads_finetuning"] is (mode == "multihead")
    if mode == "multihead":
        assert cfg["pt_train_file"] == "mp"


def test_mh1_gets_a_head_and_scratch_needs_no_foundation():
    cfg = C.build_config(mode="finetune", foundation=C.resolve_foundation("mh1"), **BASE)
    assert cfg["foundation_model"] == "mh-1" and cfg["foundation_head"] == "omat_pbe"
    cfg = C.build_config(mode="scratch", foundation=None, **BASE)
    assert "foundation_model" not in cfg and cfg["E0s"] == "average" and cfg["swa"] is True
    with pytest.raises(ValueError):
        C.build_config(mode="finetune", foundation=None, **BASE)


def test_set_overrides_are_parsed():
    extra = C.parse_set(["lr=5e-4", "max_L=2", "swa=true"])
    assert extra == {"lr": 5e-4, "max_L": 2, "swa": True}


def test_keys_validated_against_installed_mace():
    allowed = C.mace_arg_names()
    if allowed is None:
        pytest.skip("mace not installed")
    for mode in C.MODES:
        cfg = C.build_config(mode=mode, foundation="medium-mpa-0" if mode != "scratch" else None, **BASE)
        assert C.validate(cfg, allowed) == [], mode
    assert C.validate({"not_a_mace_option": 1}, allowed) == ["not_a_mace_option"]
