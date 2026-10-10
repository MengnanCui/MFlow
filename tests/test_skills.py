"""skills/: every SKILL.md follows the Agent Skills format and points at files that exist."""

import re
from pathlib import Path

import pytest
import yaml

SKILLS = sorted((Path(__file__).resolve().parents[1] / "skills").glob("*/SKILL.md"))


def frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert m, f"{path} has no YAML frontmatter"
    return yaml.safe_load(m.group(1)), m.group(2)


def test_there_are_twelve_skills():
    assert len(SKILLS) == 12


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_skill_format(path):
    meta, body = frontmatter(path)
    name = meta.get("name", "")
    assert name == path.parent.name                                   # spec: name = directory name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
    desc = meta.get("description", "")
    assert 50 <= len(desc) <= 1024 and "Use" in desc                  # says what it does and when to use it
    assert len(body.splitlines()) <= 300                              # details belong in references/
    for ref in re.findall(r"`(references/[^`]+)`", body):
        assert (path.parent / ref).exists(), f"{name}: {ref} missing"
    for other in re.findall(r"`(mflow-[a-z-]+)`", body):
        if other not in ("mflow-thresholds",):
            assert (path.parent.parent / other / "SKILL.md").exists(), f"{name} points at unknown skill {other}"
