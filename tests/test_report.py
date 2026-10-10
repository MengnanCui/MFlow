import json

from mflow.report import Report


def test_status_is_worst_check(tmp_path):
    rep = Report("x")
    assert rep.status == "pass"
    rep.check("a", "pass")
    rep.check("b", "warn", suggest="do b")
    assert rep.status == "warn" and not rep.failed
    rep.check("c", "fail", suggest="do c")
    rep.check("d", "pass", suggest="never shown")
    assert rep.status == "fail" and rep.failed
    assert rep.suggestions == ["do b", "do c"]
    data = json.loads(rep.write(tmp_path / "r.json").read_text())
    assert data["status"] == "fail" and len(data["checks"]) == 4
