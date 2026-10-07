from pathlib import Path

UNIT = (Path(__file__).resolve().parents[1] / "deploy" / "codeagent.service").read_text()


def test_service_unit_has_the_required_sections_and_restarts_on_failure():
    for section in ("[Unit]", "[Service]", "[Install]"):
        assert section in UNIT
    assert "Restart=on-failure" in UNIT
    assert "WantedBy=default.target" in UNIT       # user service, no root needed


def test_service_runs_the_daemon_command_with_laptop_friendly_limits():
    assert "codeagent daemon --inbox" in UNIT
    assert "Nice=" in UNIT and "CPUQuota=" in UNIT and "MemoryMax=" in UNIT
