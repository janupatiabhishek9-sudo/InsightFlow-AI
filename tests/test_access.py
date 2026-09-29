"""Project switch, coupons, AI switch and admin authentication (app/access.py)."""

from datetime import date, timedelta

import pytest
from pydantic import SecretStr

from app.access import AccessStore, mask, parse_coupon_seed
from app.config import Settings


def _store(tmp_path, **env) -> AccessStore:
    return AccessStore(Settings(_env_file=None, admin_state_path=str(tmp_path / "admin.json"), **env))


def test_coupon_seed_parsing():
    coupons = parse_coupon_seed("python2026:2:2030-01-01, demo ,VIP::2031-06-30")
    assert set(coupons) == {"PYTHON2026", "DEMO", "VIP"}
    assert coupons["PYTHON2026"].max_uses == 2 and coupons["PYTHON2026"].expires == date(2030, 1, 1)
    assert coupons["DEMO"].max_uses is None and coupons["VIP"].max_uses is None
    with pytest.raises(ValueError):
        parse_coupon_seed("a!")


def test_redeem_is_case_insensitive_and_counts_uses(tmp_path):
    store = _store(tmp_path, coupons="PYTHON2026:2")
    assert store.redeem(" python2026 ") is None
    assert store.redeem("PYTHON2026") is None
    assert store.redeem("PYTHON2026") == "This code has reached its usage limit."
    assert store.redeem("nope") == "Unknown code."
    assert store.coupons()[0].uses == 2


def test_expired_and_deactivated_codes(tmp_path):
    store = _store(tmp_path)
    store.add_coupon("OLD", expires=date.today() - timedelta(days=1))
    assert store.redeem("OLD") == "This code has expired."
    store.add_coupon("LIVE")
    assert store.redeem("LIVE") is None and store.check("LIVE") is None
    store.set_coupon_active("LIVE", False)
    assert store.check("LIVE") == "This code has been deactivated."  # an unlocked session is locked again
    with pytest.raises(ValueError):
        store.add_coupon("LIVE")
    with pytest.raises(ValueError):
        store.add_coupon("x")


def test_project_switch_blocks_everyone(tmp_path):
    store = _store(tmp_path, coupons="PYTHON2026")
    store.set_project_enabled(False)
    assert store.redeem("PYTHON2026") == "InsightFlow AI is switched off by the administrator."
    assert store.check("PYTHON2026") is not None
    store.set_project_enabled(True)
    assert store.redeem("PYTHON2026") is None


def test_changes_persist_and_config_codes_always_return(tmp_path):
    store = _store(tmp_path, coupons="SEED")
    store.add_coupon("ADMINMADE", max_uses=5)
    store.set_project_enabled(False)
    store.set_ai(False, groq_key="gsk_" + "x" * 30)
    reloaded = _store(tmp_path, coupons="SEED")  # simulated restart, same disk
    assert {c.code for c in reloaded.coupons()} == {"SEED", "ADMINMADE"}
    assert not reloaded.project_enabled and not reloaded.ai_enabled
    assert reloaded.groq_key_override.startswith("gsk_")
    (tmp_path / "admin.json").unlink()  # free host wiped its disk
    fresh = _store(tmp_path, coupons="SEED")
    assert [c.code for c in fresh.coupons()] == ["SEED"] and fresh.project_enabled


def test_other_process_changes_are_picked_up(tmp_path):
    ui, api = _store(tmp_path), _store(tmp_path)
    ui.set_project_enabled(False)
    assert api.project_enabled is False  # e.g. the API process sees the UI's switch


def test_admin_password(tmp_path):
    assert not _store(tmp_path).admin_configured
    store = _store(tmp_path, admin_password=SecretStr("correct-horse-battery"))
    assert store.admin_configured
    assert store.check_admin_password("correct-horse-battery")
    assert not store.check_admin_password("wrong")
    assert [e.event for e in store.audit(2)] == ["failed admin login", "admin login"]


def test_corrupt_state_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "admin.json").write_text("{not json")
    store = _store(tmp_path, coupons="SEED", project_enabled=True)
    assert [c.code for c in store.coupons()] == ["SEED"]


def test_mask_never_reveals_a_key():
    assert mask(None) == "not set"
    assert mask("gsk_abcdefghijklmnopqrstuvwxyz") == "gsk_…wxyz"


def test_ai_switch_changes_the_service_model(settings):
    from app.service import InsightFlowService

    svc = InsightFlowService(settings)
    assert svc.apply_ai_settings(False) == "rule_based/deterministic-v1" and svc.deps.llm is None
    assert svc.apply_ai_settings(True, groq_key="gsk_" + "k" * 30).startswith("groq/")
    assert svc.apply_ai_settings(False) == "rule_based/deterministic-v1"
