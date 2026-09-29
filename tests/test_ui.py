"""Headless UI tests: drive the real Streamlit app (coupon gate, investigation, approval, admin)."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT, get_settings

pytestmark = pytest.mark.integration
APP = str(PROJECT_ROOT / "frontend" / "streamlit_app.py")
ADMIN_PASSWORD = "test-admin-password"


@pytest.fixture()
def app(data_dir, tmp_path, monkeypatch):
    for var, value in {"DATA_DIR": str(data_dir), "VECTOR_DB_PATH": str(tmp_path / "vs"),
                       "CHECKPOINT_DB": str(tmp_path / "cp.sqlite"), "LLM_PROVIDER": "rule_based",
                       "UI_BACKEND": "local", "LOG_LEVEL": "WARNING", "COUPONS": "PYTHON2026:5",
                       "ADMIN_PASSWORD": ADMIN_PASSWORD, "ADMIN_STATE_PATH": str(tmp_path / "admin.json"),
                       "REQUIRE_COUPON": "true", "PROJECT_ENABLED": "true"}.items():
        monkeypatch.setenv(var, value)
    get_settings.cache_clear()
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    yield at
    get_settings.cache_clear()
    st.cache_resource.clear()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _unlock(at: AppTest, code: str = "python2026") -> None:
    at.text_input[0].input(code)
    _button(at, "Unlock InsightFlow AI").click().run()


def _text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_landing_requires_a_valid_code(app):
    assert not app.exception
    assert "Enter your access code" in _text(app)
    assert not any(b.label == "Use example sales dataset" for b in app.button)
    _unlock(app, "WRONG-CODE")
    assert any("Unknown code" in e.value for e in app.error)
    _unlock(app)
    assert any(b.label == "Use example sales dataset" for b in app.button)


def test_ui_investigation_flow(app):
    _unlock(app)
    _button(app, "Use example sales dataset").click().run()
    app.text_area(key="question").input("Why did European revenue decrease in Q3?").run()
    _button(app, "🚀 Start investigation").click().run()
    assert not app.exception
    assert "Executive Finding" in _text(app) and "Germany" in _text(app)
    assert any("Completed" in s.value for s in app.success)


def test_ui_approval_flow(app):
    _unlock(app)
    _button(app, "Use example sales dataset").click().run()
    app.text_area(key="question").input("Delete rows with negative revenue").run()
    _button(app, "🚀 Start investigation").click().run()
    assert any("ACTION REQUIRES APPROVAL" in w.value for w in app.warning)
    _button(app, "APPROVE").click().run()
    assert not app.exception
    assert any("Completed" in s.value for s in app.success)


def test_admin_login_switch_and_coupons(app):
    app.switch_page("views/admin.py").run()
    assert not app.exception
    app.text_input[0].input("wrong")
    _button(app, "Sign in").click().run()
    assert any("Wrong password" in e.value for e in app.error)
    app.text_input[0].input(ADMIN_PASSWORD)
    _button(app, "Sign in").click().run()
    assert "Admin dashboard" in _text(app)

    app.toggle(key="project_switch").set_value(False).run()  # switch the project OFF
    from ui_core import access_store

    assert access_store().project_enabled is False

    code_input = next(t for t in app.text_input if t.label == "Code")
    code_input.input("NEWCODE")
    _button(app, "Create code").click().run()
    assert "NEWCODE" in {c.code for c in access_store().coupons()}


def test_switched_off_project_blocks_visitors(app):
    from ui_core import access_store

    access_store().set_project_enabled(False)
    app.run()
    assert any("switched off" in w.value for w in app.warning)
    assert not any(b.label == "Unlock InsightFlow AI" for b in app.button)
