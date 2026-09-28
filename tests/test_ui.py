"""Headless UI test: drive the real Streamlit app with Streamlit's AppTest harness."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT, get_settings

pytestmark = pytest.mark.integration
APP = str(PROJECT_ROOT / "frontend" / "streamlit_app.py")


@pytest.fixture()
def app(data_dir, tmp_path, monkeypatch):
    for var, value in {"DATA_DIR": str(data_dir), "VECTOR_DB_PATH": str(tmp_path / "vs"),
                       "CHECKPOINT_DB": str(tmp_path / "cp.sqlite"), "LLM_PROVIDER": "rule_based",
                       "UI_BACKEND": "local", "LOG_LEVEL": "WARNING"}.items():
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


def test_ui_investigation_flow(app):
    assert not app.exception
    _button(app, "Use example sales dataset").click().run()
    assert any("Dataset profile" in e.label for e in app.expander)
    app.text_area(key="question").input("Why did European revenue decrease in Q3?").run()
    _button(app, "Start Investigation").click().run()
    assert not app.exception
    report = "\n".join(m.value for m in app.markdown)
    assert "Executive Finding" in report and "Germany" in report
    assert any("Status: completed" in s.value for s in app.success)


def test_ui_approval_flow(app):
    _button(app, "Use example sales dataset").click().run()
    app.text_area(key="question").input("Delete rows with negative revenue").run()
    _button(app, "Start Investigation").click().run()
    assert any("ACTION REQUIRES APPROVAL" in w.value for w in app.warning)
    _button(app, "APPROVE").click().run()
    assert not app.exception
    assert any("Status: completed" in s.value for s in app.success)
