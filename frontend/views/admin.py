"""Admin dashboard: project ON/OFF switch, coupons, AI/Groq switch, activity."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.access import mask
from app.config import get_settings
from app.llm.client import LLMError, ping
from ui_core import access_store, ai_label, backend, hero, pill


def login() -> bool:
    store = access_store()
    if st.session_state.get("is_admin"):
        return True
    hero("Admin", "Manage access, coupons and the AI engine.", small=True)
    if not store.admin_configured:
        st.warning("The admin page is disabled. Set `ADMIN_PASSWORD` in `.env` (local) or in the app's "
                   "Secrets (Streamlit Cloud) and restart the app.")
        return False
    _, mid, _ = st.columns([1, 2, 1])
    with mid, st.form("admin_login"):
        password = st.text_input("Admin password", type="password")
        if st.form_submit_button("Sign in", type="primary", use_container_width=True):
            if store.check_admin_password(password):
                st.session_state.is_admin = True
                st.rerun()
            st.error("Wrong password.")
    return False


def overview(store, be) -> None:
    st.markdown("#### Project")
    enabled = st.toggle("InsightFlow AI is ON for visitors", value=store.project_enabled, key="project_switch")
    if enabled != store.project_enabled:
        store.set_project_enabled(enabled)
        st.rerun()
    st.caption("When OFF, visitors see a 'switched off' message and cannot run anything. You can still use it as admin.")
    coupons = store.coupons()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "ON" if store.project_enabled else "OFF")
    c2.metric("Active codes", sum(1 for c in coupons if c.problem() is None))
    c3.metric("Code unlocks", sum(c.uses for c in coupons))
    c4.metric("Investigations", sum(c.investigations for c in coupons))
    st.markdown(pill(ai_label(be), "green" if "offline" not in ai_label(be) else "amber") +
                pill("coupon required" if store.coupon_required else "open access (REQUIRE_COUPON=false)"),
                unsafe_allow_html=True)


def coupons_tab(store) -> None:
    st.markdown("#### Access codes")
    coupons = store.coupons()
    if coupons:
        rows = [{"code": c.code, "status": "✅ usable" if c.problem() is None else f"⛔ {c.problem()}",
                 "uses": f"{c.uses}/{c.max_uses}" if c.max_uses else f"{c.uses}/∞",
                 "investigations": c.investigations, "expires": c.expires or "never"} for c in coupons]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        code = c1.selectbox("Code", [c.code for c in coupons], label_visibility="collapsed")
        if c2.button("Activate", use_container_width=True):
            store.set_coupon_active(code, True)
            st.rerun()
        if c3.button("Deactivate", use_container_width=True):
            store.set_coupon_active(code, False)
            st.rerun()
        if c4.button("Delete", use_container_width=True):
            store.delete_coupon(code)
            st.rerun()
    else:
        st.info("No codes yet. Visitors cannot unlock the app until you create one.")
    with st.form("new_coupon", clear_on_submit=True):
        st.markdown("**Create a code**")
        c1, c2, c3 = st.columns(3)
        new_code = c1.text_input("Code", placeholder="PYTHON2026")
        limit = c2.number_input("Max uses (0 = unlimited)", min_value=0, value=0, step=1)
        expires = c3.date_input("Expires", value=None, min_value=date.today())
        if st.form_submit_button("Create code", type="primary"):
            try:
                store.add_coupon(new_code, int(limit) or None, expires)
                st.success(f"Created {new_code.strip().upper()}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.caption("Codes created here are saved on the server's disk. On free hosting that disk resets when the app "
               "restarts; the codes in the `COUPONS` secret always come back.")


def ai_tab(store, be) -> None:
    st.markdown("#### AI engine (Groq)")
    if not getattr(be, "local", False):
        st.info("The AI switch is available when the UI runs the service itself (UI_BACKEND=local).")
        return
    settings = get_settings()
    st.markdown(pill(ai_label(be), "green" if be.svc.deps.llm else "amber"), unsafe_allow_html=True)
    configured = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    st.caption(f"Key from Secrets/.env: {mask(configured)} · key set here: {mask(store.groq_key_override)}")
    enabled = st.toggle("Use AI (off = offline rule-based reasoner, no tokens used)", value=store.ai_enabled)
    new_key = st.text_input("Replace Groq API key (optional)", type="password", placeholder="gsk_...")
    c1, c2, c3 = st.columns(3)
    if c1.button("Save AI settings", type="primary", use_container_width=True):
        store.set_ai(enabled, groq_key=new_key or None)
        used = be.svc.apply_ai_settings(store.ai_enabled, store.groq_key_override)
        st.success(f"Now using {used}")
        st.rerun()
    if c2.button("Remove key set here", use_container_width=True, disabled=not store.groq_key_override):
        store.set_ai(store.ai_enabled, clear_key=True)
        be.svc.apply_ai_settings(store.ai_enabled, None)
        st.rerun()
    if c3.button("Test connection", use_container_width=True, disabled=be.svc.deps.llm is None):
        try:
            st.success(ping(be.svc.deps.llm))
        except LLMError as e:
            st.error(f"Test failed: {e}")


def activity_tab(store, be) -> None:
    st.markdown("#### Admin activity")
    entries = store.audit(50)
    if entries:
        st.dataframe(pd.DataFrame([{"when (UTC)": e.at.strftime("%Y-%m-%d %H:%M"), "event": e.event} for e in entries]),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No activity yet.")
    if getattr(be, "local", False):
        st.markdown("#### Recent investigations")
        traces = be.svc.traces.list(20)
        if traces:
            st.dataframe(pd.DataFrame(traces), use_container_width=True, hide_index=True)
        else:
            st.caption("No investigations yet.")


def main() -> None:
    if not login():
        return
    store, be = access_store(), backend()
    hero("Admin dashboard", "Switch the project on or off, manage access codes and the AI engine.", small=True)
    if st.sidebar.button("Sign out of admin", use_container_width=True):
        st.session_state.pop("is_admin", None)
        st.rerun()
    tabs = st.tabs(["🏠 Overview", "🎟️ Coupons", "🤖 AI engine", "📜 Activity"])
    with tabs[0]:
        overview(store, be)
    with tabs[1]:
        coupons_tab(store)
    with tabs[2]:
        ai_tab(store, be)
    with tabs[3]:
        activity_tab(store, be)


main()
