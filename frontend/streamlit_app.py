"""InsightFlow AI - Streamlit entry point (also the main file for Streamlit Community Cloud).

Run:  python -m app.run        (or: streamlit run frontend/streamlit_app.py)
Pages: Investigate (coupon-gated) and Admin (password-protected).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Project root (for `app`) and this folder (for `ui_core`), whatever directory Streamlit starts in.
FRONTEND = Path(__file__).resolve().parent
for path in (FRONTEND.parent, FRONTEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_core import inject_css  # noqa: E402

st.set_page_config(page_title="InsightFlow AI", page_icon="📊", layout="wide")
inject_css()
page = st.navigation([
    st.Page("views/investigate.py", title="Investigate", icon="🔎", default=True),
    st.Page("views/admin.py", title="Admin", icon="🛠️"),
])
page.run()
