"""
Simple username/password authentication for the Streamlit app.

This is intentionally lightweight (single admin user via bcrypt hash in
.env) rather than a full user-management system. It is enough to keep the
assistant private behind a login screen. Swap this module out for OAuth /
SSO / a real user database if you need multi-user support in production.
"""
from datetime import datetime, timedelta

import bcrypt
import streamlit as st

from config.settings import settings


def _check_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        # Malformed hash in .env
        return False


def _init_auth_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("login_time", None)


def _session_expired() -> bool:
    login_time = st.session_state.get("login_time")
    if login_time is None:
        return True
    elapsed = datetime.utcnow() - login_time
    return elapsed > timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)


def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["login_time"] = None


def is_authenticated() -> bool:
    _init_auth_state()
    if st.session_state["authenticated"] and _session_expired():
        logout()
        st.warning("Your session expired. Please log in again.")
    return st.session_state["authenticated"]


def render_login_form():
    """Renders a centered login form. Sets session state on success."""
    _init_auth_state()

    st.markdown(
        """
        <div style="text-align:center; margin-top: 2rem;">
            <h1>🏠 Real Estate AI Assistant</h1>
            <p style="color: gray;">Please sign in to continue</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            config_problems = settings.validate()
            if any("APP_PASSWORD_HASH" in p for p in config_problems):
                st.error(
                    "No password hash is configured on the server. "
                    "Run `python -m auth.generate_password_hash` and set APP_PASSWORD_HASH in .env."
                )
                return

            valid_user = username == settings.APP_USERNAME
            valid_pass = _check_password(password, settings.APP_PASSWORD_HASH)

            if valid_user and valid_pass:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["login_time"] = datetime.utcnow()
                st.rerun()
            else:
                st.error("Invalid username or password.")
