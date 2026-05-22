"""
ECCO Pipeline Dashboard
Run with:  streamlit run ecco_pipeline/dashboard/app.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from ecco_pipeline root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from dashboard import solr_client
from dashboard.views import coverage_timeline, dataset_inspector, recent_activity

ICON_PATH = Path(__file__).parent / "assets" / "favicon.png"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ECCO Observation Pipeline Dashboard",
    page_icon=str(ICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Solr connection — read from conf/global_settings (no user-facing controls)
# ---------------------------------------------------------------------------
try:
    from conf.global_settings import SOLR_HOST, SOLR_COLLECTION
except ImportError:
    SOLR_HOST = "http://localhost:8983/solr/"
    SOLR_COLLECTION = "ecco_datasets"

solr_client.configure(SOLR_HOST, SOLR_COLLECTION)


# ---------------------------------------------------------------------------
# Data loading (cached so switching views doesn't re-query). Each cached
# function returns (value, loaded_at) so the sidebar can show data freshness.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # 5-minute cache
def load_datasets():
    return solr_client.get_datasets(), datetime.now(timezone.utc)


@st.cache_data(ttl=300)
def load_total_counts():
    return solr_client.get_total_counts(), datetime.now(timezone.utc)


@st.cache_data(ttl=60)  # short TTL so outages surface quickly
def check_solr_connection():
    return solr_client.ping()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
# Apply any pending programmatic view change (e.g. from a Recent Activity
# cross-link) before the radio widget instantiates — Streamlit rejects writes
# to a widget's session_state key once the widget has rendered.
if "_pending_view" in st.session_state:
    st.session_state["view"] = st.session_state.pop("_pending_view")

solr_error = check_solr_connection()

with st.sidebar:
    st.title("ECCO Observation Pipeline")

    if solr_error:
        st.error(f"Solr unreachable — {solr_error}")
    else:
        totals, totals_loaded_at = load_total_counts()
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Datasets", totals["datasets"])
        r1c2.metric("Granules", totals["granules"])
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Transformations", totals["transformations"])
        r2c2.metric("Aggregations", totals["aggregations"])
        st.caption(f"Updated {totals_loaded_at.strftime('%H:%M UTC')}")

    st.divider()
    page = st.radio(
        "View",
        ["Recent Activity", "Coverage Timeline", "Dataset Inspector"],
        key="view",
    )

    st.divider()
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(f"Solr: {SOLR_HOST}")
    st.caption(f"Collection: {SOLR_COLLECTION}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if solr_error:
    st.error(
        f"Can't reach Solr at `{SOLR_HOST}{SOLR_COLLECTION}` — {solr_error}. "
        "Views below will show empty data until the connection is restored."
    )

datasets_df, _ = load_datasets()

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
if page == "Recent Activity":
    recent_activity.render()
elif page == "Coverage Timeline":
    coverage_timeline.render(datasets_df)
elif page == "Dataset Inspector":
    dataset_inspector.render(datasets_df)
