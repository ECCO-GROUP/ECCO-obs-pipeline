"""
ECCO Pipeline Dashboard
Run with:  streamlit run ecco_pipeline/dashboard/app.py
"""
import sys
from pathlib import Path

# Allow imports from ecco_pipeline root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from dashboard import solr_client
from dashboard.views import coverage_timeline, dataset_inspector, recent_activity

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ECCO Observation Pipeline Dashboard",
    page_icon="ecco_pipeline/dashboard/assets/favicon.ico",
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
# Data loading (cached so switching views doesn't re-query)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # 5-minute cache
def load_datasets():
    return solr_client.get_datasets()


@st.cache_data(ttl=300)
def load_total_counts():
    return solr_client.get_total_counts()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ECCO Observation Pipeline")

    totals = load_total_counts()
    r1c1, r1c2 = st.columns(2)
    r1c1.metric("Datasets", totals["datasets"])
    r1c2.metric("Granules", totals["granules"])
    r2c1, r2c2 = st.columns(2)
    r2c1.metric("Transformations", totals["transformations"])
    r2c2.metric("Aggregations", totals["aggregations"])

    st.divider()
    page = st.radio("View", ["Recent Activity", "Coverage Timeline", "Dataset Inspector"])

    st.divider()
    if st.button("Refresh data"):
        st.cache_data.clear()

    st.divider()
    st.caption(f"Solr: {SOLR_HOST}")
    st.caption(f"Collection: {SOLR_COLLECTION}")


datasets_df = load_datasets()

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
if page == "Recent Activity":
    recent_activity.render()
elif page == "Coverage Timeline":
    coverage_timeline.render(datasets_df)
elif page == "Dataset Inspector":
    dataset_inspector.render(datasets_df)
