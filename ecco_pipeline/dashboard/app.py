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
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Solr connection — reads from conf/global_settings if available, else sidebar
# ---------------------------------------------------------------------------
try:
    from conf.global_settings import SOLR_HOST, SOLR_COLLECTION
except ImportError:
    SOLR_HOST = None
    SOLR_COLLECTION = None

with st.sidebar:
    st.title("ECCO Pipeline")
    st.divider()

    host = st.text_input("Solr host", value=SOLR_HOST or "http://localhost:8983/solr/")
    collection = st.text_input("Collection", value=SOLR_COLLECTION or "ecco_datasets")
    solr_client.configure(host, collection)

    st.divider()
    page = st.radio("View", ["Recent Activity", "Coverage Timeline", "Dataset Inspector"])

    st.divider()
    if st.button("Refresh data"):
        st.cache_data.clear()

# ---------------------------------------------------------------------------
# Data loading (cached so switching views doesn't re-query)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # 5-minute cache
def load_datasets():
    return solr_client.get_datasets()


@st.cache_data(ttl=300)
def load_total_counts():
    return solr_client.get_total_counts()


datasets_df = load_datasets()

# ---------------------------------------------------------------------------
# Global state strip — shown above every view
# ---------------------------------------------------------------------------
totals = load_total_counts()
t1, t2, t3, t4 = st.columns(4)
t1.metric("Datasets", totals["datasets"])
t2.metric("Granules", totals["granules"])
t3.metric("Transformations", totals["transformations"])
t4.metric("Aggregations", totals["aggregations"])
st.divider()

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
if page == "Recent Activity":
    recent_activity.render()
elif page == "Coverage Timeline":
    coverage_timeline.render(datasets_df)
elif page == "Dataset Inspector":
    dataset_inspector.render(datasets_df)
