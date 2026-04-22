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
from dashboard.views import dataset_inspector, recent_activity

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
    page = st.radio("View", ["Recent Activity", "Dataset Inspector"])

    st.divider()
    if st.button("Refresh data"):
        st.cache_data.clear()

# ---------------------------------------------------------------------------
# Data loading (cached so switching views doesn't re-query)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # 5-minute cache
def load_datasets():
    return solr_client.get_datasets()


datasets_df = load_datasets()

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
if page == "Recent Activity":
    recent_activity.render()
elif page == "Dataset Inspector":
    dataset_inspector.render(datasets_df)
