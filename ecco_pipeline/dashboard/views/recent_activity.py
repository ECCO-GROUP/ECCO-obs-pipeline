"""
Recent Activity view — last 7 days across all datasets, no user input required.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import solr_client

DAYS = 7

SUCCESS_COLOR = "#4CAF50"
FAIL_COLOR = "#F44336"
NEUTRAL_COLOR = "#90CAF9"


def _activity_summary(
    granules: pd.DataFrame,
    transformations: pd.DataFrame,
    aggregations: pd.DataFrame,
) -> pd.DataFrame:
    """Roll up per-dataset counts for the fleet summary table."""
    datasets = set()
    for df in [granules, transformations, aggregations]:
        if not df.empty and "dataset_s" in df.columns:
            datasets.update(df["dataset_s"].unique())

    rows = []
    for ds in sorted(datasets):
        g = granules[granules["dataset_s"] == ds] if not granules.empty else pd.DataFrame()
        t = transformations[transformations["dataset_s"] == ds] if not transformations.empty else pd.DataFrame()
        a = aggregations[aggregations["dataset_s"] == ds] if not aggregations.empty else pd.DataFrame()

        g_total = len(g)
        g_fail = int((~g["harvest_success_b"]).sum()) if not g.empty else 0
        t_total = len(t)
        t_fail = int((~t["success_b"]).sum()) if not t.empty else 0
        a_total = len(a)
        a_fail = int((~a["aggregation_success_b"]).sum()) if not a.empty else 0

        rows.append({
            "Dataset": ds,
            "Granules": g_total,
            "Harvest failures": g_fail,
            "Transformations": t_total,
            "Transform failures": t_fail,
            "Aggregations": a_total,
            "Agg failures": a_fail,
        })

    return pd.DataFrame(rows)


def _failures_table(
    granules: pd.DataFrame,
    transformations: pd.DataFrame,
    aggregations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not granules.empty:
        failed = granules[~granules["harvest_success_b"]]
        for _, r in failed.iterrows():
            rows.append({
                "Stage": "Harvest",
                "Dataset": r.get("dataset_s", ""),
                "Date": str(r["date_dt"])[:10] if pd.notna(r.get("date_dt")) else "",
                "Detail": r.get("filename_s", ""),
                "Error": r.get("error_message_s", ""),
            })

    if not transformations.empty:
        failed = transformations[~transformations["success_b"]]
        for _, r in failed.iterrows():
            rows.append({
                "Stage": "Transform",
                "Dataset": r.get("dataset_s", ""),
                "Date": str(r["date_dt"])[:10] if pd.notna(r.get("date_dt")) else "",
                "Detail": f'{r.get("grid_name_s", "")} / {r.get("field_s", "")}',
                "Error": r.get("error_message_s", ""),
            })

    if not aggregations.empty:
        failed = aggregations[~aggregations["aggregation_success_b"]]
        for _, r in failed.iterrows():
            rows.append({
                "Stage": "Aggregate",
                "Dataset": r.get("dataset_s", ""),
                "Date": str(r.get("year_i", "")),
                "Detail": f'{r.get("grid_name_s", "")} / {r.get("field_s", "")}',
                "Error": r.get("error_message_s", ""),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Stage", "Dataset", "Date", "Detail", "Error"]
    )


def render():
    st.header(f"Recent Activity — Last {DAYS} Days")

    with st.spinner("Querying Solr..."):
        granules = solr_client.get_recent_granules(DAYS)
        transformations = solr_client.get_recent_transformations(DAYS)
        aggregations = solr_client.get_recent_aggregations(DAYS)

    # ── Top-level metrics ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    g_fail = int((~granules["harvest_success_b"]).sum()) if not granules.empty and "harvest_success_b" in granules.columns else 0
    t_fail = int((~transformations["success_b"]).sum()) if not transformations.empty and "success_b" in transformations.columns else 0
    a_fail = int((~aggregations["aggregation_success_b"]).sum()) if not aggregations.empty and "aggregation_success_b" in aggregations.columns else 0

    col1.metric("Granules harvested", len(granules), delta=f"-{g_fail} failed" if g_fail else None, delta_color="inverse")
    col2.metric("Transformations run", len(transformations), delta=f"-{t_fail} failed" if t_fail else None, delta_color="inverse")
    col3.metric("Aggregations run", len(aggregations), delta=f"-{a_fail} failed" if a_fail else None, delta_color="inverse")

    st.divider()

    # ── Fleet summary table ────────────────────────────────────────────────
    st.subheader("Activity by Dataset")
    summary = _activity_summary(granules, transformations, aggregations)
    if summary.empty:
        st.info("No activity in the last 7 days.")
    else:
        st.dataframe(
            summary.style.applymap(
                lambda v: f"color: {FAIL_COLOR}; font-weight: bold" if isinstance(v, int) and v > 0 else "",
                subset=["Harvest failures", "Transform failures", "Agg failures"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Failures detail ────────────────────────────────────────────────────
    failures = _failures_table(granules, transformations, aggregations)
    total_failures = len(failures)

    st.subheader(f"Failures ({total_failures})")
    if failures.empty:
        st.success("No failures in the last 7 days.")
    else:
        st.dataframe(failures, use_container_width=True, hide_index=True)
