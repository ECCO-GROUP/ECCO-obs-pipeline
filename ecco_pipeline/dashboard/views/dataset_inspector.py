"""
Dataset Inspector view — select a dataset, all panels auto-populate.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import solr_client

SUCCESS_COLOR = "#4CAF50"
FAIL_COLOR = "#F44336"
MISSING_COLOR = "#B0BEC5"


# ---------------------------------------------------------------------------
# Harvest panel
# ---------------------------------------------------------------------------

def _harvest_panel(ds_name: str):
    st.subheader("Harvest")
    granules = solr_client.get_granules(ds_name)

    if granules.empty:
        st.info("No granule records found.")
        return

    total = len(granules)
    success = int(granules["harvest_success_b"].sum())
    failed = total - success

    c1, c2, c3 = st.columns(3)
    c1.metric("Total granules", total)
    c2.metric("Successful", success)
    c3.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")

    # Timeline: one bar per date coloured by outcome
    granules["date"] = granules["date_dt"].dt.date
    daily = (
        granules.groupby(["date", "harvest_success_b"])
        .size()
        .reset_index(name="count")
    )
    daily["status"] = daily["harvest_success_b"].map({True: "Success", False: "Failed"})
    color_map = {"Success": SUCCESS_COLOR, "Failed": FAIL_COLOR}

    if not daily.empty:
        fig = px.bar(
            daily,
            x="date",
            y="count",
            color="status",
            color_discrete_map=color_map,
            labels={"date": "Date", "count": "Granules", "status": ""},
            title="Granules by Date",
        )
        fig.update_layout(margin=dict(t=40, b=0), height=250, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # Failed granules table
    failures = granules[~granules["harvest_success_b"]][
        ["date_dt", "filename_s", "error_message_s"]
    ].rename(columns={"date_dt": "Date", "filename_s": "File", "error_message_s": "Error"})
    failures["Date"] = failures["Date"].dt.date

    if not failures.empty:
        with st.expander(f"Failed granules ({len(failures)})", expanded=True):
            st.dataframe(failures, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Transformation panel
# ---------------------------------------------------------------------------

def _transformation_panel(ds_name: str):
    st.subheader("Transformation")
    tx = solr_client.get_transformations(ds_name)

    if tx.empty:
        st.info("No transformation records found.")
        return

    total = len(tx)
    success = int(tx["success_b"].sum())
    failed = total - success

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Successful", success)
    c3.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")

    # Per-grid × per-field success rate heatmap
    if "grid_name_s" in tx.columns and "field_s" in tx.columns:
        pivot = (
            tx.groupby(["grid_name_s", "field_s"])
            .agg(total=("success_b", "count"), success=("success_b", "sum"))
            .reset_index()
        )
        pivot["success_rate"] = (pivot["success"] / pivot["total"] * 100).round(1)
        heatmap_data = pivot.pivot(index="grid_name_s", columns="field_s", values="success_rate")

        fig = go.Figure(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale=[[0, FAIL_COLOR], [0.5, "#FFF176"], [1, SUCCESS_COLOR]],
                zmin=0,
                zmax=100,
                text=[[f"{v:.0f}%" for v in row] for row in heatmap_data.values],
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="Success %"),
            )
        )
        fig.update_layout(
            title="Success Rate by Grid × Field",
            margin=dict(t=40, b=0),
            height=max(200, 60 * len(heatmap_data)),
            xaxis_title="Field",
            yaxis_title="Grid",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Failed transformations
    failures = tx[~tx["success_b"]][
        ["date_dt", "grid_name_s", "field_s", "error_message_s"]
    ].rename(columns={
        "date_dt": "Date", "grid_name_s": "Grid",
        "field_s": "Field", "error_message_s": "Error",
    })
    failures["Date"] = failures["Date"].dt.date

    if not failures.empty:
        with st.expander(f"Failed transformations ({len(failures)})", expanded=True):
            st.dataframe(failures, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Aggregation panel
# ---------------------------------------------------------------------------

def _aggregation_panel(ds_name: str):
    st.subheader("Aggregation")
    agg = solr_client.get_aggregations(ds_name)

    if agg.empty:
        st.info("No aggregation records found.")
        return

    total = len(agg)
    success = int(agg["aggregation_success_b"].sum())
    failed = total - success

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Successful", success)
    c3.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")

    # Year × grid matrix — one cell per combination, coloured success/fail
    if "grid_name_s" in agg.columns and "year_i" in agg.columns:
        # Build status string for colour mapping
        agg["status"] = agg["aggregation_success_b"].map({True: "Success", False: "Failed"})

        grids = sorted(agg["grid_name_s"].unique())
        years = sorted(agg["year_i"].dropna().unique(), reverse=True)

        # Pivot: rows=year (descending), cols=grid, value=status
        matrix = []
        annotations = []
        for yi, year in enumerate(years):
            row = []
            for xi, grid in enumerate(grids):
                cell = agg[(agg["year_i"] == year) & (agg["grid_name_s"] == grid)]
                if cell.empty:
                    row.append(0.5)  # missing — grey
                    annotations.append(dict(x=xi, y=yi, text="—", showarrow=False,
                                            font=dict(color="white")))
                elif cell.iloc[0]["aggregation_success_b"]:
                    row.append(1.0)
                    annotations.append(dict(x=xi, y=yi, text="✓", showarrow=False,
                                            font=dict(color="white")))
                else:
                    row.append(0.0)
                    annotations.append(dict(x=xi, y=yi, text="✗", showarrow=False,
                                            font=dict(color="white")))
            matrix.append(row)

        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=grids,
                y=[str(int(y)) for y in years],
                colorscale=[
                    [0.0, FAIL_COLOR],
                    [0.49, FAIL_COLOR],
                    [0.5, MISSING_COLOR],
                    [0.51, MISSING_COLOR],
                    [0.52, SUCCESS_COLOR],
                    [1.0, SUCCESS_COLOR],
                ],
                showscale=False,
                zmin=0,
                zmax=1,
            )
        )
        fig.update_layout(
            title="Aggregation Status by Year × Grid",
            annotations=annotations,
            margin=dict(t=40, b=0),
            height=max(200, 30 * len(years) + 80),
            xaxis_title="Grid",
            yaxis_title="Year",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Failed aggregations
    failures = agg[~agg["aggregation_success_b"]][
        ["year_i", "grid_name_s", "field_s", "error_message_s"]
    ].rename(columns={
        "year_i": "Year", "grid_name_s": "Grid",
        "field_s": "Field", "error_message_s": "Error",
    })

    if not failures.empty:
        with st.expander(f"Failed aggregations ({len(failures)})", expanded=True):
            st.dataframe(failures, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render(datasets_df: pd.DataFrame):
    st.header("Dataset Inspector")

    if datasets_df.empty:
        st.warning("No datasets found in Solr.")
        return

    ds_names = sorted(datasets_df["dataset_s"].dropna().unique().tolist())
    selected = st.selectbox("Select dataset", ds_names, key="dataset_selector")

    if not selected:
        return

    # Dataset-level status strip
    row = datasets_df[datasets_df["dataset_s"] == selected]
    if not row.empty:
        r = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Harvest status:** {r.get('harvest_status_s') or '—'}")
        c2.markdown(f"**Transform status:** {r.get('transformation_status_s') or '—'}")
        c3.markdown(f"**Aggregation status:** {r.get('aggregation_status_s') or '—'}")
        def _fmt_dt(val) -> str:
            return "—" if pd.isna(val) else str(val)[:19]

        st.caption(
            f"Last harvest: {_fmt_dt(r.get('last_checked_dt'))}  |  "
            f"Last transform: {_fmt_dt(r.get('last_transformation_dt'))}  |  "
            f"Last aggregation: {_fmt_dt(r.get('last_aggregation_dt'))}"
        )
        st.divider()

    _harvest_panel(selected)
    st.divider()
    _transformation_panel(selected)
    st.divider()
    _aggregation_panel(selected)
