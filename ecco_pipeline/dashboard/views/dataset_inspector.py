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

def _harvest_panel(ds_name: str, compact: bool = False):
    if compact:
        st.markdown("**Harvest**")
    else:
        st.subheader("Harvest")
    granules = solr_client.get_granules(ds_name)

    if granules.empty:
        st.info("No granule records found.")
        return

    total = len(granules)
    success = int(granules["harvest_success_b"].sum())
    failed = total - success

    successful = granules[granules["harvest_success_b"]]
    if not successful.empty and successful["date_dt"].notna().any():
        latest_date = str(successful["date_dt"].max().date())
    else:
        latest_date = "—"

    if compact:
        c1, c2 = st.columns(2)
        c1.metric("Granules", total)
        c2.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")
        st.caption(f"Latest: {latest_date}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total granules", total)
        c2.metric("Successful", success)
        c3.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")
        c4.metric("Latest granule date", latest_date)

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
        bar_kwargs = dict(
            x="date",
            y="count",
            color="status",
            color_discrete_map=color_map,
            labels={"date": "Date", "count": "Granules", "status": ""},
        )
        if not compact:
            bar_kwargs["title"] = "Granules by Date"
        fig = px.bar(daily, **bar_kwargs)
        fig.update_layout(
            margin=dict(t=10 if compact else 40, b=0),
            height=180 if compact else 250,
            legend_title_text="",
            showlegend=not compact,
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    if compact:
        return

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

def _transformation_panel(ds_name: str, compact: bool = False):
    if compact:
        st.markdown("**Transformation**")
    else:
        st.subheader("Transformation")
    tx = solr_client.get_transformations(ds_name)

    if tx.empty:
        st.info("No transformation records found.")
        return

    total = len(tx)
    success = int(tx["success_b"].sum())
    failed = total - success

    if compact:
        c1, c2 = st.columns(2)
        c1.metric("Runs", total)
        c2.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")
        rate = (success / total * 100) if total else 0
        st.caption(f"Success rate: {rate:.0f}%")
    else:
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

        heatmap_kwargs = dict(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale=[[0, FAIL_COLOR], [0.5, "#FFF176"], [1, SUCCESS_COLOR]],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}%" for v in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            showscale=not compact,
        )
        if not compact:
            heatmap_kwargs["colorbar"] = dict(title="Success %")
        fig = go.Figure(go.Heatmap(**heatmap_kwargs))
        layout_kwargs = dict(
            margin=dict(t=10 if compact else 40, b=0),
            xaxis_title="" if compact else "Field",
            yaxis_title="" if compact else "Grid",
        )
        if compact:
            layout_kwargs["height"] = 180
        else:
            layout_kwargs["height"] = max(200, 60 * len(heatmap_data))
            layout_kwargs["title"] = "Success Rate by Grid × Field"
        fig.update_layout(**layout_kwargs)
        st.plotly_chart(fig, use_container_width=True)

    if compact:
        return

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

    # Transformations that succeeded but flagged a data-quality issue (e.g. a field
    # missing from the source data, producing an empty record).
    has_msg = tx["error_message_s"].fillna("").astype(str).str.strip().ne("")
    warnings = tx[tx["success_b"] & has_msg][
        ["date_dt", "grid_name_s", "field_s", "error_message_s"]
    ].rename(columns={
        "date_dt": "Date", "grid_name_s": "Grid",
        "field_s": "Field", "error_message_s": "Message",
    })
    warnings["Date"] = warnings["Date"].dt.date

    if not warnings.empty:
        with st.expander(f"Warnings ({len(warnings)})", expanded=False):
            st.dataframe(warnings, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Aggregation panel
# ---------------------------------------------------------------------------

def _aggregation_panel(ds_name: str, compact: bool = False):
    if compact:
        st.markdown("**Aggregation**")
    else:
        st.subheader("Aggregation")
    agg = solr_client.get_aggregations(ds_name)

    if agg.empty:
        st.info("No aggregation records found.")
        return

    total = len(agg)
    success = int(agg["aggregation_success_b"].sum())
    failed = total - success

    if compact:
        c1, c2 = st.columns(2)
        c1.metric("Runs", total)
        c2.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")
        last_run = pd.to_datetime(agg.get("aggregation_time_dt"), errors="coerce", utc=True).max()
        last_str = "—" if pd.isna(last_run) else str(last_run)[:10]
        st.caption(f"Latest: {last_str}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Successful", success)
        c3.metric("Failed", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")

    # Coverage strip: one row per (grid, field), one cell per year.
    needed = {"grid_name_s", "field_s", "year_i"}
    if needed.issubset(agg.columns):
        valid = agg.dropna(subset=["year_i"]).copy()
        valid["year_i"] = valid["year_i"].astype(int)
        valid["gf"] = valid["grid_name_s"].astype(str) + " / " + valid["field_s"].astype(str)

        if not valid.empty:
            all_years = list(range(int(valid["year_i"].min()), int(valid["year_i"].max()) + 1))
            gfs = sorted(valid["gf"].unique())

            z, hover = [], []
            for gf in gfs:
                # Last-write-wins if multiple records exist for the same year.
                status_map = dict(
                    zip(
                        valid[valid["gf"] == gf]["year_i"],
                        valid[valid["gf"] == gf]["aggregation_success_b"].astype(bool),
                    )
                )
                z_row, hover_row = [], []
                for y in all_years:
                    if y in status_map:
                        z_row.append(1.0 if status_map[y] else 0.0)
                        hover_row.append("success" if status_map[y] else "failed")
                    else:
                        z_row.append(0.5)
                        hover_row.append("no aggregation")
                z.append(z_row)
                hover.append(hover_row)

            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=all_years,
                    y=gfs,
                    customdata=hover,
                    colorscale=[
                        [0.0, FAIL_COLOR], [0.49, FAIL_COLOR],
                        [0.5, MISSING_COLOR], [0.51, MISSING_COLOR],
                        [0.52, SUCCESS_COLOR], [1.0, SUCCESS_COLOR],
                    ],
                    showscale=False,
                    zmin=0, zmax=1,
                    xgap=1, ygap=4,
                    hovertemplate="%{y}<br>%{x}: %{customdata}<extra></extra>",
                )
            )
            agg_layout_kwargs = dict(
                margin=dict(t=10 if compact else 40, b=0),
                xaxis_title="" if compact else "Year",
                yaxis_title="",
            )
            if compact:
                agg_layout_kwargs["height"] = 180
            else:
                agg_layout_kwargs["height"] = max(180, 36 * len(gfs) + 60)
                agg_layout_kwargs["title"] = "Coverage by Grid / Field"
            fig.update_layout(**agg_layout_kwargs)
            fig.update_xaxes(dtick=1 if len(all_years) <= 20 else 5)
            st.plotly_chart(fig, use_container_width=True)
            if not compact:
                st.caption(
                    f"Green = success · red = failed · grey = no aggregation record for that year"
                )

            if not compact:
                # Per (grid, field) summary
                summary_rows = []
                for gf in gfs:
                    sub = valid[valid["gf"] == gf]
                    last_run = pd.to_datetime(sub.get("aggregation_time_dt"), errors="coerce", utc=True).max()
                    summary_rows.append({
                        "Grid / Field": gf,
                        "Span": f"{int(sub['year_i'].min())} – {int(sub['year_i'].max())}",
                        "Years": int(sub["year_i"].nunique()),
                        "Successful": int(sub["aggregation_success_b"].sum()),
                        "Failed": int((~sub["aggregation_success_b"]).sum()),
                        "Last run": "—" if pd.isna(last_run) else str(last_run)[:19],
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if compact:
        return

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
    sel_col, mode_col = st.columns([3, 1])
    with sel_col:
        selected = st.selectbox("Select dataset", ds_names, key="dataset_selector")
    with mode_col:
        compact = st.toggle("Compact view", value=True, key="inspector_compact")

    if not selected:
        return

    # Dataset-level status strip
    row = datasets_df[datasets_df["dataset_s"] == selected]
    if not row.empty:
        r = row.iloc[0]

        def _fmt_str(val) -> str:
            # `val or '—'` is wrong: NaN is truthy in Python.
            if pd.isna(val):
                return "—"
            s = str(val).strip()
            return s if s else "—"

        def _fmt_dt(val) -> str:
            return "—" if pd.isna(val) else str(val)[:19]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Harvest status:** {_fmt_str(r.get('harvest_status_s'))}")
        c2.markdown(f"**Transform status:** {_fmt_str(r.get('transformation_status_s'))}")
        c3.markdown(f"**Aggregation status:** {_fmt_str(r.get('aggregation_status_s'))}")

        st.caption(
            f"Last harvest: {_fmt_dt(r.get('last_checked_dt'))}  |  "
            f"Last transform: {_fmt_dt(r.get('last_transformation_dt'))}  |  "
            f"Last aggregation: {_fmt_dt(r.get('last_aggregation_dt'))}"
        )
        st.divider()

    if compact:
        col_h, col_t, col_a = st.columns(3)
        with col_h:
            _harvest_panel(selected, compact=True)
        with col_t:
            _transformation_panel(selected, compact=True)
        with col_a:
            _aggregation_panel(selected, compact=True)
    else:
        _harvest_panel(selected)
        st.divider()
        _transformation_panel(selected)
        st.divider()
        _aggregation_panel(selected)
