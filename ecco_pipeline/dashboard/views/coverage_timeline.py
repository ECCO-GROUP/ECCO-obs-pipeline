"""
Coverage Timeline view — first-to-last successfully-harvested granule date
per dataset, grouped by ECCO variable.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import solr_client

# Colour per ECCO variable; "unassigned" catches datasets with no ecco_variable yet.
VARIABLE_COLORS = {
    "sea_ice": "#4FC3F7",
    "sst": "#EF5350",
    "sss": "#66BB6A",
    "ssh": "#AB47BC",
    "mass": "#FFA726",
    "unassigned": "#90A4AE",
}


def render(datasets_df: pd.DataFrame):
    st.header("Coverage Timeline")
    st.caption(
        "Span of each dataset's successfully-harvested granules (first to last "
        "date). Interior gaps are shown per-dataset in the Dataset Inspector."
    )

    if datasets_df.empty:
        st.warning("No datasets found in Solr.")
        return

    with st.spinner("Querying Solr..."):
        coverage = solr_client.get_harvest_coverage()

    ds_meta = datasets_df[["dataset_s", "ecco_variable_s"]].copy()
    ds_meta["variable"] = ds_meta["ecco_variable_s"].fillna("").replace("", "unassigned")
    df = ds_meta[["dataset_s", "variable"]].merge(coverage, on="dataset_s", how="left")

    # ── ECCO variable filter ───────────────────────────────────────────────
    variables = sorted(df["variable"].unique())
    selected = st.multiselect("ECCO variable", variables, default=variables)
    df = df[df["variable"].isin(selected)]
    if df.empty:
        st.info("No datasets match the selected ECCO variables.")
        return

    has_data = df[df["start"].notna()].sort_values(["variable", "start"])
    no_data = df[df["start"].isna()].sort_values(["variable", "dataset_s"])

    # Row order: ECCO-variable blocks; within a block, datasets sorted by start
    # date, then any datasets with no successfully-harvested granules.
    order = []
    for var in sorted(selected):
        order += has_data[has_data["variable"] == var]["dataset_s"].tolist()
        order += no_data[no_data["variable"] == var]["dataset_s"].tolist()

    now = pd.Timestamp.now(tz="UTC")

    # Zero-width bars keep no-data datasets on the y-axis so they still get a
    # row (annotated "no data" below).
    plot_df = pd.concat(
        [has_data, no_data.assign(start=now, end=now, count=0)],
        ignore_index=True,
    )

    fig = px.timeline(
        plot_df,
        x_start="start",
        x_end="end",
        y="dataset_s",
        color="variable",
        color_discrete_map=VARIABLE_COLORS,
        hover_data={"count": True, "variable": False},
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)))

    # X-axis: earliest harvested date → today.
    if not has_data.empty:
        fig.update_xaxes(range=[has_data["start"].min(), now])

    # Faint "today" marker.
    fig.add_shape(
        type="line", x0=now, x1=now, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="grey", width=1, dash="dash"),
    )

    # "no data" labels for datasets with a dataset doc but no harvested granules.
    for ds in no_data["dataset_s"]:
        fig.add_annotation(
            x=0.0, xref="paper", y=ds, yref="y",
            text="no data", showarrow=False, xanchor="left",
            font=dict(color=VARIABLE_COLORS["unassigned"], size=10),
        )

    fig.update_layout(
        height=max(300, 22 * len(order) + 100),
        margin=dict(t=20, b=0, l=0, r=0),
        legend_title_text="ECCO variable",
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
