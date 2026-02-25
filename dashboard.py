import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Solana Orderflow Dashboard", layout="wide")

st.title("Solana Orderflow Dashboard")

# Custom CSS to make tabs larger
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 PropAMM Metrics", "📈 Intraday Block Position", "🔀 Orderflow Sankey"])

# Helper function to format volume as $XXM or $XXB
def format_volume(val):
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    elif val >= 1e6:
        return f"${val/1e6:.1f}M"
    elif val >= 1e3:
        return f"${val/1e3:.1f}K"
    else:
        return f"${val:.0f}"

# ===================
# TAB 1: PropAMM Metrics
# ===================
with tab1:
    @st.cache_data
    def load_data():
        df = pd.read_csv("propamm_allium_initial_sample.csv")
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE")
        return df

    df = load_data()

    # Get unique projects for color consistency
    projects = sorted(df["PROJECT"].unique())
    color_map = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    project_colors = {proj: color_map[i % len(color_map)] for i, proj in enumerate(projects)}

    # Pivot data for stacked bar chart
    volume_pivot = df.pivot_table(
        index="DATE",
        columns="PROJECT",
        values="VOLUME_USD",
        aggfunc="sum"
    ).fillna(0)

    # Stacked Bar Chart - Volume
    st.header("Volume (USD) by Project")
    fig_volume = go.Figure()
    for project in projects:
        if project in volume_pivot.columns:
            fig_volume.add_trace(go.Bar(
                name=project,
                x=volume_pivot.index,
                y=volume_pivot[project],
                marker_color=project_colors[project]
            ))

    fig_volume.update_layout(
        barmode="stack",
        xaxis_title="Date",
        yaxis_title="Volume (USD)",
        legend_title="Project",
        height=500
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    # Line Chart - TX Count
    st.header("Transaction Count by Project")
    fig_tx = go.Figure()
    for project in projects:
        project_data = df[df["PROJECT"] == project].sort_values("DATE")
        fig_tx.add_trace(go.Scatter(
            name=project,
            x=project_data["DATE"],
            y=project_data["TX_COUNT"],
            mode="lines+markers",
            line=dict(color=project_colors[project])
        ))

    fig_tx.update_layout(
        xaxis_title="Date",
        yaxis_title="Transaction Count",
        legend_title="Project",
        height=500
    )
    st.plotly_chart(fig_tx, use_container_width=True)

    # Line Chart - Fill Count
    st.header("Fill Count by Project")
    fig_fill = go.Figure()
    for project in projects:
        project_data = df[df["PROJECT"] == project].sort_values("DATE")
        fig_fill.add_trace(go.Scatter(
            name=project,
            x=project_data["DATE"],
            y=project_data["FILL_COUNT"],
            mode="lines+markers",
            line=dict(color=project_colors[project])
        ))

    fig_fill.update_layout(
        xaxis_title="Date",
        yaxis_title="Fill Count",
        legend_title="Project",
        height=500
    )
    st.plotly_chart(fig_fill, use_container_width=True)

    # Summary stats
    st.header("Summary Statistics")
    summary = df.groupby("PROJECT").agg({
        "VOLUME_USD": "sum",
        "TX_COUNT": "sum",
        "FILL_COUNT": "sum"
    }).round(2)
    summary.columns = ["Total Volume (USD)", "Total TX Count", "Total Fill Count"]
    st.dataframe(summary.style.format({
        "Total Volume (USD)": "${:,.2f}",
        "Total TX Count": "{:,.0f}",
        "Total Fill Count": "{:,.0f}"
    }), use_container_width=True)

# ===================
# TAB 2: Intraday Block Position Analysis
# ===================
with tab2:
    st.header("Intraday Block Position Analysis (7d)")

    @st.cache_data
    def load_intraday_data():
        df_intraday = pd.read_csv("propamm_intraday_block_position_7d.csv")
        df_intraday = df_intraday.sort_values(["PROJECT", "POSITION_BUCKET"])

        # Normalize metrics per project (as % of project's total)
        for col in ["FILL_COUNT", "VOLUME_USD"]:
            df_intraday[f"{col}_PCT"] = df_intraday.groupby("PROJECT")[col].transform(lambda x: x / x.sum() * 100)

        # For avg fill, normalize relative to project's mean avg fill
        df_intraday["AVG_FILL_USD_PCT"] = df_intraday.groupby("PROJECT")["AVG_FILL_USD"].transform(lambda x: x / x.mean() * 100)

        return df_intraday

    df_intraday = load_intraday_data()

    # Get unique projects for intraday data
    intraday_projects = sorted(df_intraday["PROJECT"].unique())
    intraday_color_map = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    intraday_colors = {proj: intraday_color_map[i % len(intraday_color_map)] for i, proj in enumerate(intraday_projects)}

    # Fill Count by Position Bucket (Normalized)
    st.subheader("Fill Count % by Block Position")
    fig_intraday_fills = go.Figure()
    for project in intraday_projects:
        project_data = df_intraday[df_intraday["PROJECT"] == project].sort_values("POSITION_BUCKET")
        fig_intraday_fills.add_trace(go.Scatter(
            name=project,
            x=project_data["POSITION_BUCKET"],
            y=project_data["FILL_COUNT_PCT"],
            mode="lines+markers",
            line=dict(color=intraday_colors[project])
        ))

    fig_intraday_fills.update_layout(
        xaxis_title="Position in Block (0 = start, 1 = end)",
        yaxis_title="% of Project's Total Fills",
        legend_title="Project",
        height=500
    )
    st.plotly_chart(fig_intraday_fills, use_container_width=True)

    # Total Volume by Position Bucket (Normalized)
    st.subheader("Volume % by Block Position")
    fig_intraday_volume = go.Figure()
    for project in intraday_projects:
        project_data = df_intraday[df_intraday["PROJECT"] == project].sort_values("POSITION_BUCKET")
        fig_intraday_volume.add_trace(go.Scatter(
            name=project,
            x=project_data["POSITION_BUCKET"],
            y=project_data["VOLUME_USD_PCT"],
            mode="lines+markers",
            line=dict(color=intraday_colors[project])
        ))

    fig_intraday_volume.update_layout(
        xaxis_title="Position in Block (0 = start, 1 = end)",
        yaxis_title="% of Project's Total Volume",
        legend_title="Project",
        height=500
    )
    st.plotly_chart(fig_intraday_volume, use_container_width=True)

    # Avg Fill Size by Position Bucket (Normalized)
    st.subheader("Avg Fill Size % by Block Position")
    st.caption("Normalized to each project's mean avg fill size (100% = project's average)")
    fig_intraday_avg = go.Figure()
    for project in intraday_projects:
        project_data = df_intraday[df_intraday["PROJECT"] == project].sort_values("POSITION_BUCKET")
        fig_intraday_avg.add_trace(go.Scatter(
            name=project,
            x=project_data["POSITION_BUCKET"],
            y=project_data["AVG_FILL_USD_PCT"],
            mode="lines+markers",
            line=dict(color=intraday_colors[project])
        ))

    fig_intraday_avg.update_layout(
        xaxis_title="Position in Block (0 = start, 1 = end)",
        yaxis_title="% of Project's Mean Avg Fill",
        legend_title="Project",
        height=500
    )
    st.plotly_chart(fig_intraday_avg, use_container_width=True)

# ===================
# TAB 3: Orderflow Sankey
# ===================
with tab3:
    st.header("Orderflow Sankey (7d)")

    @st.cache_data
    def load_sankey_data():
        df_sankey = pd.read_csv("orderflow_sankey_7d_sample.csv")

        # Group unlabeled validators (public key addresses) into one category
        def is_labeled(name):
            if len(name) > 30 and name.isalnum():
                return False
            if len(name) > 30 and all(c.isalnum() for c in name):
                return False
            return True

        df_sankey["VALIDATOR_DISPLAY"] = df_sankey["VALIDATOR"].apply(
            lambda x: x if is_labeled(x) else "Unlabeled Validators"
        )
        return df_sankey

    df_sankey_raw = load_sankey_data()

    # PropAMM DEXes (separate from regular DEX AMMs)
    propamm_dexes = {'humidifi', 'bisonfi', 'solfi', 'goonfi', 'tesserav', 'alphaq', 'aquifer', 'zerofi', 'lifinity'}

    # Filters
    st.subheader("Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    all_frontends = sorted(df_sankey_raw["FRONTEND"].unique())
    all_dexes_list = sorted(df_sankey_raw["DEX"].unique())
    all_validators = sorted(df_sankey_raw["VALIDATOR_DISPLAY"].unique())

    with filter_col1:
        selected_frontends = st.multiselect(
            "Frontend",
            options=all_frontends,
            default=all_frontends,
            key="frontend_filter"
        )

    with filter_col2:
        selected_dexes = st.multiselect(
            "DEX",
            options=all_dexes_list,
            default=all_dexes_list,
            key="dex_filter"
        )

    with filter_col3:
        selected_validators = st.multiselect(
            "Validator",
            options=all_validators,
            default=all_validators,
            key="validator_filter"
        )

    # Apply filters
    df_sankey = df_sankey_raw[
        (df_sankey_raw["FRONTEND"].isin(selected_frontends)) &
        (df_sankey_raw["DEX"].isin(selected_dexes)) &
        (df_sankey_raw["VALIDATOR_DISPLAY"].isin(selected_validators))
    ]

    # Aggregate Frontend -> DEX (sum across all validators)
    frontend_to_dex = df_sankey.groupby(["FRONTEND", "DEX"])["VOLUME_USD"].sum().reset_index()

    # Aggregate DEX -> Validator (sum across all frontends) - use display name
    dex_to_validator = df_sankey.groupby(["DEX", "VALIDATOR_DISPLAY"])["VOLUME_USD"].sum().reset_index()
    dex_to_validator.columns = ["DEX", "VALIDATOR", "VOLUME_USD"]

    # Check if data exists after filtering
    if df_sankey.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        st.stop()

    # Build node list: Frontends, DEXes (PropAMM first, then others), Validators
    frontends = sorted(df_sankey["FRONTEND"].unique())
    filtered_dexes = df_sankey["DEX"].unique()
    propamm_list = sorted([d for d in filtered_dexes if d in propamm_dexes])
    other_dexes = sorted([d for d in filtered_dexes if d not in propamm_dexes])
    dexes = propamm_list + other_dexes
    validators = sorted(df_sankey["VALIDATOR_DISPLAY"].unique())

    # Create node labels and indices
    all_nodes = frontends + dexes + validators
    node_indices = {node: i for i, node in enumerate(all_nodes)}

    # Build links for Frontend -> DEX
    sources_1 = [node_indices[row["FRONTEND"]] for _, row in frontend_to_dex.iterrows()]
    targets_1 = [node_indices[row["DEX"]] for _, row in frontend_to_dex.iterrows()]
    values_1 = frontend_to_dex["VOLUME_USD"].tolist()

    # Build links for DEX -> Validator
    sources_2 = [node_indices[row["DEX"]] for _, row in dex_to_validator.iterrows()]
    targets_2 = [node_indices[row["VALIDATOR"]] for _, row in dex_to_validator.iterrows()]
    values_2 = dex_to_validator["VOLUME_USD"].tolist()

    # Combine all links
    all_sources = sources_1 + sources_2
    all_targets = targets_1 + targets_2
    all_values = values_1 + values_2

    # Assign colors by layer and type
    node_colors = []
    node_colors.extend(["#636EFA"] * len(frontends))  # Blue for frontends
    for dex in dexes:
        if dex in propamm_dexes:
            node_colors.append("#AB63FA")  # Purple for PropAMM
        else:
            node_colors.append("#EF553B")  # Red for regular DEX
    node_colors.extend(["#00CC96"] * len(validators))  # Green for validators

    # Create custom labels with formatted volumes for links
    link_labels = [format_volume(v) for v in all_values]

    # Scale values to billions for cleaner display
    values_in_billions = [v / 1e9 for v in all_values]

    # Create Sankey diagram
    fig_sankey = go.Figure(data=[go.Sankey(
        valueformat="$.1f",
        valuesuffix="B",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=all_sources,
            target=all_targets,
            value=values_in_billions,
            customdata=link_labels,
            hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>'
        )
    )])

    fig_sankey.update_layout(
        title_text="Frontend → DEX → Validator Flow (Volume USD)<br><sup>Purple = PropAMM | Red = DEX AMM</sup>",
        font_size=12,
        height=800
    )

    st.plotly_chart(fig_sankey, use_container_width=True)

    # Summary table (excluding unknown/unlabeled entities)
    st.subheader("Top Flows by Volume (among labeled entities)")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Frontend → DEX**")
        top_frontend_dex = frontend_to_dex[
            ~frontend_to_dex["FRONTEND"].str.contains("Unknown", case=False, na=False)
        ].sort_values("VOLUME_USD", ascending=False).head(10).copy()
        top_frontend_dex["VOLUME_USD"] = top_frontend_dex["VOLUME_USD"].apply(format_volume)
        st.dataframe(top_frontend_dex, use_container_width=True, hide_index=True)

    with col2:
        st.write("**DEX → Validator**")
        top_dex_validator = dex_to_validator[
            ~dex_to_validator["VALIDATOR"].str.contains("Unlabeled", case=False, na=False)
        ].sort_values("VOLUME_USD", ascending=False).head(10).copy()
        top_dex_validator["VOLUME_USD"] = top_dex_validator["VOLUME_USD"].apply(format_volume)
        st.dataframe(top_dex_validator, use_container_width=True, hide_index=True)
