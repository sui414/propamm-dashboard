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
tab1, tab2, tab3, tab4 = st.tabs(["📊 PropAMM Metrics", "📈 Intraday Block Position", "🔀 Orderflow Sankey", "📉 MEV Market Share"])

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

    st.divider()

    # =====================
    # Trade Size Distribution
    # =====================
    st.header("Trade Size Distribution")

    @st.cache_data
    def load_trade_size_data():
        df_size = pd.read_csv("propamm_trade_size_distribution.csv")
        return df_size

    df_trade_size = load_trade_size_data()

    # Define bucket order
    bucket_order = ['<$10', '$10-$100', '$100-$1K', '$1K-$10K', '$10K-$100K', '>$100K']

    # Toggle for metric
    size_metric = st.radio(
        "Show by:",
        options=["Volume (USD)", "Fill Count"],
        horizontal=True,
        key="trade_size_metric"
    )
    size_col = "VOLUME_USD" if size_metric == "Volume (USD)" else "FILL_COUNT"

    size_col1, size_col2 = st.columns(2)

    with size_col1:
        # Stacked bar chart by project
        fig_size_stack = px.bar(
            df_trade_size,
            x="PROJECT",
            y=size_col,
            color="TRADE_SIZE_BUCKET",
            category_orders={"TRADE_SIZE_BUCKET": bucket_order},
            title=f"Trade Size Distribution by Project ({size_metric})",
            labels={size_col: size_metric, "PROJECT": "Project", "TRADE_SIZE_BUCKET": "Trade Size"}
        )
        fig_size_stack.update_layout(height=500, barmode='stack')
        st.plotly_chart(fig_size_stack, use_container_width=True)

    with size_col2:
        # Normalized 100% stacked bar
        df_size_pct = df_trade_size.copy()
        df_size_pct["PCT"] = df_size_pct.groupby("PROJECT")[size_col].transform(lambda x: x / x.sum() * 100)

        fig_size_pct = px.bar(
            df_size_pct,
            x="PROJECT",
            y="PCT",
            color="TRADE_SIZE_BUCKET",
            category_orders={"TRADE_SIZE_BUCKET": bucket_order},
            title=f"Trade Size Distribution % by Project",
            labels={"PCT": "Percentage (%)", "PROJECT": "Project", "TRADE_SIZE_BUCKET": "Trade Size"}
        )
        fig_size_pct.update_layout(height=500, barmode='stack')
        st.plotly_chart(fig_size_pct, use_container_width=True)

    st.divider()

    # =====================
    # Trading Pairs Volume
    # =====================
    st.header("Trading Pairs Breakdown")

    @st.cache_data
    def load_pair_data():
        df_pairs = pd.read_csv("propamm_pair_volume.csv")
        # Clean up the TOKEN_PAIRS column (remove brackets and quotes)
        df_pairs["TOKEN_PAIRS"] = df_pairs["TOKEN_PAIRS"].str.replace(r'[\[\]"]', '', regex=True)
        df_pairs = df_pairs.dropna(subset=["VOLUME_USD"])
        return df_pairs

    df_pairs = load_pair_data()

    # Project selector
    pair_projects = sorted(df_pairs["PROJECT"].unique())
    selected_project = st.selectbox("Select Project", pair_projects, key="pair_project_select")

    pair_col1, pair_col2 = st.columns(2)

    with pair_col1:
        # Top pairs for selected project
        project_pairs = df_pairs[df_pairs["PROJECT"] == selected_project].nlargest(15, "VOLUME_USD")

        fig_pairs = px.bar(
            project_pairs,
            x="VOLUME_USD",
            y="TOKEN_PAIRS",
            orientation='h',
            title=f"Top 15 Trading Pairs - {selected_project}",
            labels={"VOLUME_USD": "Volume (USD)", "TOKEN_PAIRS": "Pair"}
        )
        fig_pairs.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

    with pair_col2:
        # Pie chart of top pairs
        fig_pairs_pie = px.pie(
            project_pairs,
            values="VOLUME_USD",
            names="TOKEN_PAIRS",
            title=f"Volume Share by Pair - {selected_project}"
        )
        fig_pairs_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pairs_pie.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_pairs_pie, use_container_width=True)

    # All projects pair comparison
    st.subheader("Top Pairs Across All PropAMMs")

    # Get top 5 pairs per project
    top_pairs_all = df_pairs.groupby("PROJECT").apply(
        lambda x: x.nlargest(5, "VOLUME_USD")
    ).reset_index(drop=True)

    fig_pairs_all = px.bar(
        top_pairs_all,
        x="PROJECT",
        y="VOLUME_USD",
        color="TOKEN_PAIRS",
        title="Top 5 Pairs by Volume for Each PropAMM",
        labels={"VOLUME_USD": "Volume (USD)", "PROJECT": "Project", "TOKEN_PAIRS": "Pair"}
    )
    fig_pairs_all.update_layout(height=500, barmode='group')
    st.plotly_chart(fig_pairs_all, use_container_width=True)

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

        # Load validator mapping
        df_validators = pd.read_csv("Validators All.csv")

        # Create mapping from pubkey to name and client
        validator_name_map = {}
        validator_client_map = {}
        for _, row in df_validators.iterrows():
            pubkey = row["account"]
            name = row["name"] if pd.notna(row["name"]) and row["name"].strip() != "" else None
            client = row["softwareClient"] if pd.notna(row["softwareClient"]) else "Unknown Client"
            validator_name_map[pubkey] = name
            validator_client_map[pubkey] = client

        # Map validators to display names
        def get_validator_display(pubkey):
            if pubkey in validator_name_map and validator_name_map[pubkey]:
                return validator_name_map[pubkey]
            else:
                return "Unlabeled Validators"

        # Map validators to clients
        def get_validator_client(pubkey):
            return validator_client_map.get(pubkey, "Unknown Client")

        df_sankey["VALIDATOR_DISPLAY"] = df_sankey["VALIDATOR"].apply(get_validator_display)
        df_sankey["CLIENT"] = df_sankey["VALIDATOR"].apply(get_validator_client)

        return df_sankey

    df_sankey_raw = load_sankey_data()

    # PropAMM DEXes (separate from regular DEX AMMs)
    propamm_dexes = {'humidifi', 'bisonfi', 'solfi', 'goonfi', 'tesserav', 'alphaq', 'aquifer', 'zerofi', 'lifinity'}

    # Filters
    st.subheader("Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    all_frontends = sorted(df_sankey_raw["FRONTEND"].unique())
    all_dexes_list = sorted(df_sankey_raw["DEX"].unique())
    all_clients = sorted(df_sankey_raw["CLIENT"].unique())

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
        selected_clients = st.multiselect(
            "Client",
            options=all_clients,
            default=all_clients,
            key="client_filter"
        )

    # Apply filters
    df_sankey = df_sankey_raw[
        (df_sankey_raw["FRONTEND"].isin(selected_frontends)) &
        (df_sankey_raw["DEX"].isin(selected_dexes)) &
        (df_sankey_raw["CLIENT"].isin(selected_clients))
    ]

    # Check if data exists after filtering
    if df_sankey.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        st.stop()

    # Aggregate Frontend -> DEX (sum across all validators)
    frontend_to_dex = df_sankey.groupby(["FRONTEND", "DEX"])["VOLUME_USD"].sum().reset_index()

    # Aggregate DEX -> Client (skip validator layer)
    dex_to_client = df_sankey.groupby(["DEX", "CLIENT"])["VOLUME_USD"].sum().reset_index()

    # Build node list: Frontends, DEXes (PropAMM first, then others), Clients
    frontends = sorted(df_sankey["FRONTEND"].unique())
    filtered_dexes = df_sankey["DEX"].unique()
    propamm_list = sorted([d for d in filtered_dexes if d in propamm_dexes])
    other_dexes = sorted([d for d in filtered_dexes if d not in propamm_dexes])
    dexes = propamm_list + other_dexes
    clients = sorted(df_sankey["CLIENT"].unique())

    # Create node labels and indices
    all_nodes = frontends + dexes + clients
    node_indices = {node: i for i, node in enumerate(all_nodes)}

    # Build links for Frontend -> DEX
    sources_1 = [node_indices[row["FRONTEND"]] for _, row in frontend_to_dex.iterrows()]
    targets_1 = [node_indices[row["DEX"]] for _, row in frontend_to_dex.iterrows()]
    values_1 = frontend_to_dex["VOLUME_USD"].tolist()

    # Build links for DEX -> Client
    sources_2 = [node_indices[row["DEX"]] for _, row in dex_to_client.iterrows()]
    targets_2 = [node_indices[row["CLIENT"]] for _, row in dex_to_client.iterrows()]
    values_2 = dex_to_client["VOLUME_USD"].tolist()

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
    node_colors.extend(["#FFA15A"] * len(clients))  # Orange for clients

    # Create custom labels with formatted volumes for links
    link_labels = [format_volume(v) for v in all_values]

    # Scale values to millions for display
    values_in_millions = [v / 1e6 for v in all_values]

    # Create Sankey diagram
    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
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
            value=values_in_millions,
            customdata=link_labels,
            hovertemplate='%{source.label} → %{target.label}<br>%{customdata}<extra></extra>'
        )
    )])

    # Override the default value format to show $M
    fig_sankey.data[0].valueformat = "$,.0f"
    fig_sankey.data[0].valuesuffix = "M"

    fig_sankey.update_layout(
        title_text="Frontend → DEX → Client Flow (Volume USD)<br><sup>Purple = PropAMM | Red = DEX AMM | Orange = Clients</sup>",
        font_size=12,
        height=700
    )

    # Set node label text color to white
    fig_sankey.update_traces(textfont_color="white")

    st.caption("Validator labels sourced from [Anza Scheduler War](https://schedulerwar.vercel.app/)")

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
        st.write("**DEX → Client**")
        top_dex_client = dex_to_client.sort_values("VOLUME_USD", ascending=False).head(10).copy()
        top_dex_client["VOLUME_USD"] = top_dex_client["VOLUME_USD"].apply(format_volume)
        st.dataframe(top_dex_client, use_container_width=True, hide_index=True)

# ===================
# TAB 4: MEV Market Share
# ===================
with tab4:
    st.header("MEV Market Share")

    @st.cache_data
    def load_mev_data():
        df_mev = pd.read_csv("solana_mev_market_share.csv")
        df_mev["DAY"] = pd.to_datetime(df_mev["DAY"])

        # Load validator mapping
        df_validators = pd.read_csv("Validators All.csv")

        # Create mapping from pubkey to name, client, and stake
        validator_name_map = {}
        validator_client_map = {}
        validator_stake_map = {}
        for _, row in df_validators.iterrows():
            pubkey = row["account"]
            name = row["name"] if pd.notna(row["name"]) and str(row["name"]).strip() != "" else None
            client = row["softwareClient"] if pd.notna(row["softwareClient"]) else "Unknown Client"
            stake = row["activeStake"] if pd.notna(row["activeStake"]) else 0
            validator_name_map[pubkey] = name
            validator_client_map[pubkey] = client
            validator_stake_map[pubkey] = stake

        # Map validators to display names, clients, and stake
        df_mev["VALIDATOR"] = df_mev["PUBKEY"].apply(
            lambda x: validator_name_map.get(x) if validator_name_map.get(x) else "Unlabeled"
        )
        df_mev["CLIENT"] = df_mev["PUBKEY"].apply(
            lambda x: validator_client_map.get(x, "Unknown Client")
        )
        df_mev["STAKE"] = df_mev["PUBKEY"].apply(
            lambda x: validator_stake_map.get(x, 0)
        )
        # Calculate SOL per slot (mean)
        df_mev["SOL_PER_SLOT"] = df_mev["SOLS"] / df_mev["SLOTS"].replace(0, 1)

        return df_mev, validator_stake_map

    df_mev, validator_stake_map = load_mev_data()

    # =====================
    # Summary KPIs
    # =====================
    st.subheader("Summary KPIs")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

    total_sol = df_mev["SOLS"].sum()
    total_slots = df_mev["SLOTS"].sum()
    active_validators = df_mev["PUBKEY"].nunique()
    avg_sol_per_slot = total_sol / total_slots if total_slots > 0 else 0
    median_sol_per_slot = df_mev["SOL_MEDIAN"].median() if "SOL_MEDIAN" in df_mev.columns else 0
    date_range = f"{df_mev['DAY'].min().strftime('%Y-%m-%d')} to {df_mev['DAY'].max().strftime('%Y-%m-%d')}"

    with kpi_col1:
        st.metric("Total SOL Extracted", f"{total_sol:,.2f}")
    with kpi_col2:
        st.metric("Total Slots", f"{total_slots:,.0f}")
    with kpi_col3:
        st.metric("Active Validators", f"{active_validators:,}")
    with kpi_col4:
        st.metric("Median SOL/Slot", f"{median_sol_per_slot:.4f}")
    with kpi_col5:
        st.metric("Avg SOL/Slot", f"{avg_sol_per_slot:.4f}")
    with kpi_col6:
        st.metric("Date Range", date_range)

    st.divider()

    # Toggles
    toggle_col1, toggle_col2 = st.columns(2)
    with toggle_col1:
        metric_choice = st.radio(
            "Select Metric",
            options=["Slots Count", "Total SOLs Received"],
            horizontal=True,
            key="mev_metric_toggle"
        )
    with toggle_col2:
        view_choice = st.radio(
            "View Mode",
            options=["Absolute Values", "Percentage (%)"],
            horizontal=True,
            key="mev_view_toggle"
        )

    metric_col = "SLOTS" if metric_choice == "Slots Count" else "SOLS"
    metric_label = "Slots" if metric_choice == "Slots Count" else "SOL"
    is_percentage = view_choice == "Percentage (%)"
    groupnorm = "percent" if is_percentage else None
    y_suffix = "%" if is_percentage else ""

    # --- Chart 1: Market Share by Validator ---
    st.subheader(f"Market Share by Validator ({metric_label})")

    # Aggregate by day and validator (show all, no top N limit)
    validator_daily = df_mev.groupby(["DAY", "VALIDATOR"])[metric_col].sum().reset_index()

    # Pivot for stacked area
    validator_pivot = validator_daily.pivot(
        index="DAY", columns="VALIDATOR", values=metric_col
    ).fillna(0)

    # Sort columns by total (descending - biggest first for bottom of stack)
    col_totals = validator_pivot.sum().sort_values(ascending=False)
    sorted_cols = [c for c in col_totals.index if c != "Unlabeled"]
    if "Unlabeled" in col_totals.index:
        sorted_cols.append("Unlabeled")  # Unlabeled at end
    validator_pivot = validator_pivot[sorted_cols]

    # Calculate totals for pie chart
    validator_totals = validator_pivot.sum().reset_index()
    validator_totals.columns = ["Validator", "Total"]

    # Layout: area chart + pie chart
    val_col1, val_col2 = st.columns([3, 1])

    with val_col1:
        fig_validator = go.Figure()
        hover_format = '%{y:.2f}%' if is_percentage else '%{y:,.0f}'
        for col in validator_pivot.columns:
            fig_validator.add_trace(go.Scatter(
                name=col,
                x=validator_pivot.index,
                y=validator_pivot[col],
                mode='lines',
                stackgroup='one',
                groupnorm=groupnorm,
                hovertemplate=f'{col}: {hover_format}<extra></extra>'
            ))

        fig_validator.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{metric_label} {y_suffix}".strip(),
            legend_title="Validator",
            height=600,
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0)",
                traceorder="reversed",
                font=dict(size=10),
                itemsizing="constant"
            ),
            margin=dict(r=150)
        )
        st.plotly_chart(fig_validator, use_container_width=True, config={'scrollZoom': True})

    with val_col2:
        fig_val_pie = px.pie(
            validator_totals,
            values="Total",
            names="Validator",
            title="Total Share"
        )
        fig_val_pie.update_traces(textposition='inside', textinfo='percent')
        fig_val_pie.update_layout(
            showlegend=False,
            height=500,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_val_pie, use_container_width=True)

    # --- Chart 2: Market Share by Client ---
    st.subheader(f"Market Share by Client ({metric_label})")

    # Define client color mapping by family
    client_color_map = {
        # Jito family - Blues
        "JitoLabs": "#1f77b4",
        "AgaveBam": "#4a9fd4",
        # Harmonic family - Greens
        "Harmonic": "#2ca02c",
        "HarmonicAgave": "#5fd35f",
        # Frankendancer family - Oranges
        "Frankendancer Vanilla": "#ff7f0e",
        "Frankendancer Rev": "#ffb366",
        # Agave - Purple
        "Agave": "#9467bd",
        # Firedancer - Red
        "Firedancer": "#d62728",
        # Rakurai - Pink
        "Rakurai": "#e377c2",
        # Unknown - Gray
        "Unknown Client": "#7f7f7f"
    }

    # Define client group order for time series (families together)
    client_group_order = [
        "JitoLabs", "AgaveBam",
        "Harmonic", "HarmonicAgave",
        "Frankendancer Vanilla", "Frankendancer Rev",
        "Agave", "Firedancer", "Rakurai", "Unknown Client"
    ]

    # Aggregate by day and client
    client_daily = df_mev.groupby(["DAY", "CLIENT"])[metric_col].sum().reset_index()

    # Pivot for stacked area
    client_pivot = client_daily.pivot(
        index="DAY", columns="CLIENT", values=metric_col
    ).fillna(0)

    # Sort columns by family group order (for time series)
    available_clients = client_pivot.columns.tolist()
    sorted_cols_client = [c for c in client_group_order if c in available_clients]
    # Add any clients not in the predefined order
    for c in available_clients:
        if c not in sorted_cols_client:
            sorted_cols_client.append(c)
    client_pivot = client_pivot[sorted_cols_client]

    # Calculate totals for pie chart (sorted by share)
    client_totals = client_pivot.sum().sort_values(ascending=False).reset_index()
    client_totals.columns = ["Client", "Total"]

    # Layout: area chart + pie chart
    cli_col1, cli_col2 = st.columns([3, 1])

    with cli_col1:
        fig_client = go.Figure()
        for col in client_pivot.columns:
            color = client_color_map.get(col, "#bcbd22")  # Default yellow-green for unknown
            fig_client.add_trace(go.Scatter(
                name=col,
                x=client_pivot.index,
                y=client_pivot[col],
                mode='lines',
                stackgroup='one',
                groupnorm=groupnorm,
                line=dict(color=color),
                fillcolor=color,
                hovertemplate=f'{col}: {hover_format}<extra></extra>'
            ))

        fig_client.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{metric_label} {y_suffix}".strip(),
            legend_title="Client",
            height=600,
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0)",
                traceorder="reversed",
                font=dict(size=10),
                itemsizing="constant"
            ),
            margin=dict(r=150)
        )
        st.plotly_chart(fig_client, use_container_width=True)

    with cli_col2:
        # Pie chart with same color mapping, sorted by share
        pie_colors = [client_color_map.get(c, "#bcbd22") for c in client_totals["Client"]]
        fig_cli_pie = go.Figure(data=[go.Pie(
            labels=client_totals["Client"],
            values=client_totals["Total"],
            marker=dict(colors=pie_colors),
            textposition='inside',
            textinfo='percent',
            sort=False  # Keep the order from client_totals (sorted by share)
        )])
        fig_cli_pie.update_layout(
            title="Total Share",
            showlegend=False,
            height=500,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_cli_pie, use_container_width=True)

    st.divider()

    # =====================
    # SOL per Slot Efficiency
    # =====================
    st.subheader("SOL per Slot Efficiency")

    eff_col1, eff_col2 = st.columns(2)

    with eff_col1:
        # By Validator - Median only
        validator_efficiency = df_mev.groupby("VALIDATOR").agg({
            "SOL_MEDIAN": "median"
        }).reset_index()
        validator_efficiency = validator_efficiency.sort_values("SOL_MEDIAN", ascending=True).tail(20)

        fig_val_eff = px.bar(
            validator_efficiency,
            y="VALIDATOR",
            x="SOL_MEDIAN",
            orientation='h',
            title="Top 20 Validators by Median SOL/Slot",
            labels={"SOL_MEDIAN": "Median SOL/Slot", "VALIDATOR": "Validator"}
        )
        fig_val_eff.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_val_eff, use_container_width=True)

    with eff_col2:
        # By Client - Median only
        client_efficiency = df_mev.groupby("CLIENT").agg({
            "SOL_MEDIAN": "median"
        }).reset_index()
        client_efficiency = client_efficiency.sort_values("SOL_MEDIAN", ascending=False)

        fig_cli_eff = px.bar(
            client_efficiency,
            x="CLIENT",
            y="SOL_MEDIAN",
            title="Client Efficiency by Median SOL/Slot",
            labels={"SOL_MEDIAN": "Median SOL/Slot", "CLIENT": "Client"},
            color="SOL_MEDIAN",
            color_continuous_scale="Viridis"
        )
        fig_cli_eff.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_cli_eff, use_container_width=True)

    # Median SOL/Slot over time
    st.subheader("Median SOL/Slot Over Time")
    daily_efficiency = df_mev.groupby("DAY").agg({
        "SOL_MEDIAN": "median"
    }).reset_index()

    fig_median_time = px.line(
        daily_efficiency,
        x="DAY",
        y="SOL_MEDIAN",
        title="Daily Median SOL/Slot",
        labels={"SOL_MEDIAN": "Median SOL/Slot", "DAY": "Date"}
    )
    fig_median_time.update_layout(height=400)
    st.plotly_chart(fig_median_time, use_container_width=True)

    st.divider()

    # =====================
    # Stake vs MEV Share Comparison
    # =====================
    st.subheader("Stake Share vs MEV Share (Who's Overperforming?)")

    # Calculate MEV share and stake share by validator
    validator_mev_total = df_mev.groupby("PUBKEY").agg({
        "SOLS": "sum",
        "SLOTS": "sum",
        "VALIDATOR": "first",
        "STAKE": "first"
    }).reset_index()

    # Filter out validators with no stake data
    validator_mev_total = validator_mev_total[validator_mev_total["STAKE"] > 0]

    total_mev = validator_mev_total["SOLS"].sum()
    total_stake = validator_mev_total["STAKE"].sum()

    validator_mev_total["MEV_SHARE"] = (validator_mev_total["SOLS"] / total_mev * 100) if total_mev > 0 else 0
    validator_mev_total["STAKE_SHARE"] = (validator_mev_total["STAKE"] / total_stake * 100) if total_stake > 0 else 0
    validator_mev_total["OVERPERFORMANCE"] = validator_mev_total["MEV_SHARE"] - validator_mev_total["STAKE_SHARE"]

    # Scatter plot
    fig_stake_mev = px.scatter(
        validator_mev_total,
        x="STAKE_SHARE",
        y="MEV_SHARE",
        hover_name="VALIDATOR",
        size="SOLS",
        color="OVERPERFORMANCE",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={
            "STAKE_SHARE": "Stake Share (%)",
            "MEV_SHARE": "MEV Share (%)",
            "OVERPERFORMANCE": "Overperformance (%)"
        },
        title="Stake Share vs MEV Share (above diagonal = overperforming)"
    )
    # Add diagonal line (y = x)
    max_val = max(validator_mev_total["STAKE_SHARE"].max(), validator_mev_total["MEV_SHARE"].max())
    fig_stake_mev.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Fair Share (y=x)',
        showlegend=True
    ))
    fig_stake_mev.update_layout(height=500)
    st.plotly_chart(fig_stake_mev, use_container_width=True)

    st.divider()

    # =====================
    # Market Concentration
    # =====================
    st.subheader("Market Concentration")

    conc_col1, conc_col2 = st.columns(2)

    with conc_col1:
        # HHI over time (by validator)
        daily_totals = df_mev.groupby("DAY")["SOLS"].sum().reset_index()
        daily_totals.columns = ["DAY", "TOTAL_SOLS"]

        validator_daily_sols = df_mev.groupby(["DAY", "VALIDATOR"])["SOLS"].sum().reset_index()
        validator_daily_sols = validator_daily_sols.merge(daily_totals, on="DAY")
        validator_daily_sols["SHARE"] = validator_daily_sols["SOLS"] / validator_daily_sols["TOTAL_SOLS"]
        validator_daily_sols["SHARE_SQ"] = validator_daily_sols["SHARE"] ** 2

        hhi_daily = validator_daily_sols.groupby("DAY")["SHARE_SQ"].sum().reset_index()
        hhi_daily.columns = ["DAY", "HHI"]
        hhi_daily["HHI"] = hhi_daily["HHI"] * 10000  # Scale to standard HHI (0-10000)

        fig_hhi = px.line(
            hhi_daily,
            x="DAY",
            y="HHI",
            title="HHI Index Over Time (Validator Concentration)",
            labels={"HHI": "HHI (0-10000)", "DAY": "Date"}
        )
        fig_hhi.add_hline(y=2500, line_dash="dash", line_color="orange", annotation_text="Highly Concentrated")
        fig_hhi.add_hline(y=1500, line_dash="dash", line_color="green", annotation_text="Moderately Concentrated")
        fig_hhi.update_layout(height=400)
        st.plotly_chart(fig_hhi, use_container_width=True)

    with conc_col2:
        # Top 5/10 share over time
        def calc_top_n_share(group, n):
            total = group["SOLS"].sum()
            top_n = group.nlargest(n, "SOLS")["SOLS"].sum()
            return top_n / total * 100 if total > 0 else 0

        top_share = df_mev.groupby("DAY").apply(
            lambda x: pd.Series({
                "Top 5 Share": calc_top_n_share(x, 5),
                "Top 10 Share": calc_top_n_share(x, 10)
            })
        ).reset_index()

        fig_top_share = go.Figure()
        fig_top_share.add_trace(go.Scatter(
            x=top_share["DAY"],
            y=top_share["Top 5 Share"],
            name="Top 5 Validators",
            mode="lines+markers"
        ))
        fig_top_share.add_trace(go.Scatter(
            x=top_share["DAY"],
            y=top_share["Top 10 Share"],
            name="Top 10 Validators",
            mode="lines+markers"
        ))
        fig_top_share.update_layout(
            title="Top N Validator Share Over Time",
            xaxis_title="Date",
            yaxis_title="Market Share (%)",
            height=400
        )
        st.plotly_chart(fig_top_share, use_container_width=True)

    st.divider()

    # =====================
    # Validator Leaderboard
    # =====================
    st.subheader("Validator Leaderboard")

    leaderboard = df_mev.groupby("VALIDATOR").agg({
        "SOLS": "sum",
        "SLOTS": "sum",
        "CLIENT": "first",
        "SOL_MEDIAN": "median"
    }).reset_index()
    leaderboard["MEAN_SOL_PER_SLOT"] = leaderboard["SOLS"] / leaderboard["SLOTS"].replace(0, 1)
    leaderboard["MEV_SHARE_%"] = leaderboard["SOLS"] / leaderboard["SOLS"].sum() * 100
    leaderboard = leaderboard.sort_values("SOLS", ascending=False).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1  # Start rank from 1
    leaderboard.index.name = "Rank"

    # Format columns
    leaderboard_display = leaderboard.copy()
    leaderboard_display["SOLS"] = leaderboard_display["SOLS"].apply(lambda x: f"{x:,.2f}")
    leaderboard_display["SLOTS"] = leaderboard_display["SLOTS"].apply(lambda x: f"{x:,.0f}")
    leaderboard_display["MEAN_SOL_PER_SLOT"] = leaderboard_display["MEAN_SOL_PER_SLOT"].apply(lambda x: f"{x:.4f}")
    leaderboard_display["SOL_MEDIAN"] = leaderboard_display["SOL_MEDIAN"].apply(lambda x: f"{x:.4f}")
    leaderboard_display["MEV_SHARE_%"] = leaderboard_display["MEV_SHARE_%"].apply(lambda x: f"{x:.2f}%")
    leaderboard_display = leaderboard_display[["VALIDATOR", "SOLS", "SLOTS", "SOL_MEDIAN", "MEAN_SOL_PER_SLOT", "MEV_SHARE_%", "CLIENT"]]
    leaderboard_display.columns = ["Validator", "Total SOL", "Total Slots", "Median SOL/Slot", "Mean SOL/Slot", "MEV Share", "Client"]

    st.dataframe(leaderboard_display, use_container_width=True, height=400)

    st.caption("Validator labels sourced from [Anza Scheduler War](https://schedulerwar.vercel.app/)")
