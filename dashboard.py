import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Project Metrics Dashboard", layout="wide")

st.title("Project Metrics Dashboard")

# Load data
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
