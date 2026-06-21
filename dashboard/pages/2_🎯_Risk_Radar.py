"""
Page 2: Logistics Risk Radar - Raw Material Delivery Risk
Requires: python pipeline/train.py to have been run first.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,46,0.6)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
)

@st.cache_data(ttl=300)
def load_predictions():
    current_path = Path(__file__).resolve()
    if current_path.parent.name == "pages":
        base = current_path.parent.parent.parent
    else:
        base = current_path.parent.parent
    path = base / "data" / "processed" / "predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["order_date"])

@st.cache_data(ttl=300)
def load_metrics():
    current_path = Path(__file__).resolve()
    if current_path.parent.name == "pages":
        base = current_path.parent.parent.parent
    else:
        base = current_path.parent.parent
    path = base / "data" / "processed" / "model_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

df = load_predictions()
if df is None:
    st.error("Predictions not found. Run `python pipeline/train.py` first.")
    st.stop()

metrics_df = load_metrics()

# Sidebar filters
with st.sidebar:
    st.markdown("### 🎯 Risk Filters")
    risk_filter = st.multiselect(
        "Risk Level",
        options=["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
        default=["🔴 High Risk", "🟡 Medium Risk"],
    )
    regions = ["All"] + sorted(df["order_region"].dropna().unique().tolist())
    sel_region = st.selectbox("Region", regions, key="rr_region")
    modes = ["All"] + sorted(df["shipping_mode"].dropna().unique().tolist())
    sel_mode = st.selectbox("Shipping Mode", modes, key="rr_mode")
    prob_threshold = st.slider("Min. Delay Probability", 0.0, 1.0, 0.45, 0.05)

fdf = df[df["risk_label"].isin(risk_filter)].copy() if risk_filter else df.copy()
fdf = fdf[fdf["delay_probability"] >= prob_threshold]
if sel_region != "All":
    fdf = fdf[fdf["order_region"] == sel_region]
if sel_mode != "All":
    fdf = fdf[fdf["shipping_mode"] == sel_mode]

# ── KPI Row ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Predictive Risk Overview</div>', unsafe_allow_html=True)

high_risk = (df["risk_label"] == "🔴 High Risk").sum()
med_risk  = (df["risk_label"] == "🟡 Medium Risk").sum()
low_risk  = (df["risk_label"] == "🟢 Low Risk").sum()
avg_prob  = df["delay_probability"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Total Shipments", f"{len(df):,}")
c2.metric("🚨 High Risk", f"{high_risk:,}")
c3.metric("⚠️ Medium Risk", f"{med_risk:,}")
c4.metric("✅ Low Risk", f"{low_risk:,}")
c5.metric("🧠 Avg Delay Prob.", f"{avg_prob:.1%}")

st.markdown("<br>", unsafe_allow_html=True)



# ── Geo Map + Region Risk ─────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Geospatial Risk Radar</div>', unsafe_allow_html=True)
col_l, col_r = st.columns([3, 2])

with col_l:
    map_df = fdf.dropna(subset=["latitude", "longitude"]).head(2000)
    if len(map_df) > 0:
        fig = px.scatter_mapbox(
            map_df, lat="latitude", lon="longitude",
            color="delay_probability", size="delay_probability",
            hover_name="order_region",
            hover_data={"order_country": True, "shipping_mode": True,
                        "delay_probability": ":.1%", "risk_label": True,
                        "latitude": False, "longitude": False},
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            size_max=14, zoom=1, mapbox_style="carto-darkmatter",
            title="At-Risk Shipments – Map", height=380,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No shipments match current filters for map view.")

with col_r:
    region_risk = (
        df.groupby("order_region")
          .agg(avg_prob=("delay_probability", "mean"),
               high_risk=("predicted_late", "sum"),
               total=("predicted_late", "count"))
          .reset_index()
          .sort_values("avg_prob", ascending=True).tail(12)
    )
    colors = ["#ef4444" if p > 0.65 else ("#f59e0b" if p > 0.5 else "#10b981")
              for p in region_risk["avg_prob"]]
    fig = go.Figure(go.Bar(
        x=region_risk["avg_prob"], y=region_risk["order_region"],
        orientation="h", marker_color=colors,
        text=[f"{p:.0%}" for p in region_risk["avg_prob"]],
        textposition="outside", textfont=dict(color="white", size=10),
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#f59e0b", opacity=0.7)
    fig.update_layout(**CHART_LAYOUT, title="Delay Risk by Region", height=380)
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ── Risk Trends ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Risk Trends</div>', unsafe_allow_html=True)
col_l, col_r = st.columns(2)

with col_l:
    weekly = df.copy()
    weekly["week"] = weekly["order_date"].dt.to_period("W").astype(str)
    wr = weekly.groupby("week")["delay_probability"].mean().reset_index().tail(52)
    fig = go.Figure(go.Scatter(
        x=wr["week"], y=wr["delay_probability"], mode="lines",
        line=dict(color="#7c3aed", width=2.5), fill="tozeroy",
        fillcolor="rgba(124,58,237,0.1)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444",
                  annotation_text="Risk Threshold", annotation_font_color="#ef4444")
    fig.update_layout(**CHART_LAYOUT, title="Weekly Risk Trend", height=270)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    mode_risk = (df.groupby("shipping_mode")["delay_probability"].mean()
                   .reset_index().sort_values("delay_probability", ascending=False))
    colors = ["#ef4444" if p > 0.6 else ("#f59e0b" if p > 0.45 else "#10b981")
              for p in mode_risk["delay_probability"]]
    fig = go.Figure(go.Bar(
        x=mode_risk["shipping_mode"], y=mode_risk["delay_probability"],
        marker_color=colors,
        text=[f"{p:.1%}" for p in mode_risk["delay_probability"]],
        textposition="outside", textfont=dict(color="white"),
    ))
    fig.update_layout(**CHART_LAYOUT, title="Delay Risk by Shipping Mode", height=270)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ── At-Risk Orders Table ──────────────────────────────────────────
st.markdown(f'<div class="section-header">🚨 At-Risk Shipments – Action Required ({len(fdf):,})</div>',
            unsafe_allow_html=True)

if len(fdf) == 0:
    st.info("✅ No shipments match current filters. Adjust the sidebar filters.")
else:
    display_cols = ["order_id", "order_date", "order_region", "order_country",
                    "shipping_mode", "category_name", "days_for_shipment_scheduled",
                    "delay_probability", "risk_label"]
    display_cols = [c for c in display_cols if c in fdf.columns]
    table_df = fdf[display_cols].copy()
    table_df["order_date"]        = table_df["order_date"].dt.strftime("%Y-%m-%d")
    table_df["delay_probability"] = table_df["delay_probability"].round(3)
    table_df = table_df.sort_values("delay_probability", ascending=False).head(200)
    table_df.rename(columns={
        "order_id": "Shipment ID", "order_date": "Order Date", "order_region": "Region",
        "order_country": "Country", "shipping_mode": "Ship Mode", "category_name": "Category",
        "days_for_shipment_scheduled": "Sched. Days",
        "delay_probability": "Delay Prob.", "risk_label": "Risk Level",
    }, inplace=True)

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=420,
        column_config={
            "Delay Prob.": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.1%"),
        })

    st.download_button("📥 Export Risk Report (CSV)", data=table_df.to_csv(index=False),
                       file_name="high_risk_orders.csv", mime="text/csv", use_container_width=True)
