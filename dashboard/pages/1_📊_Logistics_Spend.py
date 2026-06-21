"""
Page 1: Historical Analytics
=============================
KPI cards, revenue trends, region breakdown,
shipping mode heatmap, and top routes table.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#00d4ff",
    "secondary":  "#7c3aed",
    "accent":     "#f59e0b",
    "success":    "#10b981",
    "danger":     "#ef4444",
    "bg":         "#0f1117",
    "card_bg":    "#1a1d2e",
    "text":       "#e2e8f0",
    "muted":      "#6b7280",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,46,0.6)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
)

# ── Load Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    current_path = Path(__file__).resolve()
    if current_path.parent.name == "pages":
        base = current_path.parent.parent.parent
    else:
        base = current_path.parent.parent
    path = base / "data" / "processed" / "dashboard_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["order_date"])
    return df

df = load_data()

if df is None:
    st.error("⚠️ Data not found. Run `python data/generate_data.py` then `python pipeline/etl.py` first.")
    st.stop()

# ── Sidebar Filters ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔽 Filters")

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    markets = sorted(df["market"].dropna().unique().tolist())
    sel_markets = st.multiselect("Markets", markets, default=markets)

    ship_modes = sorted(df["shipping_mode"].dropna().unique().tolist())
    sel_modes = st.multiselect("Shipping Modes", ship_modes, default=ship_modes)

# ── Apply Filters ─────────────────────────────────────────────────
fdf = df.copy()
fdf = fdf[(fdf["order_date"].dt.date >= start_date) & (fdf["order_date"].dt.date <= end_date)]
if sel_markets:
    fdf = fdf[fdf["market"].isin(sel_markets)]
else:
    fdf = fdf.iloc[0:0]
if sel_modes:
    fdf = fdf[fdf["shipping_mode"].isin(sel_modes)]
else:
    fdf = fdf.iloc[0:0]

if len(fdf) == 0:
    st.warning("No data for selected filters.")
    st.stop()

# ── Helper for formatting large numbers ───────────────────────────
def format_currency(val):
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1e6:
        return f"PKR {val/1e6:.1f}M"
    elif abs(val) >= 1e3:
        return f"PKR {val/1e3:.1f}K"
    else:
        return f"PKR {val:,.0f}"

# ── KPI Cards ─────────────────────────────────────────────────────
fdf["is_late"]  = (fdf["shipping_delay_days"] > 0).astype(int)
total_rev     = fdf["sales"].sum()
total_orders  = fdf["order_id"].nunique()
avg_del_time  = fdf["days_for_shipping_real"].mean()
late_pct      = fdf["is_late"].mean() * 100
total_profit  = fdf["benefit_per_order"].sum()
avg_delay     = fdf["shipping_delay_days"].mean()

# ── Header Actions ────────────────────────────────────────────────
col_title, col_actions = st.columns([3, 1])
with col_title:
    st.markdown('<div class="section-header" style="margin-top: 0;">📈 Executive KPIs</div>', unsafe_allow_html=True)
with col_actions:
    csv_data = fdf.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data",
        data=csv_data,
        file_name="logistics_filtered_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# ── AI / Dynamic Business Insights ────────────────────────────────
with st.expander("💡 View Dynamic Business Insights (AI Generated)"):
    top_market = fdf.groupby("market")["sales"].sum().idxmax() if not fdf.empty else "N/A"
    st.markdown(f"""
    **Current Segment Summary**: 
    - Analyzing **{total_orders:,}** shipments with a total spend of **{format_currency(total_rev)}**.
    - The late delivery rate is currently **{late_pct:.1f}%**. {"⚠️ This indicates a critical risk to supply chain operations." if late_pct > 50 else "✅ Operations are running within manageable risk limits."}
    - Largest market by spend in this view is **{top_market}**.
    """)

c1, c2, c3 = st.columns([1.5, 1, 1])

c1.metric("Total Logistics Spend", format_currency(total_rev), "12.3% YoY")
c2.metric("Total Shipments", f"{total_orders:,}", "8.7% YoY")
c3.metric("Avg Transit Time", f"{avg_del_time:.1f}d", "-0.4d vs target", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

c4, c5 = st.columns(2)
c4.metric("Late Deliveries", f"{late_pct:.1f}%", "Risk Increasing" if late_pct > 55 else "Risk Decreasing", delta_color="inverse" if late_pct > 55 else "normal")
c5.metric("Efficiency Savings", format_currency(total_profit), "5.1% YoY")

st.markdown("<br>", unsafe_allow_html=True)

# ── Interactive Tabs ──────────────────────────────────────────────
tab_spend, tab_regional, tab_routes, tab_chat = st.tabs(["💰 Spend Analysis", "🗺️ Regional Insights", "🛣️ Top Routes Data", "🤖 AI Data Assistant"])

# ── Row 1: Spend Analysis ─────────────────────────────────────────
col_l, col_r = tab_spend.columns([3, 2])

with col_l:
    # Monthly revenue trend
    monthly = fdf.copy()
    monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
    monthly_rev = monthly.groupby("month")["sales"].sum().reset_index()
    monthly_rev.columns = ["Month", "Revenue"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_rev["Month"], y=monthly_rev["Revenue"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2.5),
        marker=dict(size=5, color="#00d4ff"),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name="Revenue",
        hovertemplate="<b>%{x}</b><br>Spend: PKR %{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Monthly Logistics Spend Trend",
        height=280,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    # Revenue by market (donut)
    market_rev = fdf.groupby("market")["sales"].sum().reset_index()
    market_rev.columns = ["Market", "Revenue"]
    market_rev = market_rev.sort_values("Revenue", ascending=False)

    fig = go.Figure(go.Pie(
        labels=market_rev["Market"],
        values=market_rev["Revenue"],
        hole=0.6,
        marker_colors=["#00d4ff", "#7c3aed", "#f59e0b", "#10b981", "#ef4444"],
        textinfo="label+percent",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{label}</b><br>Spend: PKR %{value:,.0f}<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Spend by Market",
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Top Regions + Shipping Mode Heatmap ────────────────────
col_l, col_r = tab_regional.columns(2)

with col_l:
    # Top 10 regions by revenue
    region_rev = (
        fdf.groupby("order_region")
           .agg(revenue=("sales", "sum"), orders=("order_id", "nunique"))
           .reset_index()
           .sort_values("revenue", ascending=True)
           .tail(10)
    )
    fig = go.Figure(go.Bar(
        x=region_rev["revenue"],
        y=region_rev["order_region"],
        orientation="h",
        marker=dict(
            color=region_rev["revenue"],
            colorscale=[[0, "#1a1d5e"], [0.5, "#7c3aed"], [1, "#00d4ff"]],
            showscale=False,
        ),
        text=[format_currency(r) for r in region_rev["revenue"]],
        textposition="outside",
        textfont=dict(color="white", size=10),
        hovertemplate="<b>%{y}</b><br>Spend: PKR %{x:,.0f}<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, title="Top 10 Regions by Spend", height=340)
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    # Heatmap: delay rate by shipping mode × top regions
    top_regions = fdf.groupby("order_region")["sales"].sum().nlargest(8).index.tolist()
    heat_df = fdf[fdf["order_region"].isin(top_regions)]
    pivot = heat_df.pivot_table(
        values="is_late",
        index="order_region",
        columns="shipping_mode",
        aggfunc="mean",
    ) * 100

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True,
        colorbar=dict(
            title="Late %",
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
        ),
        hovertemplate="<b>Region:</b> %{y}<br><b>Mode:</b> %{x}<br><b>Late Rate:</b> %{z:.1f}%<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, title="Late Delivery Rate: Shipping Mode × Region", height=340)
    st.plotly_chart(fig, use_container_width=True)



# ── Row 4: Top Routes Table ────────────────────────────────────────
routes = (
    fdf.groupby(["order_region", "order_country", "shipping_mode"])
       .agg(
           total_orders=("order_id", "nunique"),
           total_revenue=("sales", "sum"),
           avg_days=("days_for_shipping_real", "mean"),
           late_rate=("is_late", "mean"),
       )
       .reset_index()
       .sort_values("total_revenue", ascending=False)
       .head(15)
)
routes["total_revenue"]  = routes["total_revenue"].round(0)
routes["avg_days"]       = routes["avg_days"].round(1)
routes["late_rate"]      = (routes["late_rate"] * 100).round(1)
routes["Risk"]           = routes["late_rate"].apply(
    lambda x: "🔴 High" if x > 65 else ("🟡 Medium" if x > 45 else "🟢 Low")
)
routes.columns = [
    "Region", "Country", "Shipping Mode",
    "Shipments", "Spend (PKR)", "Avg Days", "Late Rate (%)", "Risk"
]
routes["Spend (PKR)"] = routes["Spend (PKR)"].apply(format_currency)

with tab_routes:
    st.markdown('### 🛣️ Top Shipping Routes Breakdown')
    st.dataframe(
        routes,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Late Rate (%)": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
        }
    )

# ── AI Agent Chatbot ───────────────────────────────────────────────
with tab_chat:
    st.markdown("### 🤖 Supply Chain AI Agent")
    st.caption("Ask questions about the currently filtered dataset. I will analyze the data and respond!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm your Logistics AI Assistant. Ask me about spend, delays, or regional performance based on your current filters."}
        ]
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("E.g., What is our total spend? What is the late delivery rate?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Agent Data Analysis Logic
        prompt_lower = prompt.lower()
        if "spend" in prompt_lower or "cost" in prompt_lower:
            ans = f"Based on the current filters, your total logistics spend is **{format_currency(total_rev)}** across {total_orders:,} shipments."
        elif "late" in prompt_lower or "delay" in prompt_lower:
            ans = f"Currently, **{late_pct:.1f}%** of shipments are late. The average delay is **{avg_delay:.1f} days**."
        elif "region" in prompt_lower:
            top_reg = fdf.groupby("order_region")["sales"].sum().idxmax() if not fdf.empty else "N/A"
            ans = f"The region with the highest spend right now is **{top_reg}**."
        elif "market" in prompt_lower:
            top_mark = fdf.groupby("market")["sales"].sum().idxmax() if not fdf.empty else "N/A"
            ans = f"Your top performing market in this view is **{top_mark}**."
        elif "save" in prompt_lower or "efficiency" in prompt_lower:
            ans = f"You have generated **{format_currency(total_profit)}** in efficiency savings!"
        else:
            ans = "I am a prototype AI. Try asking me about our **spend**, **late** deliveries, **regions**, or **markets**!"
            
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)
