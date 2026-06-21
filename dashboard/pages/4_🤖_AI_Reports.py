"""
Page 3 — AI Report Center (Hugging Face NLP)
Auto-generates executive briefings from live analytics data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime


st.markdown("""
<style>

.report-box {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 1.6rem; margin: 1rem 0; line-height: 1.8;
}
.report-headline {
    font-size: 1.1rem; font-weight: 700;
    padding: .6rem 1rem; border-radius: 8px; margin-bottom: 1rem;
}
.risk-high   { background: rgba(255,77,79,.15); border-left: 4px solid #ff4d4f; }
.risk-medium { background: rgba(250,219,20,.10); border-left: 4px solid #fadb14; }
.risk-low    { background: rgba(82,196,26,.10);  border-left: 4px solid #52c41a; }
.tag { display:inline-block; padding:.2rem .7rem; border-radius:20px;
       font-size:.75rem; font-weight:600; margin:.2rem; }
</style>
""", unsafe_allow_html=True)

BASE         = os.path.join(os.path.dirname(__file__), "..", "..")
HEALTH_CSV   = os.path.join(BASE, "data", "processed", "machine_health.csv")
PREDICT_CSV  = os.path.join(BASE, "data", "processed", "predictions.csv")
REPORT_DIR   = os.path.join(BASE, "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_health():
    if os.path.exists(HEALTH_CSV):
        return pd.read_csv(HEALTH_CSV)
    np.random.seed(7)
    probs = np.random.beta(2, 5, 10)
    return pd.DataFrame({
        "machine_id": [f"M-{i:03d}" for i in range(1,11)],
        "failure_probability": probs.round(4),
        "health_score": ((1-probs)*100).round(1),
        "rul_hours": ((1-probs)*72).astype(int),
        "severity": pd.cut(probs,[0,.25,.5,.75,1],labels=["LOW","MEDIUM","HIGH","CRITICAL"]),
    })

@st.cache_data(ttl=300)
def load_predictions():
    if os.path.exists(PREDICT_CSV):
        cols = ["order_region","predicted_late","delay_probability","shipping_mode"]
        available = pd.read_csv(PREDICT_CSV, nrows=1).columns.tolist()
        use_cols = [c for c in cols if c in available]
        return pd.read_csv(PREDICT_CSV, usecols=use_cols)
    np.random.seed(7)
    regions = ["Western Europe","Central America","Southeast Asia","West Africa","South Asia"]
    modes   = ["Standard Class","Second Class","First Class","Same Day"]
    n = 500
    return pd.DataFrame({
        "order_region":     np.random.choice(regions, n),
        "shipping_mode":    np.random.choice(modes, n),
        "predicted_late":   np.random.choice([0,1], n, p=[0.65,0.35]),
        "delay_probability":np.random.beta(2,4,n).round(4),
    })

health_df = load_health()
pred_df   = load_predictions()

st.markdown("# 🤖 AI Report Center")
st.markdown("**Powered by Hugging Face `flan-t5-base`** · Natural language operational intelligence")
st.divider()

# ── Risk Score Meter ──────────────────────────────────────────────────────────
maint_risk = health_df["failure_probability"].mean()
sc_risk    = pred_df["predicted_late"].mean() if "predicted_late" in pred_df.columns else 0.3
overall    = round((maint_risk * 0.5 + sc_risk * 0.5) * 100, 1)

if overall >= 55:
    risk_class, risk_label = "risk-high",   "🔴 HIGH OPERATIONAL RISK"
elif overall >= 30:
    risk_class, risk_label = "risk-medium", "🟡 ELEVATED RISK"
else:
    risk_class, risk_label = "risk-low",    "🟢 OPERATIONS NORMAL"

st.markdown(f"""
<div class="report-headline {risk_class}">
    {risk_label} &nbsp;|&nbsp; Composite Risk Score: <strong>{overall}/100</strong>
    &nbsp;|&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.metric("🏭 Maintenance Risk",  f"{maint_risk:.1%}")
with c2: st.metric("🚚 Supply Chain Risk", f"{sc_risk:.1%}")
with c3: st.metric("⚠️ Composite Score",   f"{overall}/100")

st.divider()

# ── Generate Report Button ─────────────────────────────────────────────────────
st.markdown("### 📝 Generate AI Executive Briefing")
col_btn, col_mode = st.columns([2,1])
with col_mode:
    report_type = st.selectbox("Report Type", [
        "Full Daily Briefing",
        "Maintenance Only",
        "Supply Chain Only"
    ])

with col_btn:
    generate = st.button("🤖 Generate AI Report", use_container_width=True, type="primary")

if generate or st.session_state.get("auto_generated"):
    st.session_state["auto_generated"] = True

    with st.spinner("🔄 AI is analyzing operational data and generating report..."):
        try:
            from models.nlp_reporter import build_daily_briefing, build_maintenance_report, build_supply_chain_report
            if report_type == "Maintenance Only":
                text = build_maintenance_report(health_df)
                briefing = {"maintenance_summary": text, "date": datetime.now().strftime("%A, %d %B %Y"),
                            "overall_risk_score": overall, "total_alerts": int(len(health_df[health_df["severity"].isin(["CRITICAL","HIGH"])]))}
            elif report_type == "Supply Chain Only":
                text = build_supply_chain_report(pred_df)
                briefing = {"supply_chain_summary": text, "date": datetime.now().strftime("%A, %d %B %Y"),
                            "overall_risk_score": overall, "total_alerts": int(pred_df["predicted_late"].sum() if "predicted_late" in pred_df.columns else 0)}
            else:
                briefing = build_daily_briefing(health_df, pred_df)
            st.session_state["last_briefing"] = briefing
        except Exception as e:
            # Graceful fallback without HF
            critical_m = health_df[health_df["severity"]=="CRITICAL"]
            delay_rate = pred_df["predicted_late"].mean() if "predicted_late" in pred_df.columns else 0.3
            delayed_n  = int(pred_df["predicted_late"].sum()) if "predicted_late" in pred_df.columns else 150

            maint_text = (
                f"The Lahore Textile Plant currently monitors **{len(health_df)} machines** with an average health score of "
                f"**{health_df['health_score'].mean():.1f}/100**. "
                + (f"**{len(critical_m)} machine(s) are in CRITICAL condition** ({', '.join(critical_m['machine_id'].tolist())}) "
                   f"and require immediate maintenance within the next {critical_m['rul_hours'].min()}h." if not critical_m.empty
                   else "No critical machines detected at this time. Continue routine monitoring.")
            )
            sc_text = (
                f"Raw material logistics analysis of **{len(pred_df):,} shipments** across Pakistan shows a predicted delay rate of "
                f"**{delay_rate:.1%}** ({delayed_n:,} shipments at risk). "
                f"High-risk regions should be prioritized for carrier renegotiation and buffer stock."
            )
            briefing = {
                "date": datetime.now().strftime("%A, %d %B %Y"),
                "overall_risk_score": overall,
                "total_alerts": len(critical_m) + delayed_n,
                "maintenance_summary": maint_text,
                "supply_chain_summary": sc_text,
                "executive_headline": risk_label,
                "model_used": "Template Engine (install transformers for AI mode)",
                "generated_at": datetime.now().isoformat()
            }
            st.session_state["last_briefing"] = briefing

if "last_briefing" in st.session_state:
    b = st.session_state["last_briefing"]

    st.markdown(f"### 📅 Briefing — {b.get('date','')}")
    st.markdown(f"**Model:** `{b.get('model_used','Template Engine')}` &nbsp;|&nbsp; "
                f"**Generated:** {b.get('generated_at','')[:19]}")

    if "maintenance_summary" in b:
        st.markdown("#### 🏭 Factory Operations Summary")
        st.markdown(f'<div class="report-box">{b["maintenance_summary"]}</div>', unsafe_allow_html=True)

    if "supply_chain_summary" in b:
        st.markdown("#### 🚚 Supply Chain Intelligence")
        st.markdown(f'<div class="report-box">{b["supply_chain_summary"]}</div>', unsafe_allow_html=True)

    # Save & download
    st.divider()
    json_str = json.dumps(b, indent=2, ensure_ascii=False)
    st.download_button("⬇️ Download Report (JSON)", json_str,
                       file_name=f"briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                       mime="application/json")

st.divider()

# ── Report History ────────────────────────────────────────────────────────────
st.markdown("### 🗂️ Report History")
try:
    from models.nlp_reporter import list_saved_reports
    history = list_saved_reports()
except Exception:
    history = []

if history:
    for rep in history[:10]:
        risk_c = "risk-high" if rep["risk_score"] >= 55 else ("risk-medium" if rep["risk_score"] >= 30 else "risk-low")
        with st.expander(f"📄 {rep['date']} — Risk: {rep['risk_score']}/100 — Alerts: {rep['alerts']}"):
            try:
                with open(rep["path"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "maintenance_summary" in data:
                    st.markdown("**Maintenance:**")
                    st.write(data["maintenance_summary"])
                if "supply_chain_summary" in data:
                    st.markdown("**Supply Chain:**")
                    st.write(data["supply_chain_summary"])
            except Exception:
                st.write("Could not load report.")
else:
    st.info("No saved reports yet. Generate your first report above ↑")
