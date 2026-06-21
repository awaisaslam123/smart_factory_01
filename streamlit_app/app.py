"""
HuggingFace Spaces - Supply Chain Delay Predictor
==================================================
Single-page Streamlit app for public deployment.
Users input order details and get an instant delay prediction.

Deploy at: huggingface.co/spaces/<your-username>/supply-chain-predictor
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Delay Predictor | AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0a0e1a; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 100%); }
[data-testid="stSidebar"] { background: #0d1225 !important; }
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    color: white; border: none; border-radius: 12px;
    font-weight: 700; font-size: 1.1rem; padding: 14px 28px;
    transition: all 0.3s ease; width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,212,255,0.3); }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #a0aec0 !important; font-weight: 500; }
div[data-testid="stForm"] { border: 1px solid rgba(0,212,255,0.15); border-radius: 16px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try relative path (HuggingFace) then local project path
    for p in [
        Path("models/supply_chain_model.joblib"),
        Path(__file__).parent.parent / "models" / "supply_chain_model.joblib",
    ]:
        if p.exists():
            return joblib.load(p)
    return None

model_pkg = load_model()

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 24px;">
  <div style="font-size:3.5rem;">🚢</div>
  <h1 style="font-size:2.4rem; font-weight:800;
      background:linear-gradient(135deg,#00d4ff,#7c3aed,#f59e0b);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text; margin:12px 0 8px;">
    Supply Chain Delay Predictor
  </h1>
  <p style="color:#6b7280; font-size:1rem; max-width:560px; margin:0 auto;">
    Enter your shipment details below and our XGBoost AI model will instantly
    predict whether your order is at risk of being delayed.
  </p>
  <div style="display:flex; gap:12px; justify-content:center; margin-top:16px; flex-wrap:wrap;">
    <span style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
          color:#00d4ff;border-radius:20px;padding:4px 14px;font-size:0.8rem;">
      ⚡ XGBoost Model
    </span>
    <span style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);
          color:#7c3aed;border-radius:20px;padding:4px 14px;font-size:0.8rem;">
      🧠 50,000 Training Records
    </span>
    <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
          color:#f59e0b;border-radius:20px;padding:4px 14px;font-size:0.8rem;">
      🌍 Global Supply Chain
    </span>
  </div>
</div>
<hr style="border-color:rgba(0,212,255,0.1); margin:0 0 28px;">
""", unsafe_allow_html=True)

if model_pkg is None:
    st.error("⚠️ Model not found. Run `python pipeline/train.py` first, then copy `models/supply_chain_model.joblib` here.")
    st.info("**Quick Setup:**\n```bash\ncd supply_chain\npip install -r requirements.txt\npython data/generate_data.py\npython pipeline/etl.py\npython pipeline/train.py\n```")
    st.stop()

preprocessor       = model_pkg["preprocessor"]
classifier         = model_pkg["classifier"]
all_features       = model_pkg["feature_names"]
cat_features       = model_pkg["categorical_features"]
num_features       = model_pkg["numeric_features"]
saved_metrics      = model_pkg.get("metrics", {})

# ── Input Form ────────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<h3 style="color:#e2e8f0;font-weight:700;margin-bottom:16px;">📋 Order Details</h3>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            shipping_mode = st.selectbox("Shipping Mode", [
                "Standard Class", "Second Class", "First Class", "Same Day"
            ])
            order_region = st.selectbox("Order Region", [
                "East US", "West US", "Central US", "Canada",
                "Western Europe", "Central Europe", "Southern Europe", "Northern Europe",
                "South America", "Caribbean", "Central America",
                "Southeast Asia", "East Asia", "South Asia", "Oceania",
                "West Africa", "East Africa", "North Africa", "South Africa",
            ])
            market = st.selectbox("Market", [
                "USCA", "Europe", "LATAM", "Pacific Asia", "Africa"
            ])
            category_name = st.selectbox("Product Category", [
                "Mobiles & Tablets", "Cameras", "Laptops", "Audio",
                "Men's Fashion", "Women's Fashion", "Footwear", "Accessories",
                "Furniture", "Kitchen", "Garden Tools", "Bedding",
                "Fitness Equipment", "Outdoor Sports", "Team Sports",
                "Car Accessories", "Books", "Vitamins", "Grocery",
            ])

        with c2:
            department_name = st.selectbox("Department", [
                "Electronics", "Clothing", "Home & Garden",
                "Sports", "Automotive", "Books & Media", "Health", "Food",
            ])
            customer_segment = st.selectbox("Customer Segment", [
                "Consumer", "Corporate", "Home Office"
            ])
            payment_type = st.selectbox("Payment Type", [
                "DEBIT", "TRANSFER", "PAYMENT", "CASH"
            ])
            scheduled_days = st.number_input("Scheduled Shipping Days", min_value=0, max_value=30, value=5)

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            order_qty   = st.number_input("Order Quantity",       min_value=1, max_value=50, value=2)
            product_price = st.number_input("Product Price ($)",  min_value=1.0, max_value=5000.0, value=150.0, step=10.0)
            sales         = st.number_input("Total Sale Value ($)", min_value=1.0, max_value=20000.0, value=300.0, step=10.0)
        with c4:
            discount_rate = st.slider("Discount Rate", 0.0, 0.5, 0.1, 0.01)
            order_hour    = st.slider("Order Hour (0-23)",  0, 23, 14)
            order_month   = st.slider("Order Month (1-12)", 1, 12, 6)

        is_weekend = st.checkbox("Weekend Order (Saturday / Sunday)", value=False)
        submitted  = st.form_submit_button("🔮 Predict Delay Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────
with col_result:
    st.markdown('<h3 style="color:#e2e8f0;font-weight:700;margin-bottom:16px;">🎯 Prediction Result</h3>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div style="background:rgba(26,29,46,0.6);border:1px dashed rgba(0,212,255,0.2);
             border-radius:16px;padding:48px 32px;text-align:center;">
          <div style="font-size:3rem;margin-bottom:16px;">🤖</div>
          <p style="color:#6b7280;font-size:1rem;">
            Fill in the order details on the left and click<br>
            <strong style="color:#00d4ff;">Predict Delay Risk</strong> to get your result.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build feature vector
        rev_per_unit    = round(sales / max(order_qty, 1), 2)
        has_discount    = 1 if discount_rate > 0 else 0
        is_profitable   = 1

        # Market risk approximations (from training data patterns)
        market_risk_map = {
            "Africa": 0.72, "LATAM": 0.64, "Pacific Asia": 0.59,
            "Europe": 0.52, "USCA": 0.48,
        }
        mode_risk_map = {
            "Standard Class": 0.63, "Second Class": 0.56,
            "First Class": 0.48, "Same Day": 0.35,
        }
        region_risk_map = {
            "West Africa": 0.74, "East Africa": 0.72, "North Africa": 0.70,
            "South Africa": 0.69, "South America": 0.65, "Caribbean": 0.63,
            "Central America": 0.62, "Southeast Asia": 0.60, "East Asia": 0.58,
            "South Asia": 0.57, "Oceania": 0.55, "Southern Europe": 0.54,
            "Central Europe": 0.52, "Western Europe": 0.51, "Northern Europe": 0.50,
            "Central US": 0.50, "Canada": 0.48, "East US": 0.47, "West US": 0.46,
        }

        market_risk  = market_risk_map.get(market, 0.55)
        mode_risk    = mode_risk_map.get(shipping_mode, 0.55)
        region_risk  = region_risk_map.get(order_region, 0.55)

        input_data = pd.DataFrame([{
            "shipping_mode":                shipping_mode,
            "order_region":                 order_region,
            "market":                       market,
            "category_name":                category_name,
            "department_name":              department_name,
            "customer_segment":             customer_segment,
            "payment_type":                 payment_type,
            "days_for_shipment_scheduled":  scheduled_days,
            "order_item_quantity":          order_qty,
            "order_item_discount_rate":     discount_rate,
            "order_item_product_price":     product_price,
            "sales":                        sales,
            "revenue_per_unit":             rev_per_unit,
            "is_weekend_order":             int(is_weekend),
            "has_discount":                 has_discount,
            "order_hour":                   order_hour,
            "order_month":                  order_month,
            "market_risk_score":            market_risk,
            "region_risk_score":            region_risk,
            "mode_risk_score":              mode_risk,
        }])

        for col_name in cat_features:
            if col_name in input_data.columns:
                input_data[col_name] = input_data[col_name].astype(str)

        try:
            X_proc   = preprocessor.transform(input_data[all_features])
            pred     = classifier.predict(X_proc)[0]
            prob     = classifier.predict_proba(X_proc)[0][1]

            if prob > 0.70:
                risk_level = "HIGH RISK"
                risk_icon  = "🔴"
                risk_color = "#ef4444"
                risk_bg    = "rgba(239,68,68,0.08)"
                msg        = "This order has a HIGH probability of being delayed. Consider upgrading to First Class or Same Day shipping, or sourcing from a closer warehouse."
            elif prob > 0.45:
                risk_level = "MEDIUM RISK"
                risk_icon  = "🟡"
                risk_color = "#f59e0b"
                risk_bg    = "rgba(245,158,11,0.08)"
                msg        = "This order has MODERATE delay risk. Monitor closely and have a contingency plan ready."
            else:
                risk_level = "LOW RISK"
                risk_icon  = "🟢"
                risk_color = "#10b981"
                risk_bg    = "rgba(16,185,129,0.08)"
                msg        = "This order is likely to be delivered ON TIME. Continue with the current shipping plan."

            # Result card
            st.markdown(f"""
            <div style="background:{risk_bg};border:2px solid {risk_color}40;border-radius:20px;
                 padding:32px 28px;text-align:center;margin-bottom:20px;">
              <div style="font-size:4rem;">{risk_icon}</div>
              <div style="font-size:2rem;font-weight:800;color:{risk_color};margin:12px 0 4px;">
                {risk_level}
              </div>
              <div style="font-size:3rem;font-weight:800;color:white;margin:8px 0;">
                {prob:.1%}
              </div>
              <div style="color:#6b7280;font-size:0.85rem;">Delay Probability</div>
              <hr style="border-color:{risk_color}30;margin:20px 0;">
              <p style="color:#a0aec0;font-size:0.9rem;line-height:1.6;margin:0;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Delay Probability", "font": {"color": "white", "size": 14}},
                number={"suffix": "%", "font": {"color": "white", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": risk_color, "thickness": 0.25},
                    "bgcolor": "#1a1d2e",
                    "bordercolor": "#2d3561",
                    "steps": [
                        {"range": [0, 45],  "color": "rgba(16,185,129,0.15)"},
                        {"range": [45, 70], "color": "rgba(245,158,11,0.15)"},
                        {"range": [70, 100],"color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=220,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key factors
            st.markdown("""
            <div style="background:rgba(26,29,46,0.6);border:1px solid rgba(255,255,255,0.08);
                 border-radius:12px;padding:16px 20px;">
              <div style="color:#e2e8f0;font-weight:600;margin-bottom:12px;">📊 Risk Factors</div>
            """, unsafe_allow_html=True)

            factors = [
                ("Market Risk Score",   f"{market_risk:.0%}",  market_risk),
                ("Region Risk Score",   f"{region_risk:.0%}",  region_risk),
                ("Mode Risk Score",     f"{mode_risk:.0%}",    mode_risk),
                ("Scheduled Days",      f"{scheduled_days}d",  min(scheduled_days / 10, 1)),
                ("Discount Rate",       f"{discount_rate:.0%}", discount_rate),
            ]
            for label, val, score in factors:
                color = "#ef4444" if score > 0.65 else ("#f59e0b" if score > 0.45 else "#10b981")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                  <span style="color:#a0aec0;font-size:0.85rem;">{label}</span>
                  <span style="color:{color};font-weight:600;font-size:0.9rem;">{val}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

# ── Model Info Footer ─────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
for col_ui, metric, value in [
    (c1, "Model", "XGBoost"),
    (c2, "Training Data", "50,000 orders"),
    (c3, "AUC-ROC", f"{saved_metrics.get('auc_roc', 'N/A')}"),
    (c4, "F1-Score", f"{saved_metrics.get('f1', 'N/A')}"),
]:
    col_ui.markdown(f"""
    <div style="text-align:center;background:rgba(26,29,46,0.4);border-radius:10px;padding:14px;">
      <div style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;">{metric}</div>
      <div style="color:#00d4ff;font-weight:700;font-size:1.1rem;margin-top:4px;">{value}</div>
    </div>""", unsafe_allow_html=True)
