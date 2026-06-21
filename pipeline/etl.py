"""
Phase 2: ETL Pipeline
======================
Connects to SQLite via SQLAlchemy, extracts data using the analytical
SQL queries, cleans the data, engineers new features, and saves the
processed dataset for ML training and dashboard use.

Run: python pipeline/etl.py
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DB_PATH       = BASE_DIR / "supply_chain.db"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Connect to SQLite via SQLAlchemy ───────────────────────
print("=" * 60)
print("  SUPPLY CHAIN ETL PIPELINE")
print("=" * 60)
print("\n[STEP 1] Connecting to database via SQLAlchemy...")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM orders")).fetchone()
        print(f"  ✅ Connected! Orders in DB: {result[0]:,}")
except Exception as e:
    print(f"  ❌ Connection failed: {e}")
    print("  → Run 'python data/generate_data.py' first!")
    raise SystemExit(1)

# ── Step 2: Extract via SQL JOIN (Analytical Query) ────────────────
print("\n[STEP 2] Extracting data via SQL JOIN across 3 tables...")

EXTRACT_SQL = """
SELECT
    o.order_id,
    o.order_item_id,
    o.order_date,
    o.shipping_date,
    o.shipping_mode,
    o.order_status,
    o.delivery_status,
    o.late_delivery_risk,
    o.days_for_shipping_real,
    o.days_for_shipment_scheduled,
    o.order_item_quantity,
    o.order_item_discount_rate,
    o.order_item_product_price,
    o.order_item_profit_ratio,
    o.sales,
    o.benefit_per_order,
    o.sales_per_customer,
    o.order_region,
    o.order_country,
    o.market,
    o.type AS payment_type,
    -- Customer fields
    c.customer_segment,
    c.customer_country,
    c.latitude,
    c.longitude,
    -- Product fields
    p.category_name,
    p.department_name,
    p.product_price
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE o.order_status != 'CANCELED'
"""

df = pd.read_sql(EXTRACT_SQL, engine)
print(f"  ✅ Extracted {len(df):,} records with {df.shape[1]} columns")

# ── Step 3: Data Cleaning ──────────────────────────────────────────
print("\n[STEP 3] Cleaning data...")

original_len = len(df)

# 3a. Remove duplicates
df.drop_duplicates(subset=["order_item_id"], inplace=True)
print(f"  → Duplicates removed: {original_len - len(df):,}")

# 3b. Parse datetime columns
df["order_date"]    = pd.to_datetime(df["order_date"],    errors="coerce")
df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")
print(f"  → Parsed order_date and shipping_date to datetime")

# 3c. Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

missing_before = df.isnull().sum().sum()

for col in num_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

missing_after = df.isnull().sum().sum()
print(f"  → Missing values: {missing_before} → {missing_after}")

# 3d. Fix negative sales (data quality issue)
negative_sales = (df["sales"] < 0).sum()
df.loc[df["sales"] < 0, "sales"] = df.loc[df["sales"] < 0, "sales"].abs()
print(f"  → Fixed {negative_sales} negative sales values")

# 3e. Clip extreme outliers (99th percentile)
for col in ["sales", "benefit_per_order", "sales_per_customer"]:
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper)
print(f"  → Clipped outliers in sales columns")

print(f"  ✅ Clean dataset: {len(df):,} records")

# ── Step 4: Feature Engineering ───────────────────────────────────
print("\n[STEP 4] Engineering new features...")

# FE 1: Shipping delay (actual - scheduled)
df["shipping_delay_days"] = (
    df["days_for_shipping_real"] - df["days_for_shipment_scheduled"]
)
print("  → [FE1] shipping_delay_days (actual − scheduled)")

# FE 2: Order hour and day of week
df["order_hour"]       = df["order_date"].dt.hour
df["order_day_of_week"] = df["order_date"].dt.dayofweek   # 0=Mon, 6=Sun
df["order_month"]      = df["order_date"].dt.month
df["order_year"]       = df["order_date"].dt.year
print("  → [FE2] order_hour, order_day_of_week, order_month, order_year")

# FE 3: Weekend flag
df["is_weekend_order"] = (df["order_day_of_week"] >= 5).astype(int)
print("  → [FE3] is_weekend_order (1=Sat/Sun)")

# FE 4: Revenue per unit
df["revenue_per_unit"] = (df["sales"] / df["order_item_quantity"].clip(lower=1)).round(2)
print("  → [FE4] revenue_per_unit")

# FE 5: Discount flag
df["has_discount"] = (df["order_item_discount_rate"] > 0).astype(int)
print("  → [FE5] has_discount")

# FE 6: Profit flag (is order profitable?)
df["is_profitable"] = (df["benefit_per_order"] > 0).astype(int)
print("  → [FE6] is_profitable")

# FE 7: Scheduled days category
df["scheduled_days_bin"] = pd.cut(
    df["days_for_shipment_scheduled"],
    bins=[-1, 1, 3, 5, 10, 100],
    labels=["same_day", "1-3_days", "3-5_days", "5-10_days", "10+_days"]
)
print("  → [FE7] scheduled_days_bin")

# FE 8: Market risk score (based on historical late rate)
market_risk = df.groupby("market")["late_delivery_risk"].mean().rename("market_risk_score")
df = df.merge(market_risk, on="market", how="left")
print("  → [FE8] market_risk_score (market-level late rate)")

# FE 9: Region risk score
region_risk = df.groupby("order_region")["late_delivery_risk"].mean().rename("region_risk_score")
df = df.merge(region_risk, on="order_region", how="left")
print("  → [FE9] region_risk_score")

# FE 10: Shipping mode risk score
mode_risk = df.groupby("shipping_mode")["late_delivery_risk"].mean().rename("mode_risk_score")
df = df.merge(mode_risk, on="shipping_mode", how="left")
print("  → [FE10] mode_risk_score")

print(f"\n  ✅ Feature engineering complete. Total columns: {df.shape[1]}")

# ── Step 5: Save Processed Data ────────────────────────────────────
print("\n[STEP 5] Saving processed data...")

# Save full clean dataset
out_path = PROCESSED_DIR / "clean_data.csv"
df.to_csv(out_path, index=False)
print(f"  ✅ Full clean dataset → {out_path}")

# Save ML-ready features (subset for training)
ML_FEATURES = [
    "shipping_mode", "order_region", "market", "category_name",
    "department_name", "customer_segment", "payment_type",
    "days_for_shipment_scheduled", "order_item_quantity",
    "order_item_discount_rate", "order_item_product_price",
    "sales", "revenue_per_unit", "is_weekend_order",
    "has_discount", "order_hour", "order_month",
    "market_risk_score", "region_risk_score", "mode_risk_score",
    "late_delivery_risk"  # target
]

ml_df = df[ML_FEATURES].dropna()
ml_path = PROCESSED_DIR / "ml_features.csv"
ml_df.to_csv(ml_path, index=False)
print(f"  ✅ ML feature set ({len(ml_df):,} rows × {len(ML_FEATURES)} cols) → {ml_path}")

# Save dashboard data (with geo + dates for viz)
DASH_FEATURES = ML_FEATURES[:-1] + [
    "order_id", "order_date", "shipping_date", "order_country",
    "delivery_status", "benefit_per_order", "order_item_profit_ratio",
    "shipping_delay_days", "days_for_shipping_real",
    "order_year", "order_month", "order_day_of_week",
    "is_profitable", "latitude", "longitude"
]
dash_cols = [c for c in DASH_FEATURES if c in df.columns]
dash_df   = df[dash_cols].dropna(subset=["order_date"])
dash_path = PROCESSED_DIR / "dashboard_data.csv"
dash_df.to_csv(dash_path, index=False)
print(f"  ✅ Dashboard dataset ({len(dash_df):,} rows) → {dash_path}")

# ── Summary Statistics ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ETL PIPELINE SUMMARY")
print("=" * 60)
print(f"  Total records processed : {len(df):,}")
print(f"  Late delivery rate      : {df['late_delivery_risk'].mean():.1%}")
print(f"  Total revenue           : ${df['sales'].sum():,.0f}")
print(f"  Avg shipping delay      : {df['shipping_delay_days'].mean():.2f} days")
print(f"  Date range              : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
print(f"  Markets covered         : {df['market'].nunique()}")
print(f"  Regions covered         : {df['order_region'].nunique()}")
print(f"  Product categories      : {df['category_name'].nunique()}")
print("\n  ✅ ETL complete! Run: python pipeline/train.py")
