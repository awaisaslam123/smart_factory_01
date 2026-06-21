"""
Supply Chain Synthetic Data Generator
======================================
Generates 50,000 realistic supply chain orders matching
the DataCo Smart Supply Chain dataset schema.

Saves:
  - data/raw/supply_chain_raw.csv  (flat CSV)
  - supply_chain.db                (SQLite database with 3 normalized tables)
"""

import sqlite3
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Seed for reproducibility ──────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
DB_PATH     = BASE_DIR / "supply_chain.db"
SQL_DDL     = BASE_DIR / "sql" / "01_create_tables.sql"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference Data ────────────────────────────────────────────────
MARKETS = {
    "Punjab":      ["Lahore", "Faisalabad", "Multan", "Gujranwala", "Sialkot"],
    "Sindh":       ["Karachi", "Hyderabad", "Sukkur", "Larkana", "Mirpur Khas"],
    "KPK":         ["Peshawar", "Mardan", "Abbottabad", "Swat", "Kohat"],
    "Balochistan": ["Quetta", "Gwadar", "Khuzdar", "Turbat", "Chaman"],
    "Federal":     ["Islamabad", "Rawalpindi", "Gilgit", "Muzaffarabad"],
}

REGIONS = {
    "Punjab":      ["Central Punjab", "South Punjab", "North Punjab"],
    "Sindh":       ["Urban Sindh", "Rural Sindh"],
    "KPK":         ["North KPK", "South KPK"],
    "Balochistan": ["North Balochistan", "South Balochistan"],
    "Federal":     ["Capital Territory", "Northern Areas"],
}

SHIPPING_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
SHIP_DAYS = {
    "Standard Class": (4, 7),
    "Second Class":   (3, 5),
    "First Class":    (2, 3),
    "Same Day":       (0, 1),
}

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]
ORDER_STATUSES    = ["COMPLETE", "PENDING", "CLOSED", "CANCELED", "PENDING_PAYMENT"]
PAYMENT_TYPES     = ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]

DEPARTMENTS_CATEGORIES = {
    "Electronics":    ["Mobiles & Tablets", "Cameras", "Laptops", "Audio"],
    "Clothing":       ["Men's Fashion", "Women's Fashion", "Footwear", "Accessories"],
    "Home & Garden":  ["Furniture", "Kitchen", "Garden Tools", "Bedding"],
    "Sports":         ["Fitness Equipment", "Outdoor Sports", "Team Sports"],
    "Automotive":     ["Car Accessories", "Tools & Equipment", "GPS & Navigation"],
    "Books & Media":  ["Books", "Music", "Movies & TV"],
    "Health":         ["Vitamins", "Personal Care", "Medical Supplies"],
    "Food":           ["Grocery", "Beverages", "Snacks"],
}

FIRST_NAMES = [
    "Mohammed", "Ahmed", "Ali", "Omar", "Usman", "Sarah", "Fatima", "Aisha",
    "Zainab", "Maryam", "Bilal", "Hamza", "Hassan", "Hussain", "Saad", "Tariq",
    "Amna", "Khadija", "Sana", "Iqra", "Imran", "Kamran", "Salman", "Nida",
]
LAST_NAMES = [
    "Khan", "Ali", "Ahmed", "Hussain", "Mahmood", "Tariq", "Javed",
    "Qureshi", "Malik", "Raza", "Sheikh", "Shah", "Baig", "Iqbal",
    "Chaudhry", "Butt", "Nawaz", "Akhtar", "Zaman", "Rehman",
]

# ── Helper Functions ──────────────────────────────────────────────

def random_date(start_year=2021, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days),
                             hours=random.randint(0, 23),
                             minutes=random.randint(0, 59))

def delivery_status_from_risk(risk: int, scheduled: int, actual: int) -> str:
    if risk == 0:
        if actual < scheduled:
            return "Advance shipping"
        return "Shipping on time"
    else:
        if actual > scheduled + 3:
            return "Late delivery"
        return "Late delivery"

def compute_late_risk(shipping_mode: str, market: str, scheduled: int, actual: int) -> int:
    """Business-rule-based late delivery risk."""
    # Base rate by market
    market_base = {
        "Federal": 0.72, "Balochistan": 0.64,
        "KPK": 0.59, "Sindh": 0.52, "Punjab": 0.48,
    }
    # Shipping mode penalty
    mode_penalty = {
        "Standard Class": 0.15, "Second Class": 0.08,
        "First Class": -0.05, "Same Day": -0.15,
    }
    prob = market_base.get(market, 0.55) + mode_penalty.get(shipping_mode, 0)
    prob = min(max(prob, 0.1), 0.95)
    return 1 if random.random() < prob else 0


# ── Generate Master Customer List ─────────────────────────────────
print("🔧 Generating customers...")
N_CUSTOMERS = 5000
customers = []
for cid in range(1, N_CUSTOMERS + 1):
    market   = random.choice(list(MARKETS.keys()))
    country  = random.choice(MARKETS[market])
    segment  = random.choice(CUSTOMER_SEGMENTS)
    fname    = random.choice(FIRST_NAMES)
    lname    = random.choice(LAST_NAMES)
    customers.append({
        "customer_id":       cid,
        "customer_fname":    fname,
        "customer_lname":    lname,
        "customer_email":    f"{fname.lower()}.{lname.lower().replace('-','')}{cid}@email.com",
        "customer_segment":  segment,
        "customer_city":     f"City_{cid % 200}",
        "customer_state":    f"State_{cid % 50}",
        "customer_country":  country,
        "customer_zipcode":  f"{random.randint(10000, 99999)}",
        "latitude":          round(random.uniform(-60, 70), 4),
        "longitude":         round(random.uniform(-180, 180), 4),
    })
customers_df = pd.DataFrame(customers)

# ── Generate Product Catalog ──────────────────────────────────────
print("🔧 Generating products...")
N_PRODUCTS = 500
products = []
pid = 1
for dept, categories in DEPARTMENTS_CATEGORIES.items():
    for cat in categories:
        for _ in range(N_PRODUCTS // (len(DEPARTMENTS_CATEGORIES) * 4) + 1):
            price = round(random.uniform(1500, 300000), 2)
            products.append({
                "product_id":      pid,
                "product_name":    f"{cat} Product #{pid}",
                "category_id":     hash(cat) % 50 + 1,
                "category_name":   cat,
                "department_id":   hash(dept) % 20 + 1,
                "department_name": dept,
                "product_price":   price,
                "product_status":  0,
            })
            pid += 1
            if pid > N_PRODUCTS:
                break
        if pid > N_PRODUCTS:
            break
products_df = pd.DataFrame(products[:N_PRODUCTS])

# ── Generate Orders ───────────────────────────────────────────────
print("🔧 Generating 50,000 orders...")
N_ORDERS = 50000
orders = []

for i in range(1, N_ORDERS + 1):
    cust      = customers_df.sample(1).iloc[0]
    prod      = products_df.sample(1).iloc[0]
    market    = [m for m, countries in MARKETS.items() if cust["customer_country"] in countries]
    market    = market[0] if market else "USCA"
    region    = random.choice(REGIONS[market])
    ship_mode = random.choice(SHIPPING_MODES)

    sched_days = random.randint(*SHIP_DAYS[ship_mode])
    # Actual days: sometimes better, often worse
    drift = random.choices([-1, 0, 1, 2, 3, 4, 5], weights=[5, 25, 20, 20, 15, 10, 5])[0]
    actual_days = max(0, sched_days + drift)

    late_risk = compute_late_risk(ship_mode, market, sched_days, actual_days)
    if late_risk == 1:
        actual_days = max(sched_days + 1, actual_days)

    order_dt   = random_date()
    ship_dt    = order_dt + timedelta(days=actual_days)
    order_status = random.choices(
        ORDER_STATUSES,
        weights=[55, 15, 15, 10, 5]
    )[0]

    qty   = random.randint(1, 5)
    price = float(prod["product_price"])
    disc_rate = round(random.uniform(0, 0.35), 3)
    discount  = round(price * disc_rate * qty, 2)
    sales     = round(price * qty * (1 - disc_rate), 2)
    profit_ratio = round(random.uniform(-0.1, 0.45), 4)
    benefit   = round(sales * profit_ratio, 2)

    delivery_st = delivery_status_from_risk(late_risk, sched_days, actual_days)
    if order_status == "CANCELED":
        delivery_st = "Shipping canceled"
        late_risk   = 0

    orders.append({
        "order_id":                    random.randint(10000, 99999),
        "order_item_id":               i,
        "customer_id":                 int(cust["customer_id"]),
        "product_id":                  int(prod["product_id"]),
        "order_date":                  order_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "shipping_date":               ship_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "shipping_mode":               ship_mode,
        "order_status":                order_status,
        "delivery_status":             delivery_st,
        "late_delivery_risk":          late_risk,
        "days_for_shipping_real":      actual_days,
        "days_for_shipment_scheduled": sched_days,
        "order_item_quantity":         qty,
        "order_item_discount":         discount,
        "order_item_discount_rate":    disc_rate,
        "order_item_product_price":    price,
        "order_item_profit_ratio":     profit_ratio,
        "sales":                       sales,
        "benefit_per_order":           benefit,
        "sales_per_customer":          round(sales * random.uniform(0.8, 2.5), 2),
        "order_region":                region,
        "order_country":               cust["customer_country"],
        "order_city":                  cust["customer_city"],
        "order_state":                 cust["customer_state"],
        "market":                      market,
        "type":                        random.choice(PAYMENT_TYPES),
    })

    if i % 10000 == 0:
        print(f"  → Generated {i:,} / {N_ORDERS:,} orders...")

orders_df = pd.DataFrame(orders)

# ── Save raw CSV ──────────────────────────────────────────────────
print("\n💾 Saving raw CSV...")
# Merge for flat CSV (mimicking DataCo format)
flat = orders_df.copy()
flat = flat.merge(
    customers_df[["customer_id","customer_fname","customer_lname",
                  "customer_segment","customer_country","latitude","longitude"]],
    on="customer_id", how="left"
)
flat = flat.merge(
    products_df[["product_id","product_name","category_name","department_name","product_price"]],
    on="product_id", how="left"
)
flat.to_csv(RAW_DIR / "supply_chain_raw.csv", index=False)
print(f"  ✅ Saved {len(flat):,} rows → data/raw/supply_chain_raw.csv")

# ── Create SQLite Database ────────────────────────────────────────
print("\n🗄️  Creating SQLite database...")
DB_PATH.unlink(missing_ok=True)
conn = sqlite3.connect(DB_PATH)

# Execute DDL
with open(SQL_DDL, "r") as f:
    conn.executescript(f.read())

# Load tables
customers_df.to_sql("customers", conn, if_exists="append", index=False)
products_df.to_sql("products",  conn, if_exists="append", index=False)
orders_df.to_sql("orders",      conn, if_exists="append", index=False)

conn.commit()
conn.close()
print(f"  ✅ SQLite DB created → supply_chain.db")

# ── Verification ──────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
for tbl in ["customers", "products", "orders"]:
    count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {tbl}", conn).iloc[0, 0]
    print(f"  📊 {tbl}: {count:,} rows")

# Show delivery status distribution
dist = pd.read_sql("""
    SELECT delivery_status, COUNT(*) as cnt,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(), 1) as pct
    FROM orders GROUP BY delivery_status ORDER BY cnt DESC
""", conn)
print("\n📦 Delivery Status Distribution:")
print(dist.to_string(index=False))
conn.close()

print("\n✅ Data generation complete! Ready for ETL pipeline.")
