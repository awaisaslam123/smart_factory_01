-- =============================================================
-- Smart Supply Chain Database Schema
-- Dubai E-Commerce Logistics | SQLite-compatible DDL
-- =============================================================

-- Drop tables if they exist (for re-runs)
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

-- ---------------------------------------------------------------
-- Table 1: CUSTOMERS
-- Stores customer demographic and geographic data
-- ---------------------------------------------------------------
CREATE TABLE customers (
    customer_id         INTEGER PRIMARY KEY,
    customer_fname      TEXT NOT NULL,
    customer_lname      TEXT NOT NULL,
    customer_email      TEXT,
    customer_segment    TEXT NOT NULL,   -- Consumer, Corporate, Home Office
    customer_city       TEXT,
    customer_state      TEXT,
    customer_country    TEXT NOT NULL,
    customer_zipcode    TEXT,
    latitude            REAL,
    longitude           REAL
);

-- ---------------------------------------------------------------
-- Table 2: PRODUCTS
-- Stores product catalog and category information
-- ---------------------------------------------------------------
CREATE TABLE products (
    product_id          INTEGER PRIMARY KEY,
    product_name        TEXT NOT NULL,
    category_id         INTEGER,
    category_name       TEXT NOT NULL,
    department_id       INTEGER,
    department_name     TEXT NOT NULL,
    product_price       REAL NOT NULL,
    product_status      INTEGER DEFAULT 0  -- 0=active, 1=discontinued
);

-- ---------------------------------------------------------------
-- Table 3: ORDERS
-- Core fact table linking customers, products, and shipping data
-- ---------------------------------------------------------------
CREATE TABLE orders (
    order_id                    INTEGER NOT NULL,
    order_item_id               INTEGER PRIMARY KEY,
    customer_id                 INTEGER NOT NULL,
    product_id                  INTEGER NOT NULL,
    order_date                  TEXT NOT NULL,   -- ISO format: YYYY-MM-DD HH:MM:SS
    shipping_date               TEXT,
    shipping_mode               TEXT NOT NULL,   -- Standard Class, First Class, Second Class, Same Day
    order_status                TEXT NOT NULL,   -- COMPLETE, PENDING, CLOSED, CANCELED
    delivery_status             TEXT NOT NULL,   -- Advance shipping, Late delivery, Shipping canceled, Shipping on time
    late_delivery_risk          INTEGER NOT NULL, -- 0 = On Time, 1 = Late
    days_for_shipping_real      INTEGER,
    days_for_shipment_scheduled INTEGER,
    order_item_quantity         INTEGER NOT NULL,
    order_item_discount         REAL DEFAULT 0,
    order_item_discount_rate    REAL DEFAULT 0,
    order_item_product_price    REAL NOT NULL,
    order_item_profit_ratio     REAL,
    sales                       REAL NOT NULL,
    benefit_per_order           REAL,
    sales_per_customer          REAL,
    order_region                TEXT NOT NULL,
    order_country               TEXT NOT NULL,
    order_city                  TEXT,
    order_state                 TEXT,
    market                      TEXT,            -- Africa, Europe, LATAM, Pacific Asia, USCA
    type                        TEXT,            -- DEBIT, TRANSFER, PAYMENT, CASH
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id)  REFERENCES products(product_id)
);

-- ---------------------------------------------------------------
-- Indexes for query performance
-- ---------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_orders_customer   ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_product    ON orders(product_id);
CREATE INDEX IF NOT EXISTS idx_orders_region     ON orders(order_region);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders(delivery_status);
CREATE INDEX IF NOT EXISTS idx_orders_date       ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_ship_mode  ON orders(shipping_mode);
