-- =============================================================
-- Smart Supply Chain - Data Loading Reference
-- NOTE: Data is loaded via Python (pipeline/etl.py) using
--       pandas .to_sql() for efficiency with 50,000 records.
--       This file documents the loading logic for reference.
-- =============================================================

-- Step 1: Verify data was loaded correctly
SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL
SELECT 'products',  COUNT(*) FROM products
UNION ALL
SELECT 'orders',    COUNT(*) FROM orders;

-- Step 2: Check for any NULL values in critical columns
SELECT
    SUM(CASE WHEN customer_id       IS NULL THEN 1 ELSE 0 END) AS null_customer_id,
    SUM(CASE WHEN order_date        IS NULL THEN 1 ELSE 0 END) AS null_order_date,
    SUM(CASE WHEN delivery_status   IS NULL THEN 1 ELSE 0 END) AS null_delivery_status,
    SUM(CASE WHEN late_delivery_risk IS NULL THEN 1 ELSE 0 END) AS null_late_risk,
    SUM(CASE WHEN sales             IS NULL THEN 1 ELSE 0 END) AS null_sales
FROM orders;

-- Step 3: Verify referential integrity
SELECT COUNT(*) AS orphan_orders
FROM orders o
WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id);

SELECT COUNT(*) AS orphan_order_items
FROM orders o
WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = o.product_id);

-- Step 4: Sample data check
SELECT
    o.order_id,
    c.customer_fname || ' ' || c.customer_lname AS customer_name,
    c.customer_country,
    p.product_name,
    p.category_name,
    o.shipping_mode,
    o.delivery_status,
    o.sales
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
LIMIT 10;
