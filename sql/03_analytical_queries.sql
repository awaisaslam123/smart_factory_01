-- =============================================================
-- Smart Supply Chain - Analytical Queries
-- Demonstrates: JOINs, GROUP BY, Window Functions, Subqueries
-- =============================================================

-- ---------------------------------------------------------------
-- QUERY 1: EXECUTIVE SUMMARY (JOIN across all 3 tables)
-- Total revenue, orders, average delay by market
-- ---------------------------------------------------------------
SELECT
    o.market,
    o.order_region,
    COUNT(DISTINCT o.order_id)                          AS total_orders,
    COUNT(o.order_item_id)                              AS total_items,
    ROUND(SUM(o.sales), 2)                              AS total_revenue,
    ROUND(AVG(o.benefit_per_order), 2)                  AS avg_profit_per_order,
    ROUND(AVG(o.days_for_shipping_real), 1)             AS avg_actual_days,
    ROUND(AVG(o.days_for_shipment_scheduled), 1)        AS avg_scheduled_days,
    ROUND(
        100.0 * SUM(o.late_delivery_risk) / COUNT(*), 1
    )                                                   AS late_delivery_pct,
    COUNT(DISTINCT o.customer_id)                       AS unique_customers
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
GROUP BY o.market, o.order_region
ORDER BY total_revenue DESC;

-- ---------------------------------------------------------------
-- QUERY 2: SHIPPING MODE PERFORMANCE (GROUP BY)
-- Compare real vs. scheduled days by shipping mode
-- ---------------------------------------------------------------
SELECT
    o.shipping_mode,
    COUNT(*)                                                AS total_shipments,
    ROUND(AVG(o.days_for_shipping_real), 2)                AS avg_actual_days,
    ROUND(AVG(o.days_for_shipment_scheduled), 2)           AS avg_scheduled_days,
    ROUND(AVG(o.days_for_shipping_real
              - o.days_for_shipment_scheduled), 2)         AS avg_delay_days,
    SUM(o.late_delivery_risk)                              AS late_count,
    ROUND(
        100.0 * SUM(o.late_delivery_risk) / COUNT(*), 1
    )                                                      AS late_pct,
    ROUND(SUM(o.sales), 2)                                 AS total_revenue
FROM orders o
GROUP BY o.shipping_mode
ORDER BY late_pct DESC;

-- ---------------------------------------------------------------
-- QUERY 3: CUSTOMER SEGMENT ANALYSIS (JOIN + GROUP BY)
-- Revenue, loyalty, and risk by customer segment
-- ---------------------------------------------------------------
SELECT
    c.customer_segment,
    c.customer_country,
    COUNT(DISTINCT o.order_id)                             AS total_orders,
    ROUND(SUM(o.sales), 2)                                 AS total_revenue,
    ROUND(AVG(o.sales_per_customer), 2)                    AS avg_sales_per_customer,
    ROUND(
        100.0 * SUM(o.late_delivery_risk) / COUNT(*), 1
    )                                                      AS late_pct,
    ROUND(SUM(o.benefit_per_order), 2)                     AS total_profit
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.customer_segment, c.customer_country
ORDER BY total_revenue DESC
LIMIT 20;

-- ---------------------------------------------------------------
-- QUERY 4: TOP PRODUCT CATEGORIES (JOIN + GROUP BY)
-- Best selling categories by revenue and delay rate
-- ---------------------------------------------------------------
SELECT
    p.department_name,
    p.category_name,
    COUNT(o.order_item_id)                                 AS items_sold,
    ROUND(SUM(o.sales), 2)                                 AS total_revenue,
    ROUND(AVG(o.order_item_profit_ratio), 3)               AS avg_profit_ratio,
    ROUND(
        100.0 * SUM(o.late_delivery_risk) / COUNT(*), 1
    )                                                      AS late_delivery_pct
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.department_name, p.category_name
ORDER BY total_revenue DESC
LIMIT 15;

-- ---------------------------------------------------------------
-- QUERY 5: WINDOW FUNCTION - Running Revenue & Rank
-- Monthly cumulative revenue with rolling rank per region
-- ---------------------------------------------------------------
SELECT
    order_month,
    order_region,
    monthly_revenue,
    ROUND(
        SUM(monthly_revenue) OVER (
            PARTITION BY order_region
            ORDER BY order_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2
    )                                                       AS cumulative_revenue,
    RANK() OVER (
        PARTITION BY order_month
        ORDER BY monthly_revenue DESC
    )                                                       AS revenue_rank_in_month
FROM (
    SELECT
        SUBSTR(o.order_date, 1, 7)     AS order_month,
        o.order_region,
        ROUND(SUM(o.sales), 2)         AS monthly_revenue
    FROM orders o
    GROUP BY order_month, o.order_region
) monthly
ORDER BY order_month, revenue_rank_in_month;

-- ---------------------------------------------------------------
-- QUERY 6: WINDOW FUNCTION - Delay Rate Ranking by Route
-- Rank shipping routes by late delivery percentage
-- ---------------------------------------------------------------
SELECT
    order_region,
    order_country,
    shipping_mode,
    total_shipments,
    late_shipments,
    late_pct,
    RANK() OVER (ORDER BY late_pct DESC)                   AS worst_route_rank,
    RANK() OVER (ORDER BY total_revenue DESC)              AS revenue_rank
FROM (
    SELECT
        o.order_region,
        o.order_country,
        o.shipping_mode,
        COUNT(*)                                           AS total_shipments,
        SUM(o.late_delivery_risk)                         AS late_shipments,
        ROUND(
            100.0 * SUM(o.late_delivery_risk) / COUNT(*), 1
        )                                                  AS late_pct,
        ROUND(SUM(o.sales), 2)                            AS total_revenue
    FROM orders o
    GROUP BY o.order_region, o.order_country, o.shipping_mode
    HAVING COUNT(*) > 50
) routes
ORDER BY worst_route_rank
LIMIT 20;

-- ---------------------------------------------------------------
-- QUERY 7: WINDOW FUNCTION - Customer Lifetime Value (CLV)
-- With row number and percentile ranking
-- ---------------------------------------------------------------
SELECT
    customer_name,
    customer_country,
    customer_segment,
    total_orders,
    total_spend,
    ROW_NUMBER() OVER (ORDER BY total_spend DESC)          AS clv_rank,
    ROUND(
        100.0 * RANK() OVER (ORDER BY total_spend) 
        / COUNT(*) OVER (), 1
    )                                                      AS spend_percentile
FROM (
    SELECT
        c.customer_fname || ' ' || c.customer_lname        AS customer_name,
        c.customer_country,
        c.customer_segment,
        COUNT(DISTINCT o.order_id)                         AS total_orders,
        ROUND(SUM(o.sales), 2)                             AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY o.customer_id, c.customer_fname, c.customer_lname,
             c.customer_country, c.customer_segment
) clv
ORDER BY clv_rank
LIMIT 25;

-- ---------------------------------------------------------------
-- QUERY 8: SUBQUERY - High-Risk Upcoming Orders (for Risk Radar)
-- Orders with > 70% predicted delay probability based on history
-- ---------------------------------------------------------------
SELECT
    o.order_id,
    o.order_date,
    o.order_region,
    o.shipping_mode,
    p.category_name,
    o.sales,
    o.days_for_shipment_scheduled,
    o.delivery_status,
    o.late_delivery_risk,
    high_risk_region.avg_late_pct                          AS region_late_rate
FROM orders o
JOIN products p ON o.product_id = p.product_id
JOIN (
    SELECT
        order_region,
        ROUND(
            100.0 * SUM(late_delivery_risk) / COUNT(*), 1
        )                                                  AS avg_late_pct
    FROM orders
    GROUP BY order_region
    HAVING avg_late_pct > 55
) high_risk_region ON o.order_region = high_risk_region.order_region
WHERE o.late_delivery_risk = 1
ORDER BY high_risk_region.avg_late_pct DESC, o.sales DESC
LIMIT 100;
