import pandas as pd
from src.db import get_engine

ORDERS_SQL = """
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state                         AS state,
    o.order_status,
    o.order_purchase_timestamp               AS purchase_ts,
    o.order_approved_at                      AS approved_ts,
    o.order_delivered_customer_date          AS delivered_ts,
    o.order_estimated_delivery_date          AS estimated_ts,
    it.n_items, it.n_sellers, it.items_value, it.freight_value, it.main_category,
    it.avg_weight_g, it.avg_photos, it.avg_desc_len,
    pay.payment_value, pay.installments, pay.payment_type, pay.n_payments, pay.used_voucher,
    rev.review_score, rev.has_comment, rev.comment_len
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN (
    SELECT i.order_id,
           COUNT(*)                          AS n_items,
           COUNT(DISTINCT i.seller_id)       AS n_sellers,
           SUM(i.price)                      AS items_value,
           SUM(i.freight_value)              AS freight_value,
           AVG(p.product_weight_g)           AS avg_weight_g,
           AVG(p.product_photos_qty)         AS avg_photos,
           AVG(p.product_description_lenght) AS avg_desc_len,
           (ARRAY_AGG(COALESCE(t.product_category_name_english, 'unknown')
                      ORDER BY i.order_item_id))[1] AS main_category
    FROM order_items i
    LEFT JOIN products p ON p.product_id = i.product_id
    LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
    GROUP BY i.order_id
) it ON it.order_id = o.order_id
LEFT JOIN (
    SELECT order_id,
           SUM(payment_value)                AS payment_value,
           MAX(payment_installments)         AS installments,
           COUNT(*)                          AS n_payments,
           (BOOL_OR(payment_type = 'voucher'))::int AS used_voucher,
           (ARRAY_AGG(payment_type ORDER BY payment_sequential))[1] AS payment_type
    FROM order_payments
    GROUP BY order_id
) pay ON pay.order_id = o.order_id
LEFT JOIN (
    SELECT order_id,
           AVG(review_score)                 AS review_score,
           MAX((COALESCE(review_comment_message, '') <> '')::int) AS has_comment,
           MAX(LENGTH(COALESCE(review_comment_message, '')))      AS comment_len
    FROM order_reviews
    GROUP BY order_id
) rev ON rev.order_id = o.order_id
"""

def load_orders() -> pd.DataFrame:
    """One row per order, pulled from PostgreSQL."""
    df = pd.read_sql(
        ORDERS_SQL,
        get_engine(),
        parse_dates=["purchase_ts", "approved_ts", "delivered_ts", "estimated_ts"],
    )
    return df.sort_values("purchase_ts").reset_index(drop=True)