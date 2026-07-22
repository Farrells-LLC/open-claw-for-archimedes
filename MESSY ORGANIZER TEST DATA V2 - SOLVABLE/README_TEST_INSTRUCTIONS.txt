Messy ecommerce organizer test pack v2.

Recommended organizer instruction:
Merge all files into one clean ecommerce dataset for stockout, lost revenue, refund, customer, product, and inventory analysis. Use order_id to connect refunds to orders, customer_id to connect customers to orders, and sku to connect products and weekly_inventory to orders. Preserve all uploaded files; do not silently drop weekly_inventory.tsv, refunds_export.json, or either customer file. If weekly inventory is at sku-week grain, aggregate inventory by sku before joining to order-level rows and keep clear inventory field names. Preserve leading zeros in customer IDs.

Expected useful output columns include order_id, customer_id, sku, product/category, region/channel, units_sold, stockout_flag, lost_revenue_estimate, gross_margin_dollars, refund fields, and inventory/demand fields.
