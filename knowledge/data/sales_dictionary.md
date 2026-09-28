---
document_type: data_dictionary
department: data
version: 1.4
access_level: public
effective_date: 2024-01-01
---
# Sales Data Dictionary

## Columns
| column | description |
|---|---|
| order_id | Unique identifier of an order line |
| order_date | Date the order was placed (revenue recognition date) |
| region | Sales region: Europe, North America or APAC |
| country | Customer country |
| customer_segment | Customer segment: Consumer, SMB or Enterprise |
| product_category | Product category |
| product | Product name |
| quantity | Units ordered; negative for returns |
| unit_price | List price per unit in USD before discount |
| discount | Discount rate between 0 and 1 |
| revenue | Net revenue in USD = quantity x unit_price x (1 - discount) |
| cost | Cost of goods sold in USD |
| profit | revenue - cost in USD |
