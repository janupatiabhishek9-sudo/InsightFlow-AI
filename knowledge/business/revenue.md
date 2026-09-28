---
document_type: definition
department: finance
version: 2.0
access_level: internal
effective_date: 2024-01-01
---
# Revenue Definition

## Formula
Revenue = quantity x unit price x (1 - discount).
Revenue is recorded net of discounts. Gross revenue is quantity x unit price before discounts.

## Recognition
Revenue is recognised on the order date. All monetary values are reported in USD.

## Returns
Returns are booked as order lines with negative quantity and negative revenue. They reduce
revenue in the period in which the return is recorded and should not be removed from revenue
analyses unless the analyst explicitly asks for gross sales.

## Revenue change drivers
A revenue change can be explained by four drivers: number of orders, units per order (basket
size), average list price, and effective discount. These drivers multiply to revenue, so their
effects can be separated with a sequential decomposition that sums exactly to the total change.
