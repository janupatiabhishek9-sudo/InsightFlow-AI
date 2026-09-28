---
document_type: data_documentation
department: data
version: 1.2
access_level: internal
effective_date: 2024-01-01
---
# Sales Schema Documentation

## Grain
One row per order line. There is no customer identifier, so customer-level behaviour (churn,
new versus returning customers) cannot be analysed from this table.

## Known data-quality issues
- Occasional duplicate rows caused by double-loaded batches from the APAC ERP.
- Some early APAC rows have a missing discount; their revenue was booked at list price.
- Returns appear as negative quantity and revenue.
- Country names from one source system may carry trailing whitespace.
