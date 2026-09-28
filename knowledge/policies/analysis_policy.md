---
document_type: policy
department: analytics
version: 1.0
access_level: internal
effective_date: 2024-01-01
---
# Analysis Policy

## Computation
All reported numbers must come from deterministic computation (SQL or statistical tools), never
from model estimation. Critical totals should be cross-checked with an independent query.

## Causality
Descriptive data can show where a change happened, not why. Causal explanations that are not
directly observable in the data must be labelled as hypotheses and paired with a proposed test.

## Data modification
Analysts must not modify source data. Any cleaning operation that deletes or updates rows is
applied only to an investigation's working copy and requires explicit human approval.
