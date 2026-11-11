"""Query-role diagnostic (Fig. 3a, protocol A_query_v2).

Full / Partial / Center first-layer query sets with shared token-wise FFNs, trained independently with paired
sampling and initialization; test accuracy is compared across context budgets K.

experiment.py       experiment entry point
gated.py            reader with per-position update gates (query-set membership)
common.py           training loop
"""
