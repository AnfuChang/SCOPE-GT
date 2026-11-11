"""Relation-evidence diagnostic (Fig. 1a, protocol R_information_v3).

Five paired variants that differ only in the original-graph relations injected into the keys and values
read by the center: Plain, Semantics (common-neighbor attributes, mode name ``witness``), Strength,
Neighborhood and All.

motivation.py    experiment entry point (training, evaluation, CSV export)
information.py   relation channels and the reader model
graph.py         graph loading and lookup tables
relation.py      uniform context sampling inside h-hop balls
"""
