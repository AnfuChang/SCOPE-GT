"""SCOPE-GT (Shared Context with Original-graph Pair Evidence) for node classification.

Sampled contexts (sampler.py) -> relation evidence per (center, context) pair: topology,
shortest-path connection and background (sampler.py, relation.py) -> center-query attention
with typed Key/Value residuals over a packed batch (batch.py, model.py) -> train.py.
"""
