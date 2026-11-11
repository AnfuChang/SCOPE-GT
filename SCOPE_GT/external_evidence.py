"""External supporting-node controls with unchanged tokens and original-graph topology.

The structure control removes paths with any intermediate node outside S_c from
connection statistics only. Attribute records, original transition degrees, and
calibration data remain those of the full graph.
"""
import numpy as np


class ExternalPathStatisticsSampler:
    def __init__(self, sampler):
        self.base = sampler

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def cache_signature(self):
        return (*self.base.cache_signature, 'external_path_statistics_off_v1')

    def draw(self, centers, streams, relation_source=None):
        draw = self.base.draw(centers, streams, relation_source)
        return self.mask_statistics(centers, draw)

    def mask_statistics(self, centers, draw):
        # Keep moment_count/moment_mass intact: fit_statistic_scales and the
        # attribute means use the same full-graph data in both controls.
        out = dict(draw)
        shape = draw['ctx'].shape
        count, mass = np.zeros(shape), np.zeros(shape)
        positions = np.zeros(shape + (max(self.hops - 1, 1),), np.float32)
        graph = self.graph
        for row, center in enumerate(np.asarray(centers, np.int64)):
            cols = np.flatnonzero(draw['ctx_valid'][row])
            if not len(cols):
                continue
            nodes = np.r_[center, draw['ctx'][row, cols]]
            size = len(nodes)
            adjacency = graph.has_edge(np.repeat(nodes, size), np.tile(nodes, size)).reshape(size, size)
            adjacency = adjacency.astype(np.float64)
            # Preserve original-graph degrees; renormalizing inside the sequence
            # would change the mass of retained paths as well as removing paths.
            transition = adjacency / np.maximum(graph.degree[nodes, None], 1)
            counts2, masses2 = adjacency[0] @ adjacency, transition[0] @ transition
            distance = draw['hop'][row, cols]
            short = (distance <= 2) & draw['connection_known'][row, cols]
            count[row, cols[short]] = counts2[1:][short]
            mass[row, cols[short]] = masses2[1:][short]
            long = distance == 3
            if np.any(long):
                count[row, cols[long]] = (counts2 @ adjacency)[1:][long]
                mass[row, cols[long]] = (masses2 @ transition)[1:][long]
            positions[row, cols, 0] = count[row, cols] > 0
            if positions.shape[-1] > 1:
                positions[row, cols, 1] = long & (count[row, cols] > 0)
            out['edge_queries'] = out.get('edge_queries', 0) + size * size
        out.update(connection_count_used=count, connection_mass_used=mass,
                   connection_positions_used=positions)
        return out
