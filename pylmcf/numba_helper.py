from numba import njit
import numpy as np

@njit
def match_nodes(self_theoretical_nodes,
                WNM_empirical_node_ids,
                max_dist,
                WNM_empirical_spectrum_positions,
                theoretical_spectrum_positions,
                dist_scaling):
        matching_edge_ids = []
        matching_edge_start_nodes = []
        matching_edge_end_nodes = []
        for ii, theoretical_node in enumerate(self_theoretical_nodes):
            for jj, empirical_node in enumerate(WNM_empirical_node_ids):
                dist_val = np.int64(dist_scaling * np.linalg.norm(
                    WNM_empirical_spectrum_positions[:, jj],
                    theoretical_spectrum_positions[:, ii],
                ))
                if dist_val < max_dist:
                    matching_edge_ids.append(
                        self.G.add_edge(empirical_node, theoretical_node, int(dist_val))
                    )
                    matching_edge_start_nodes.append(empirical_node)
                    matching_edge_end_nodes.append(theoretical_node)
        return np.array(matching_edge_ids), np.array(matching_edge_start_nodes), np.array(matching_edge_end_nodes)