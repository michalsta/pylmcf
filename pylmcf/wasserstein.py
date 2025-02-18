import numpy as np
from numba import njit
import pylmcf_cpp


@njit
def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def wasserstein_network(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    sum_intensities = np.sum(intensities1, dtype=np.int64)
    # Create a graph
    nodes_supply = np.zeros(len(intensities1) + len(intensities2) + 2, dtype=np.int64)
    SRC_IDX = 0
    SINK_IDX = len(nodes_supply) - 1
    MAX_COST = np.int64(2**30)
    SCALING_FACTOR = 1 #np.int64(MAX_COST / trash_cost)
    SCALED_TRASH_COST = np.int64(trash_cost * SCALING_FACTOR / 2.0)
    nodes_supply[SRC_IDX] = sum_intensities
    nodes_supply[SINK_IDX] = -sum_intensities

    edge_starts = []
    edge_ends = []
    edge_capacities = []
    edge_costs = []

    LAYER1_START_IDX = 1
    LAYER2_START_IDX = len(intensities1) + 1

    # The intensity-carrying edges:
    # Add edges from source to first layer
    for i in range(len(intensities1)):
        # Source to layer 1
        # print("Source to layer 1", SRC_IDX, LAYER1_START_IDX + i)
        edge_starts.append(SRC_IDX)
        edge_ends.append(LAYER1_START_IDX + i)
        edge_capacities.append(intensities1[i])
        edge_costs.append(np.int64(0))

    # Add edges from second layer to sink
    for i in range(len(intensities2)):
        # print("Layer 2 to sink", LAYER2_START_IDX + i, SINK_IDX)
        # Layer 2 to sink
        edge_starts.append(LAYER2_START_IDX + i)
        edge_ends.append(SINK_IDX)
        edge_capacities.append(intensities2[i])
        edge_costs.append(np.int64(0))


    # The matching edges:
    matches = 0
    for i in range(len(intensities1)):
        for j in range(len(intensities2)):
            # print("Layer 1 to layer 2: ", LAYER1_START_IDX + i, LAYER2_START_IDX + j)
            dist_val = dist(X1[i], Y1[i], X2[j], Y2[j])
            if dist_val < trash_cost:
                edge_starts.append(LAYER1_START_IDX + i)
                edge_ends.append(LAYER2_START_IDX + j)
                edge_capacities.append(min(intensities1[i], intensities2[j]))
                edge_costs.append(np.int64(SCALING_FACTOR * dist_val))
                matches += 1

    '''
    # The trash edges:
    # Add edges from first layer to sink
    for i in range(len(intensities1)):
        # Layer 1 to sink
        print("Layer 1 to sink", LAYER1_START_IDX + i, SINK_IDX)
        edge_starts.append(LAYER1_START_IDX + i)
        edge_ends.append(SINK_IDX)
        edge_capacities.append(intensities1[i])
        edge_costs.append(SCALED_TRASH_COST)

    # Add edges from source to second layer
    for i in range(len(intensities2)):
        # Source to layer 2
        print("Source to layer 2", SRC_IDX, LAYER2_START_IDX + i)
        edge_starts.append(SRC_IDX)
        edge_ends.append(LAYER2_START_IDX + i)
        edge_capacities.append(intensities2[i])
        edge_costs.append(SCALED_TRASH_COST)
    '''

    # The trash edges:
    nodes_supply = np.append(nodes_supply, 0)
    for i in range(len(intensities1)):
        # Layer 1 to trash
        # print("Layer 1 to trash", LAYER1_START_IDX + i, len(nodes_supply) - 1)
        edge_starts.append(LAYER1_START_IDX + i)
        edge_ends.append(len(nodes_supply) - 1)
        edge_capacities.append(intensities1[i])
        edge_costs.append(SCALED_TRASH_COST)

    for i in range(len(intensities2)):
        # Layer 2 to trash
        # print("Layer 2 to trash", len(nodes_supply) - 1, LAYER2_START_IDX + i)
        edge_starts.append(len(nodes_supply) - 1)
        edge_ends.append(LAYER2_START_IDX + i)
        edge_capacities.append(intensities2[i])
        edge_costs.append(SCALED_TRASH_COST)

    return nodes_supply, np.asarray(edge_starts, dtype=np.int64), np.asarray(edge_ends, dtype=np.int64), np.asarray(edge_capacities, dtype=np.int64), np.asarray(edge_costs, dtype=np.int64)


def wasserstein_integer(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    #assert all(np.issubdtype(x.dtype, np.integer) for x in [X1, Y1, intensities1, X2, Y2, intensities2, trash_cost]), "All arguments must be integer type"
    assert trash_cost % 2 == 0, "Trash cost must be even (divisible by 2)"
    nodes_supply, edge_starts, edge_ends, edge_capacities, edge_costs = wasserstein_network(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost)
    flows = pylmcf_cpp.lmcf(nodes_supply, edge_starts, edge_ends, edge_capacities, edge_costs)
    src_trashed = flows[len(flows)-len(X1)-len(X2):len(flows)-len(X2)]
    dst_trashed = flows[len(flows)-len(X2):]
    sources = edge_starts[len(X1)+len(X2):len(flows)-len(X1)-len(X2)] - 1
    sinks = edge_ends[len(X1)+len(X2):len(flows)-len(X1)-len(X2)] - (1 + len(X1))
    total_cost = np.sum(flows * edge_costs)
    flows = flows[len(X1)+len(X2):len(flows)-len(X1)-len(X2)]



    mask = flows > 0

    return {
        "src_trashed": src_trashed.copy(),
        "dst_trashed": dst_trashed.copy(),
        "transport_source_idx": sources[mask],
        "transport_sink_idx": sinks[mask],
        "transport_flow": flows[mask],
        "total_cost": total_cost
    }


def wasserstein(X1, Y1, intensities1, X2, Y2, intensities2, trash_cost):
    return wasserstein_integer(X1, Y1, intensities1.astype(np.int64) , X2, Y2, intensities2.astype(np.int64), trash_cost)
