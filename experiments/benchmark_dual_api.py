"""Compare canonical/raw Python reconstruction with fused C++ extraction.

Run with the interpreter containing the built pylmcf:
    python experiments/benchmark_dual_api.py --output timings.json
"""
import argparse
import json
import platform
import time
from statistics import median

import numpy as np
from pylmcf import Graph, NetworkSimplexLCT, NetworkSimplexLCTDyn


def measure(fn, repeats):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(7):
        begin = time.perf_counter_ns()
        for _ in range(repeats):
            fn()
        samples.append((time.perf_counter_ns() - begin) / repeats / 1000)
    return dict(median_us=median(samples), min_us=min(samples), samples_us=samples)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for n in (100, 500, 2000):
        rng = np.random.default_rng(42 + n)
        starts = np.repeat(np.arange(n, dtype=np.int64), 6)
        ends = rng.integers(0, n, len(starts), dtype=np.int64)
        order = np.lexsort((ends, starts))
        starts, ends = starts[order], ends[order]
        caps = rng.integers(1, 20, len(starts), dtype=np.int64)
        costs = rng.integers(0, 100, len(starts), dtype=np.int64)
        witness = rng.integers(0, caps + 1)
        supply = np.zeros(n, dtype=np.int64)
        np.add.at(supply, starts, witness)
        np.add.at(supply, ends, -witness)
        for cls in (Graph, NetworkSimplexLCT, NetworkSimplexLCTDyn):
            if cls is Graph:
                g = cls(n, starts, ends)
                g.set_node_supply(supply)
                g.set_edge_capacities(caps)
                g.set_edge_costs(costs)
            else:
                g = cls(supply, starts, ends, caps, costs)
            begin = time.perf_counter_ns()
            g.solve()
            cold_us = (time.perf_counter_ns() - begin) / 1000
            buffers = [np.empty(k, dtype=np.int64) for k in (n, len(starts), len(starts), len(starts))]

            def reconstruct(raw=True):
                pi = g.raw_potentials() if raw else g.potentials()
                rc = costs + pi[starts] - pi[ends]
                return pi, rc, np.maximum(rc, 0), np.minimum(rc, 0)

            methods = dict(raw_numpy=lambda: reconstruct(), snapshot=g.dual_values,
                           into=lambda: g.dual_values_into(*buffers), warm_solve=g.solve)
            if cls is Graph:
                methods["canonical_numpy"] = lambda: reconstruct(False)
            timing = {name: measure(fn, 100 if n < 2000 else 30) for name, fn in methods.items()}
            data = g.dual_values()
            dual = (-sum(int(b) * int(p) for b, p in zip(supply, data["potentials"]))
                    + sum(int(u) * int(p) for u, p in zip(caps, data["upper_bound_multipliers"])))
            assert dual == g.total_cost()
            rows.append(dict(solver=cls.__name__, nodes=n, arcs=len(starts), cold_solve_us=cold_us, timings=timing))
            print(cls.__name__, n, {k: round(v["median_us"], 2) for k, v in timing.items()}, flush=True)
    report = dict(platform=platform.platform(), python=platform.python_version(), rows=rows)
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2)


if __name__ == "__main__":
    run()
