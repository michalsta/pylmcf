"""The free-threaded build must actually stay free-threaded.

On CPython 3.15t the extension is built in split mode with nanobind's
FREE_THREADED option, which declares ``Py_MOD_GIL_NOT_USED``. If that option is
ever dropped, or the build quietly falls back to a linked mode, importing the
extension re-enables the GIL and the wheel is free-threaded in name only --
nothing else in the suite would notice, because every test still passes.

The whole module is skipped unless the GIL is genuinely off *after* importing
the extension. That is the point: on 3.14t the linked fallback is expected to
turn the GIL back on, so this correctly stays quiet there rather than failing.
"""

import sys
import threading

import numpy as np
import pytest

from pylmcf.pylmcf_cpp import CGraph, lmcf

pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="needs a free-threaded interpreter with the GIL still disabled after import",
)

N_NODES = 40
N_EDGES = 200
N_THREADS = 8
N_ROUNDS = 25


def _problem():
    rng = np.random.default_rng(0)
    starts = np.sort(rng.integers(0, N_NODES, N_EDGES)).astype(np.int32)
    ends = ((starts + 1 + rng.integers(0, N_NODES - 1, N_EDGES)) % N_NODES).astype(
        np.int32
    )
    order = np.lexsort((ends, starts))
    starts, ends = starts[order], ends[order]
    caps = rng.integers(5, 50, N_EDGES).astype(np.int64)
    costs = rng.integers(0, 100, N_EDGES).astype(np.int64)
    supply = np.zeros(N_NODES, dtype=np.int64)
    supply[0] = 30
    supply[-1] = -30
    return starts, ends, caps, costs, supply


def _solve_once(starts, ends, caps, costs, supply):
    # A fresh graph per call: the module claims no *global* state, not that one
    # CGraph may be driven from two threads at once.
    g = CGraph(N_NODES, starts, ends)
    g.set_node_supply(supply)
    g.set_edge_capacities(caps)
    g.set_edge_costs(costs)
    g.solve()
    flows = lmcf(
        supply, starts.astype(np.int64), ends.astype(np.int64), caps, costs
    )
    return g.total_cost(), int((flows * costs).sum())


def test_gil_stays_disabled_after_import():
    assert not sys._is_gil_enabled()


def test_concurrent_solves_agree_with_serial():
    problem = _problem()
    expected = _solve_once(*problem)

    results = []
    errors = []

    def worker():
        try:
            for _ in range(N_ROUNDS):
                results.append(_solve_once(*problem))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert len(results) == N_THREADS * N_ROUNDS
    assert all(r == expected for r in results)
    # If the GIL had been re-enabled behind our back, the run above proves nothing.
    assert not sys._is_gil_enabled()
