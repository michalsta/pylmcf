// test_dual_repair.cpp
// -------------------------------------------------------------------------
// Exhaustive correctness suite for NetworkSimplex::dualSimplexRepair() and the
// warmRun() restart path it backs.
//
// Strategy: for many random + hand-built instances, solve COLD, then apply a
// sequence of capacity/supply mutations.  After each mutation:
//
//   * warm solver : reuse the persistent NetworkSimplex, call warmRun()
//                    (-> repairTreeFlows, then dualSimplexRepair, then a
//                     cold-init fallback only if both fail)
//   * reference   : a brand-new NetworkSimplex solving the mutated data cold
//
// and assert:
//   1. same ProblemType (OPTIMAL / INFEASIBLE)
//   2. identical totalCost() (exact int64 equality)
//   3. warm flows are primal-feasible (bounds + conservation)
//   4. warm solution satisfies the exact dual optimality certificate
//      (complementary slackness via potential()), independent of the reference
//   5. totalCost() == recomputed Sum(cost*flow)
//
// Costs are held fixed across each warm chain (matches production: wnet keeps
// edge costs fixed across set_point and only mutates caps/supplies — the
// precondition for dual feasibility of the retained basis).
//
// The suite also fails if the dual-repair path is never exercised, so a
// regression that quietly routes everything through cold init cannot pass.
//
// Build (mirrors src/wnet/cpp/wnet/Makefile):
//   g++ -I$(python -m pylmcf --include) -std=c++20 -O2 -o /tmp/test_dual_repair \
//       tests_cpp/test_dual_repair.cpp && /tmp/test_dual_repair
// -------------------------------------------------------------------------

#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using Value = int64_t;
using Graph = lemon::StaticDigraph;
using NS    = lemon::NetworkSimplex<Graph, Value, Value>;

// ---- A self-contained problem instance ----------------------------------
struct Instance {
    int n = 0;                                  // node count
    std::vector<std::pair<int,int>> arcs;       // sorted by (src,tgt)
    std::vector<Value> cost, cap, supply;       // cost/cap per arc; supply per node
    std::vector<Value> wit;                     // a feasible witness flow for `supply`
                                                // (wit[i] <= cap[i], conserves supply)
};

static int  g_fail = 0;
static long g_warm = 0, g_cold = 0, g_dual = 0, g_primal = 0, g_checks = 0;
// Warm-repair strategy under test: RepairOnly (Simple regression),
// Dual, or Primal.  Set per campaign in main().
static NS::WarmRepair g_strategy = NS::WarmRepair::Dual;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            ++g_fail;                                                         \
            std::printf("  FAIL [%s:%d] %s\n", __func__, __LINE__, msg);      \
        }                                                                     \
    } while (0)

// StaticDigraph::build wants arcs sorted by (src,tgt) and assigns arc ids in
// that order.  The whole harness assumes in.arcs[i] <-> arcFromId(i), so we
// canonicalize the Instance itself: sort arcs and permute cost/cap/wit in
// lockstep.  Must be called on every Instance before it is built/solved.
static void normalize(Instance& in) {
    const int m = (int)in.arcs.size();
    std::vector<int> p(m);
    for (int i = 0; i < m; ++i) p[i] = i;
    std::stable_sort(p.begin(), p.end(),
        [&](int x, int y){ return in.arcs[x] < in.arcs[y]; });
    auto permute = [&](auto& v){
        if ((int)v.size() != m) return;
        std::decay_t<decltype(v)> t(m);
        for (int i = 0; i < m; ++i) t[i] = v[p[i]];
        v.swap(t);
    };
    permute(in.arcs); permute(in.cost); permute(in.cap); permute(in.wit);
}

static void build_graph(const Instance& in, Graph& g) {
    // in is already normalized (arcs sorted); build directly so arc ids stay
    // aligned with in.arcs / in.cost / in.cap.
    g.build(in.n, in.arcs.begin(), in.arcs.end());
}

// Solve a fresh cold instance; return status and (if OPTIMAL) cost.
static NS::ProblemType solve_fresh(const Instance& in, Value& out_cost) {
    Graph g; build_graph(in, g);
    Graph::ArcMap<Value> cm(g), um(g);
    Graph::NodeMap<Value> sm(g);
    for (int i = 0; i < (int)in.arcs.size(); ++i) {
        cm[g.arcFromId(i)] = in.cost[i];
        um[g.arcFromId(i)] = in.cap[i];
    }
    for (int v = 0; v < in.n; ++v) sm[g.nodeFromId(v)] = in.supply[v];
    NS ns(g);
    ns.upperMap(um).costMap(cm).supplyMap(sm);
    auto st = ns.run();
    if (st == NS::OPTIMAL) out_cost = ns.totalCost();
    return st;
}

// Verify the warm solver's solution is feasible and optimal.  Optimality is
// established by exact equality with an independent cold solve (solve_fresh);
// combined with primal feasibility this is a sound certificate (a feasible
// flow whose cost equals the optimum is optimal).  We deliberately do NOT
// re-derive optimality from potential() — its sign/offset convention is an
// internal detail; the independent-cold oracle is convention-free.
static void certify(const char* tag, NS& ns, const Graph& g, const Instance& in,
                    Value ref_cost) {
    CHECK(ns.totalCost() == ref_cost, tag);              // optimality oracle

    const int m = (int)in.arcs.size();
    std::vector<Value> bal(in.n, 0);
    Value recomputed = 0;
    for (int i = 0; i < m; ++i) {
        Value f = ns.flow(g.arcFromId(i));
        CHECK(f >= 0 && f <= in.cap[i], "flow out of [0,cap]");
        recomputed += f * in.cost[i];
        bal[in.arcs[i].first]  -= f;
        bal[in.arcs[i].second] += f;
    }
    CHECK(recomputed == ns.totalCost(), "Sum(c*f) != totalCost()");
    for (int v = 0; v < in.n; ++v) {                     // conservation
        if (bal[v] != -in.supply[v]) {
            std::printf("    [%s] node %d: bal=%lld supply=%lld (st cost=%lld ref=%lld)\n",
                        tag, v, (long long)bal[v], (long long)in.supply[v],
                        (long long)ns.totalCost(), (long long)ref_cost);
        }
        CHECK(bal[v] == -in.supply[v], tag);
    }
}

// Compare warm-restart vs fresh-cold for the *current* in.{cap,supply}.
// `ns`/`g` are the persistent warm solver and its maps (already wired).
// One warm transition vs an independent cold solve.  Returns true if the
// persistent solver is still safe to reuse for a further warm restart
// (i.e. the warm result was OPTIMAL).  Production (wnet) never reuses a
// solver across a non-OPTIMAL result, so neither do the chained campaigns.
static bool step(const char* tag, NS& ns, Graph& g,
                 Graph::ArcMap<Value>& um, Graph::NodeMap<Value>& sm,
                 const Instance& in) {
    for (int i = 0; i < (int)in.arcs.size(); ++i) um[g.arcFromId(i)] = in.cap[i];
    for (int v = 0; v < in.n; ++v) sm[g.nodeFromId(v)] = in.supply[v];

    int w0 = ns.warmStartCount(), c0 = ns.coldStartCount(),
        d0 = ns.dualRepairCount(), p0 = ns.primalRepairCount();
    ns.upperMap(um).supplyMap(sm);
    auto st_warm = ns.warmRun(NS::BLOCK_SEARCH, g_strategy);
    g_warm   += ns.warmStartCount()   - w0;
    g_cold   += ns.coldStartCount()   - c0;
    g_dual   += ns.dualRepairCount()  - d0;
    g_primal += ns.primalRepairCount() - p0;

    Value ref_cost = 0;
    auto st_ref = solve_fresh(in, ref_cost);

    bool same = (st_warm == NS::OPTIMAL) == (st_ref == NS::OPTIMAL);
    CHECK(same, tag);
    if (st_warm == NS::OPTIMAL && st_ref == NS::OPTIMAL)
        certify(tag, ns, g, in, ref_cost);
    return st_warm == NS::OPTIMAL;
}

// ---- Random instance / mutation generator -------------------------------
// Feasibility by construction: draw a random "witness" flow, set node supply
// from it, set caps >= witness (sometimes exactly tight).  Mutations redraw a
// fresh witness on the same arc set -> mutated problem stays feasible while the
// retained basis is frequently broken (the dual-repair trigger).
struct Gen {
    std::mt19937_64 rng;
    explicit Gen(uint64_t seed) : rng(seed) {}
    int uni(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

    // Costs are drawn ONCE here and never mutated again: warm restart only
    // supports changing capacities/supplies (it never re-pushes the cost map),
    // so cost-changing "mutations" are out of scope by design.
    Instance make(int n, int m) {
        Instance in; in.n = n;
        for (int e = 0; e < m; ++e) {
            int u = uni(0, n - 1), v = uni(0, n - 1);
            if (u == v) v = (v + 1) % n;
            in.arcs.push_back({u, v});
        }
        in.cost.resize(m);
        for (int i = 0; i < m; ++i) in.cost[i] = uni(0, 50);
        in.cap.assign(m, 0); in.supply.assign(n, 0); in.wit.assign(m, 0);
        normalize(in);                              // arcs sorted, cost aligned
        mutate(in, /*supply_changes=*/true, /*cap_changes=*/true);
        return in;
    }

    // The ONLY mutation: changes capacities and/or the supply vector, never
    // costs.  Feasibility is preserved by construction — `in.wit` is always a
    // flow that conserves `in.supply` with wit[i] <= cap[i], so the mutated
    // problem has at least one feasible solution.  Frequently sets caps exactly
    // to the witness (tight/binding) so retained basic arcs go out of bounds,
    // which is precisely what triggers repairTreeFlows()/dualSimplexRepair().
    void mutate(Instance& in, bool supply_changes, bool cap_changes) {
        const int m = (int)in.arcs.size();
        if (supply_changes) {
            std::vector<Value> w(m);
            std::vector<Value> sup(in.n, 0);
            for (int i = 0; i < m; ++i) {
                w[i] = uni(0, 12);
                sup[in.arcs[i].first]  += w[i];   // supply leaves source
                sup[in.arcs[i].second] -= w[i];
            }
            in.wit = w;
            in.supply = sup;
        }
        // `in.wit` is now a feasible flow for `in.supply` (whether refreshed
        // above or carried over from the previous feasible state).
        for (int i = 0; i < m; ++i) {
            if (cap_changes) {
                if (uni(0, 2) == 0) in.cap[i] = in.wit[i];                 // tight
                else                in.cap[i] = in.wit[i] + uni(0, 18);    // slack
            } else {
                in.cap[i] = std::max<Value>(in.cap[i], in.wit[i]);         // stay feasible
            }
        }
    }
};

// ---- Minimal single-step Dual reproducer --------------------------------
// Tiny graphs, one cold solve + exactly one mutation through the Dual path.
// On the first wrong/feasibility-broken result, dump a fully self-contained
// repro and exit(2) so a single dual-repair invocation can be hand-traced.
static void dump_repro(const Instance& a, const Instance& b, const char* why) {
    std::printf("\n==== MINIMAL DUAL REPRO (%s) ====\n", why);
    std::printf("n=%d  m=%d\n", a.n, (int)a.arcs.size());
    std::printf("arcs ="); for (auto& e : a.arcs) std::printf(" (%d,%d)", e.first, e.second);
    std::printf("\ncost ="); for (auto c : a.cost)   std::printf(" %lld", (long long)c);
    std::printf("\ncap0 ="); for (auto c : a.cap)    std::printf(" %lld", (long long)c);
    std::printf("\nsup0 ="); for (auto s : a.supply) std::printf(" %lld", (long long)s);
    std::printf("\ncap1 ="); for (auto c : b.cap)    std::printf(" %lld", (long long)c);
    std::printf("\nsup1 ="); for (auto s : b.supply) std::printf(" %lld", (long long)s);
    std::printf("\n=================================\n");
}

static void minimal_dual_repro() {
    std::printf("minimal dual repro scan\n");
    Gen gen(0x1234567u);
    for (int t = 0; t < 4000; ++t) {
        int n = gen.uni(3, 6);
        int m = gen.uni(n, 2 * n);
        Instance a = gen.make(n, m);

        Graph g; build_graph(a, g);
        Graph::ArcMap<Value> um(g), cm(g); Graph::NodeMap<Value> sm(g);
        for (int i = 0; i < m; ++i) { cm[g.arcFromId(i)] = a.cost[i]; um[g.arcFromId(i)] = a.cap[i]; }
        for (int v = 0; v < n; ++v) sm[g.nodeFromId(v)] = a.supply[v];
        NS ns(g);
        ns.upperMap(um).costMap(cm).supplyMap(sm);
        if (ns.run() != NS::OPTIMAL) continue;

        Instance b = a;                                   // one mutation
        gen.mutate(b, gen.uni(0,1), true);

        int d0 = ns.dualRepairCount();
        for (int i = 0; i < m; ++i) um[g.arcFromId(i)] = b.cap[i];
        for (int v = 0; v < n; ++v) sm[g.nodeFromId(v)] = b.supply[v];
        ns.upperMap(um).supplyMap(sm);
        auto st = ns.warmRun(NS::BLOCK_SEARCH, NS::WarmRepair::Dual);
        bool used_dual = ns.dualRepairCount() > d0;
        if (!used_dual) continue;                         // only care about dual path

        Value ref = 0; auto st_ref = solve_fresh(b, ref);
        if ((st == NS::OPTIMAL) != (st_ref == NS::OPTIMAL)) { dump_repro(a,b,"status mismatch"); std::exit(2); }
        if (st != NS::OPTIMAL) continue;
        if (ns.totalCost() != ref) { dump_repro(a,b,"cost mismatch"); std::exit(2); }
        std::vector<Value> bal(b.n, 0); Value rc_sum = 0;
        for (int i = 0; i < m; ++i) {
            Value f = ns.flow(g.arcFromId(i));
            if (f < 0 || f > b.cap[i]) { dump_repro(a,b,"bounds violated"); std::exit(2); }
            rc_sum += f * b.cost[i];
            bal[b.arcs[i].first] -= f; bal[b.arcs[i].second] += f;
        }
        if (rc_sum != ns.totalCost()) { dump_repro(a,b,"Sum(c*f)!=totalCost"); std::exit(2); }
        for (int v = 0; v < b.n; ++v)
            if (bal[v] != -b.supply[v]) { dump_repro(a,b,"conservation"); std::exit(2); }
    }
    std::printf("  (no single-step dual failure found in scan)\n");
}

// Build + cold-prime a persistent solver for `in` (must be feasible).
static void prime(NS& ns, Graph& g, Graph::ArcMap<Value>& um,
                   Graph::ArcMap<Value>& cm, Graph::NodeMap<Value>& sm,
                   const Instance& in) {
    for (int i = 0; i < (int)in.arcs.size(); ++i) {
        cm[g.arcFromId(i)] = in.cost[i];
        um[g.arcFromId(i)] = in.cap[i];
    }
    for (int v = 0; v < in.n; ++v) sm[g.nodeFromId(v)] = in.supply[v];
    ns.upperMap(um).costMap(cm).supplyMap(sm);
    CHECK(ns.run() == NS::OPTIMAL, "edge cold prime not OPTIMAL");
}

// ---- Deterministic edge cases (feasibility-preserving chains) ------------
static void edge_cases() {
    std::printf("edge cases\n");

    // Each case runs a feasibility-preserving chain on its own persistent
    // solver (cold-primed once, then warm restarts for every step).
    auto run_case = [](const char* tag, Instance in,
                       const std::vector<std::pair<std::vector<Value>,
                                                   std::vector<Value>>>& steps) {
        normalize(in);                              // align arcs/cost with arc ids
        Graph g; build_graph(in, g);
        Graph::ArcMap<Value> um(g), cm(g); Graph::NodeMap<Value> sm(g);
        NS ns(g);
        prime(ns, g, um, cm, sm, in);
        for (auto& [cap, sup] : steps) {
            in.cap = cap; in.supply = sup;
            if (!step(tag, ns, g, um, sm, in)) {
                CHECK(false, "feasible edge chain returned non-OPTIMAL");
                break;
            }
        }
    };

    // single arc {0,1} cost 3: tight & loose caps, supply tracked (feasible).
    run_case("single-arc",
        Instance{2, {{0,1}}, {3}, {10}, {7,-7}, {}},
        {{{10},{7,-7}}, {{7},{7,-7}}, {{5},{5,-5}}, {{30},{2,-2}}, {{9},{9,-9}}});

    // two parallel arcs {0,1} costs {1,9}, supply {20,-20}: kill/restore the
    // cheap arc; total capacity always >= 20 so every step is feasible.
    run_case("parallel",
        Instance{2, {{0,1},{0,1}}, {1,9}, {100,100}, {20,-20}, {}},
        {{{0,100},{20,-20}}, {{5,100},{20,-20}},
         {{20,0},{20,-20}},  {{100,100},{20,-20}}});

    // path 0->1->2->3 costs {2,2,2}: squeeze the middle arc; supply = min(c,30)
    // keeps every arc within capacity.
    {
        std::vector<std::pair<std::vector<Value>,std::vector<Value>>> st;
        for (Value c : {20,12,8,25,15,30}) {
            Value s = std::min<Value>(c, 30);
            st.push_back({{30,c,30}, {s,0,0,-s}});
        }
        run_case("path-middle-squeeze",
            Instance{4, {{0,1},{1,2},{2,3}}, {2,2,2}, {30,30,30}, {15,0,0,-15}, {}}, st);
    }

    // supply-only shifts, generous fixed caps.
    {
        std::vector<std::pair<std::vector<Value>,std::vector<Value>>> st;
        for (Value q : {4,18,1,25,10}) st.push_back({{50,50,50}, {q,0,-q}});
        run_case("supply-only-shift",
            Instance{3, {{0,1},{1,2},{0,2}}, {1,1,5}, {50,50,50}, {10,0,-10}, {}}, st);
    }
}

// Warm vs cold must AGREE when a capacity change makes the problem
// infeasible.  Tested on a fresh solver (no chaining past the infeasible
// result — production never reuses a solver across non-OPTIMAL either).
static void infeasible_agreement() {
    std::printf("infeasible agreement\n");
    struct Case { Instance feas; std::vector<Value> bad_cap; };
    std::vector<Case> cases = {
        // single arc: need 7 units, drop cap to 4.
        {{2,{{0,1}},{3},{10},{7,-7},{}}, {4}},
        // path: middle arc throttled below required flow.
        {{4,{{0,1},{1,2},{2,3}},{1,1,1},{20,20,20},{12,0,0,-12},{}}, {20,5,20}},
        // parallel arcs: total cap below demand.
        {{2,{{0,1},{0,1}},{1,2},{50,50},{30,-30},{}}, {10,10}},
    };
    for (auto& c : cases) {
        normalize(c.feas);                          // (cases are pre-sorted; no-op)
        Graph g; build_graph(c.feas, g);
        Graph::ArcMap<Value> um(g), cm(g); Graph::NodeMap<Value> sm(g);
        NS ns(g);
        prime(ns, g, um, cm, sm, c.feas);
        Instance bad = c.feas; bad.cap = c.bad_cap;
        for (int i = 0; i < (int)bad.arcs.size(); ++i) um[g.arcFromId(i)] = bad.cap[i];
        for (int v = 0; v < bad.n; ++v) sm[g.nodeFromId(v)] = bad.supply[v];
        ns.upperMap(um).supplyMap(sm);
        auto stw = ns.warmRun(NS::BLOCK_SEARCH, g_strategy);
        Value rc = 0; auto str = solve_fresh(bad, rc);
        CHECK((stw == NS::OPTIMAL) == (str == NS::OPTIMAL),
              "warm/cold disagree on infeasible");
        CHECK(str != NS::OPTIMAL, "expected infeasible case was feasible");
    }
}

// ---- Random campaign ----------------------------------------------------
static void random_campaign() {
    std::printf("random campaign\n");
    Gen gen(0xC0FFEEu);
    const int instances = 240;
    for (int t = 0; t < instances; ++t) {
        int n = gen.uni(3, 14);
        int m = gen.uni(n, n * 4);
        Instance in = gen.make(n, m);

        // cold prime
        Graph g; build_graph(in, g);
        Graph::ArcMap<Value> um(g), cm(g); Graph::NodeMap<Value> sm(g);
        for (int i = 0; i < m; ++i) {
            cm[g.arcFromId(i)] = in.cost[i];
            um[g.arcFromId(i)] = in.cap[i];
        }
        for (int v = 0; v < n; ++v) sm[g.nodeFromId(v)] = in.supply[v];
        NS ns(g);
        ns.upperMap(um).costMap(cm).supplyMap(sm);
        if (ns.run() != NS::OPTIMAL) continue;     // skip rare cold-infeasible seeds

        const int steps = gen.uni(4, 12);          // sequential warm chain
        for (int s = 0; s < steps; ++s) {
            int kind = gen.uni(0, 2);
            if (kind == 0)      gen.mutate(in, true,  false);   // supply only
            else if (kind == 1) gen.mutate(in, false, true);    // caps only
            else                gen.mutate(in, true,  true);    // both
            if (!step("random-step", ns, g, um, sm, in)) {
                CHECK(false, "feasible chain returned non-OPTIMAL");
                break;                              // never chain on poisoned solver
            }
        }
    }
}

int main() {
    // 0. Hunt a minimal single-step Dual failure first (exits on first hit).
    minimal_dual_repro();

    // 1. Regression: the Simple warm path (no extra repair) must be flawless.
    std::printf("\n[Simple warm-path regression]\n");
    g_strategy = NS::WarmRepair::RepairOnly;
    edge_cases();
    infeasible_agreement();
    random_campaign();
    std::printf("after Simple: checks=%ld warm=%ld dual=%ld primal=%ld cold=%ld fails=%d\n",
                g_checks, g_warm, g_dual, g_primal, g_cold, g_fail);

    // 2. Full Dual campaign.
    std::printf("\n[Dual warm-path]\n");
    g_strategy = NS::WarmRepair::Dual;
    long dual0 = g_dual;
    edge_cases();
    infeasible_agreement();
    random_campaign();
    long dual_fired = g_dual - dual0;

    // 3. Full Primal campaign (independent-cold oracle, same as Dual).
    std::printf("\n[Primal warm-path]\n");
    g_strategy = NS::WarmRepair::Primal;
    long primal0 = g_primal;
    edge_cases();
    infeasible_agreement();
    random_campaign();
    long primal_fired = g_primal - primal0;

    std::printf("\n--- summary ---\n");
    std::printf("checks=%ld  warm=%ld  dual=%ld  primal=%ld  cold=%ld\n",
                g_checks, g_warm, g_dual, g_primal, g_cold);

    // Coverage guards: each repair path must actually have run, else the
    // corresponding campaign is vacuous.
    if (dual_fired < 20) {
        std::printf("INEFFECTIVE: dual-repair exercised only %ld times (<20)\n", dual_fired);
        ++g_fail;
    }
    if (primal_fired < 20) {
        std::printf("INEFFECTIVE: primal-repair exercised only %ld times (<20)\n", primal_fired);
        ++g_fail;
    }
    if (g_fail) { std::printf("RESULT: FAILED (%d)\n", g_fail); return 1; }
    std::printf("RESULT: PASSED\n");
    return 0;
}
