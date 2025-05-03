#ifndef GRAPH_ELEMENTS_HPP
#define GRAPH_ELEMENTS_HPP

#include <iostream>
#include <vector>
#include <span>
#include <algorithm>
#include <stdexcept>

#include "basics.hpp"

class FlowNode {
    const LEMON_INT id;
public:
    FlowNode(LEMON_INT id) : id(id) {}
    LEMON_INT get_id() const { return id; }
};

class SourceNode final : public FlowNode {
public:
    SourceNode(LEMON_INT id) : FlowNode(id) {}
};

class SinkNode : public FlowNode {
public:
    SinkNode(LEMON_INT id) : FlowNode(id) {}
};

class EmpiricalNode final : public FlowNode {
    const size_t peak_index;
    const LEMON_INT intensity;
public:
    EmpiricalNode(LEMON_INT id, size_t peak_index, LEMON_INT intensity)
        : FlowNode(id), peak_index(peak_index), intensity(intensity) {}
    size_t get_peak_index() const { return peak_index; }
    LEMON_INT get_intensity() const { return intensity; }
};


class TheoreticalNode final : public FlowNode {
    const size_t spectrum_id;
    const size_t peak_index;
    const LEMON_INT intensity;
public:
    TheoreticalNode(LEMON_INT id, size_t spectrum_id, size_t peak_index, LEMON_INT intensity)
        : FlowNode(id), spectrum_id(spectrum_id), peak_index(peak_index), intensity(intensity) {}
    size_t get_spectrum_id() const { return spectrum_id; }
    size_t get_peak_index() const { return peak_index; }
    LEMON_INT get_intensity() const { return intensity; }
};

using FlowNodeVariant = std::variant<SourceNode, SinkNode, EmpiricalNode, TheoreticalNode>;

class FlowEdge {
    const LEMON_INT id;
    const FlowNode& start_node;
    const FlowNode& end_node;
public:
    FlowEdge(LEMON_INT id, const FlowNode& start_node, const FlowNode& end_node)
        : id(id), start_node(start_node), end_node(end_node) {}
    LEMON_INT get_id() const { return id; }
    const FlowNode& get_start_node() const { return start_node; }
    const FlowNode& get_end_node() const { return end_node; }
};

class MatchingEdge final : public FlowEdge {
    const size_t empirical_peak_index;
    const size_t theoretical_spectrum_id;
    const size_t theoretical_peak_index;
public:
    MatchingEdge(LEMON_INT id, const FlowNode& start_node, const FlowNode& end_node,
                 size_t empirical_peak_index, size_t theoretical_spectrum_id, size_t theoretical_peak_index)
        : FlowEdge(id, start_node, end_node),
          empirical_peak_index(empirical_peak_index),
          theoretical_spectrum_id(theoretical_spectrum_id),
          theoretical_peak_index(theoretical_peak_index) {}
    size_t get_empirical_peak_index() const { return empirical_peak_index; }
    size_t get_theoretical_spectrum_id() const { return theoretical_spectrum_id; }
    size_t get_theoretical_peak_index() const { return theoretical_peak_index; }
};

class SrcToEmpiricalEdge final : public FlowEdge {
public:
    SrcToEmpiricalEdge(LEMON_INT id, const FlowNode& start_node, const FlowNode& end_node)
        : FlowEdge(id, start_node, end_node) {}
};

class TheoreticalToSinkEdge final : public FlowEdge {
public:
    TheoreticalToSinkEdge(LEMON_INT id, const FlowNode& start_node, const FlowNode& end_node)
        : FlowEdge(id, start_node, end_node) {}
};

class SimpleTrashEdge final : public FlowEdge {
    LEMON_INT cost;
public:
    SimpleTrashEdge(LEMON_INT id, const FlowNode& start_node, const FlowNode& end_node, LEMON_INT cost)
        : FlowEdge(id, start_node, end_node), cost(cost) {}
    LEMON_INT get_cost() const { return cost; }
};

using FlowEdgeVariant = std::variant<MatchingEdge, SrcToEmpiricalEdge, TheoreticalToSinkEdge, SimpleTrashEdge>;

#endif // GRAPH_ELEMENTS_HPP