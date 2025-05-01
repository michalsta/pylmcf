# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Union

BIGINT = np.int64(2**56)

TODO_REMOVE_ME = 982347589

@dataclass(frozen=True)
class FlowNode:
    id: int

@dataclass(frozen=True)
class SourceNode(FlowNode):
    @property
    def layer(self):
        return 0


@dataclass(frozen=True)
class SinkNode(FlowNode):
    @property
    def layer(self):
        return 3


@dataclass(frozen=True)
class EmpiricalNode(FlowNode):
    peak_idx: int
    intensity: int
    @property
    def layer(self):
        return 1


@dataclass(frozen=True)
class TheoreticalNode(FlowNode):
    peak_idx: int
    spectrum_id: int
    intensity: int
    @property
    def layer(self):
        return 2


Node = Union[SourceNode, SinkNode, EmpiricalNode, TheoreticalNode]

@dataclass(frozen=True)
class FlowEdge:
    start_node: Node
    end_node: Node

    @property
    def start_node_id(self):
        return self.start_node.id

    @property
    def end_node_id(self):
        return self.end_node.id


@dataclass(frozen=True)
class MatchingEdge(FlowEdge):
    emp_peak_idx: int
    theo_spectrum_id: int
    theo_peak_idx: int
    cost: int

@dataclass(frozen=True)
class SrcToEmpEdge(FlowEdge):
    emp_peak_intensity: int
    @property
    def cost(self):
        return 0

@dataclass(frozen=True)
class TheoToSinkEdge(FlowEdge):
    theo_spectrum_id: int
    theo_peak_intensity: int
    @property
    def cost(self):
        return 0

@dataclass(frozen=True)
class SimpleTrashEdge(FlowEdge):
    cost: int

@dataclass(frozen=True)
class TheoryTrashEdge(FlowEdge):
    #theo_spectrum_id: int
    #theo_peak_intensity: int
    cost: int

@dataclass(frozen=True)
class EmpiricalTrashEdge(FlowEdge):
    emp_peak_intensity: int
    cost: int
