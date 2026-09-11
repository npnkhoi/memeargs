"""
Conversion functions between the standard Graph schema and the
top-down / bottom-up schemas used by LLMProcessor, plus the GraphConverter processor.
"""
from pydantic import BaseModel

from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import (
    Argument,
    EdgeConstructionOutput,
    Graph,
    TopDownGraph,
    BottomUpGraph,
    top_down_graph_to_graph,
    bottom_up_graph_to_graph,
    graph_to_top_down,
    graph_to_bottom_up,
)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class GraphConverter(AnnotationProcessor):
    """Convert TopDownGraph or BottomUpGraph to canonical Graph."""

    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        input_name = self._input_names_snake[0]
        value = getattr(annotations, input_name)
        if value is None:
            return None
        if isinstance(value, TopDownGraph):
            return top_down_graph_to_graph(value)
        elif isinstance(value, BottomUpGraph):
            return bottom_up_graph_to_graph(value)
        elif isinstance(value, Graph):
            return graph_to_bottom_up(value)
        else:
            raise TypeError(f"GraphConverter: unsupported input type {type(value)}")


class EdgeConstructionToModelGraphProcessor(AnnotationProcessor):
    """Combine EdgeConstructionOutput arguments with nodes from GraphByHuman to produce GraphByModel."""

    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        edge_output: EdgeConstructionOutput = annotations.edge_construction_output
        human_graph: Graph = annotations.graph_by_human
        if edge_output is None or human_graph is None:
            return None

        valid_indices = {v.idx for v in human_graph.nodes}
        filtered_args = []
        for arg in edge_output.arguments:
            if arg.claim not in valid_indices:
                continue
            valid_premises = [p for p in arg.premises if p in valid_indices]
            if not valid_premises:
                continue
            filtered_args.append(Argument(claim=arg.claim, premises=valid_premises, type=arg.type))

        return Graph(nodes=list(human_graph.nodes), arguments=filtered_args)
