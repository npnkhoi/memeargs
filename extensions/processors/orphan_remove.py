from copy import deepcopy

from pydantic import BaseModel

from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import Graph


def remove_orphan_nodes(graph: Graph) -> Graph:
    """Remove nodes that do not appear as a premise or claim in any argument."""
    graph = deepcopy(graph)
    active = {p for arg in graph.arguments for p in arg.premises} | {arg.claim for arg in graph.arguments}
    graph.nodes = [v for v in graph.nodes if v.idx in active]
    return graph


class OrphanRemoveProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.background_audited_graph

        if graph is None:
            return None

        return remove_orphan_nodes(graph)
