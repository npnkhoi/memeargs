from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from copy import deepcopy

from extensions.typing.graph import GraphByModel

class GraphCleaningProcessor(AnnotationProcessor):
    """
    This processor, as all others, takes Annotations when called.
    The annotations must have notna graph field.
    Then this processor clean the graph by: removing the invalid edges.
    """
    
    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)
        assert ("Graph", "model") in self._stage_config.inputs \
            or ("Graph", "human") in self._stage_config.inputs
    
    def process(self, annotations: Annotations, logger=None) -> GraphByModel | None:
        graph = deepcopy(annotations.graph)

        if graph is None:
            return None
        
        # remove invalid arguments (old: edges)
        # graph.edges = [
        #     edge for edge in graph.edges
        #     if edge.from_idx < len(graph.nodes) \
        #         and edge.to_idx < len(graph.nodes)
        # ]
        graph.arguments = [
            arg for arg in graph.arguments
            if arg.claim < len(graph.nodes)
            and all(p < len(graph.nodes) for p in arg.premises)
        ]
        return graph