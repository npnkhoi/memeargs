from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from copy import deepcopy

from extensions.typing.graph import Graph


class ParaphraseProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> Graph:
        EMPTY_GRAPH = Graph(nodes=[], edges=[])
        result = annotations.argument_graph_conversion_result
        if not result.argumentative or len(result.nodes) != len(annotations.graph.nodes):
            return EMPTY_GRAPH
        
        # replace the text
        graph = deepcopy(annotations.graph)
        for node, new_node in zip(graph.nodes, result.nodes):
            node.text = new_node.text  
        return graph
            
        