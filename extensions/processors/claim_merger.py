from pydantic import BaseModel

from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import Argument, ClaimEquivalencePairs, Graph, merge_pair


class ClaimMergeProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = getattr(annotations, self._input_names_snake[0])
        pairs: ClaimEquivalencePairs = annotations.claim_equivalence_pairs

        if graph is None or pairs is None:
            return None

        for pair in pairs.pairs:
            remove = pair.idx_b if pair.keep_idx == pair.idx_a else pair.idx_a
            graph = merge_pair(graph, pair.keep_idx, remove)

        return graph
