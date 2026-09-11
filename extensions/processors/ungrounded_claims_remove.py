from copy import deepcopy

from pydantic import BaseModel

from extensions.const import VertexType
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import Argument, Graph

_GROUNDING_TYPES = {VertexType.ELEMENT, VertexType.CLAIM}


def remove_ungrounded_claims(graph: Graph) -> Graph:
    """Iteratively remove claim nodes that have no Element or Claim premise across
    all arguments pointing to them. Each pass may expose new ungrounded claims
    (because a removed claim was the sole grounding premise of another claim),
    so we repeat until fixpoint."""
    graph = deepcopy(graph)

    while True:
        node_type = {v.idx: v.type for v in graph.nodes}
        claim_idxs = {v.idx for v in graph.nodes if v.type == VertexType.CLAIM}

        ungrounded: set[int] = set()
        for claim_idx in claim_idxs:
            args_for_claim = [a for a in graph.arguments if a.claim == claim_idx]
            has_grounding = any(
                node_type.get(p) in _GROUNDING_TYPES
                for arg in args_for_claim
                for p in arg.premises
            )
            if not has_grounding:
                ungrounded.add(claim_idx)

        if not ungrounded:
            break

        graph.nodes = [v for v in graph.nodes if v.idx not in ungrounded]
        graph.arguments = [a for a in graph.arguments if a.claim not in ungrounded]
        new_args = []
        for arg in graph.arguments:
            new_premises = [p for p in arg.premises if p not in ungrounded]
            if new_premises:
                new_args.append(Argument(claim=arg.claim, premises=new_premises, type=arg.type))
        graph.arguments = new_args

    return graph


class UngroundedClaimsRemoveProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.argument_audited_graph

        if graph is None:
            return None

        return remove_ungrounded_claims(graph)
