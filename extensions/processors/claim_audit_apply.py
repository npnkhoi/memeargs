from copy import deepcopy

from pydantic import BaseModel

from extensions.const import EdgeType, VertexType
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import Argument, ClaimAudit, Graph, Vertex


class ClaimAuditApplyProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.claim_merged_graph
        audit: ClaimAudit = annotations.claim_audit

        if graph is None or audit is None:
            return None

        graph = deepcopy(graph)

        # Step 1: apply edits
        text_updates = {
            v.node_idx: v.new_text
            for v in audit.existing
            if v.action == "edit" and v.new_text is not None
        }
        for node in graph.nodes:
            if node.idx in text_updates:
                node.text = text_updates[node.idx]

        # Step 2: delete nodes
        to_delete = {v.node_idx for v in audit.existing if v.action == "delete"}
        if to_delete:
            graph.nodes = [n for n in graph.nodes if n.idx not in to_delete]
            # remove arguments where claim is deleted
            graph.arguments = [a for a in graph.arguments if a.claim not in to_delete]
            # remove deleted indices from premises; drop argument if premises become empty
            new_args = []
            for arg in graph.arguments:
                new_premises = [p for p in arg.premises if p not in to_delete]
                if new_premises:
                    new_args.append(Argument(claim=arg.claim, premises=new_premises, type=arg.type))
            graph.arguments = new_args

        # Step 3: add missing claims
        next_idx = max((n.idx for n in graph.nodes), default=-1) + 1
        for missing in audit.missing:
            new_idx = next_idx
            next_idx += 1
            graph.nodes.append(Vertex(idx=new_idx, text=missing.text, type=VertexType.CLAIM))
            for parent in missing.parents:
                graph.arguments.append(Argument(
                    claim=parent.node_idx,
                    premises=[new_idx],
                    type=EdgeType(parent.type),
                ))
            for arg in missing.arguments:
                graph.arguments.append(Argument(
                    claim=new_idx,
                    premises=arg.premises,
                    type=EdgeType(arg.type),
                ))

        return graph
