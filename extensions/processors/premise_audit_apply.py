from copy import deepcopy

from pydantic import BaseModel

from extensions.const import VertexType
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import Graph, PremiseAudit


def apply_premise_audit(graph: Graph, audit: PremiseAudit, log_file=None) -> Graph | None:
    def log_error(msg: str):
        if log_file is not None:
            print(f"ERROR: {msg}", file=log_file, flush=True)

    graph = deepcopy(graph)
    args_dict = {i: arg for i, arg in enumerate(graph.arguments)}

    for removal in audit.arguments:
        arg_idx = removal.arg_idx
        if arg_idx not in args_dict:
            log_error(f"arg_idx={arg_idx} not found in graph arguments")
            return None
        arg = args_dict[arg_idx]
        for p in removal.remove_premises:
            if p not in arg.premises:
                log_error(f"arg_idx={arg_idx}: premise={p} not found in argument premises {arg.premises}")
                return None
            arg.premises.remove(p)

    # Drop empty arguments last, after all removals are applied
    graph.arguments = [arg for arg in args_dict.values() if arg.premises]

    # Remove background nodes that no longer participate in any argument
    active_idxs = {p for arg in graph.arguments for p in arg.premises} | {arg.claim for arg in graph.arguments}
    graph.nodes = [v for v in graph.nodes if v.type != VertexType.BACKGROUND or v.idx in active_idxs]

    claim_idxs = {v.idx for v in graph.nodes if v.type == VertexType.CLAIM}
    premise_idxs = {p for arg in graph.arguments for p in arg.premises}
    conclusions = [idx for idx in claim_idxs if idx not in premise_idxs]
    if len(conclusions) != 1:
        log_error(f"result has {len(conclusions)} claim conclusions after premise removal: {conclusions}")
        return None

    return graph


class PremiseAuditApplyProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.ungrounded_claims_removed_graph
        audit: PremiseAudit = annotations.premise_audit

        if graph is None or audit is None:
            return None

        return apply_premise_audit(graph, audit, log_file=logger)
