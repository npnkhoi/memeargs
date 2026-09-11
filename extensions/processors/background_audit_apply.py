from copy import deepcopy

from pydantic import BaseModel

from extensions.const import VertexType
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import BackgroundAudit, Graph, Vertex


def apply_background_audit(graph: Graph, audit: BackgroundAudit, log_file=None) -> Graph | None:
    def log_error(msg: str):
        if log_file is not None:
            print(f"ERROR: {msg}", file=log_file, flush=True)

    graph = deepcopy(graph)
    node_map = {v.idx: v for v in graph.nodes}
    args_dict = {i: arg for i, arg in enumerate(graph.arguments)}

    # --- edit ---
    for action in audit.edit:
        node_idx = action.node_idx
        if node_idx not in node_map:
            log_error(f"edit: node_idx={node_idx} not found in graph nodes")
            return None
        if node_map[node_idx].type != VertexType.BACKGROUND:
            log_error(f"edit: node_idx={node_idx} is not a background node")
            return None
        node_map[node_idx].text = action.new_content

    # --- remove ---
    to_remove: set[int] = set()
    for action in audit.remove:
        node_idx = action.node_idx
        if node_idx not in node_map:
            log_error(f"remove: node_idx={node_idx} not found in graph nodes")
            return None
        if node_map[node_idx].type != VertexType.BACKGROUND:
            log_error(f"remove: node_idx={node_idx} is not a background node")
            return None
        to_remove.add(node_idx)

    for arg in args_dict.values():
        arg.premises = [p for p in arg.premises if p not in to_remove]
    graph.nodes = [v for v in graph.nodes if v.idx not in to_remove]
    node_map = {v.idx: v for v in graph.nodes}

    # Drop arguments emptied by removal
    args_dict = {i: arg for i, arg in args_dict.items() if arg.premises}

    # --- add ---
    next_idx = max((v.idx for v in graph.nodes), default=-1) + 1
    for action in audit.add:
        if action.arg_idx not in args_dict:
            log_error(f"add: arg_idx={action.arg_idx} not found in graph arguments")
            return None
        new_node = Vertex(idx=next_idx, text=action.content, type=VertexType.BACKGROUND)
        graph.nodes.append(new_node)
        node_map[next_idx] = new_node
        args_dict[action.arg_idx].premises.append(next_idx)
        next_idx += 1

    graph.arguments = list(args_dict.values())

    claim_idxs = {v.idx for v in graph.nodes if v.type == VertexType.CLAIM}
    premise_idxs = {p for arg in graph.arguments for p in arg.premises}
    conclusions = [idx for idx in claim_idxs if idx not in premise_idxs]
    if len(conclusions) != 1:
        log_error(f"result has {len(conclusions)} claim conclusions after background audit: {conclusions}")
        return None

    return graph


class BackgroundAuditApplyProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.premise_audited_graph
        audit: BackgroundAudit = annotations.background_audit

        if graph is None or audit is None:
            return None

        return apply_background_audit(graph, audit, log_file=logger)
