from copy import deepcopy

import networkx as nx
from pydantic import BaseModel

from extensions.const import EdgeType, VertexType
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import Argument, ArgumentAudit, Graph


def _build_nx(graph: Graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.idx)
    for arg in graph.arguments:
        for p in arg.premises:
            G.add_edge(p, arg.claim)
    return G


def _is_valid_premise(graph: Graph, node_idx: int) -> bool:
    """Element, Background, or Claim nodes may be added as premises; node must exist."""
    node_map = {v.idx: v for v in graph.nodes}
    return node_idx in node_map and node_map[node_idx].type != VertexType.NONE



def _would_create_cycle(G: nx.DiGraph, claim_idx: int, premise_idx: int) -> bool:
    """True if adding premise_idx -> claim_idx would create a cycle."""
    try:
        return nx.has_path(G, claim_idx, premise_idx)
    except nx.NetworkXError:
        return False


def apply_argument_audit(graph: Graph, audit: ArgumentAudit, log_file=None) -> Graph | None:
    def log_error(msg: str):
        if log_file is not None:
            print(f"ERROR: {msg}", file=log_file, flush=True)
    graph = deepcopy(graph)
    node_idxs = {v.idx for v in graph.nodes}
    claim_idxs = {arg.claim for arg in graph.arguments} | set(graph.get_conclusions())

    # Build a dict keyed by original argument index so that all arg_idx references
    # in the audit remain valid even after merges remove entries.
    # None marks a slot that was consumed by a merge.
    args_dict: dict[int, Argument | None] = {i: arg for i, arg in enumerate(graph.arguments)}
    # Newly created arguments (from add actions) are collected separately; they are
    # appended after the original slots are rebuilt.
    added_args: list[Argument] = []

    def _current_args() -> list[Argument]:
        """Live view of all arguments for skip-edge / cycle checks."""
        return [a for a in args_dict.values() if a is not None] + added_args

    for claim_audit in audit.claims:
        claim_idx = claim_audit.claim_idx
        if claim_idx not in node_idxs:
            log_error(f"claim_idx={claim_idx} not found in graph nodes")
            return None
        if claim_idx not in claim_idxs:
            log_error(f"claim_idx={claim_idx} is not a claim node")
            return None

        expands = [a for a in claim_audit.actions if a.type == "expand"]
        adds    = [a for a in claim_audit.actions if a.type == "add"]
        merges  = [a for a in claim_audit.actions if a.type == "merge"]

        # --- expand ---
        for action in expands:
            arg_idx = action.arg_idx
            node_idx = action.node_idx
            if arg_idx not in args_dict or args_dict[arg_idx] is None:
                log_error(f"expand action for claim {claim_idx}: arg_idx={arg_idx} not found (out of range or already merged)")
                return None
            if args_dict[arg_idx].claim != claim_idx:
                log_error(f"expand action for claim {claim_idx}: arg_idx={arg_idx} has claim={args_dict[arg_idx].claim}, expected {claim_idx}")
                return None
            if not _is_valid_premise(graph, node_idx):
                log_error(f"expand action for claim {claim_idx}: node_idx={node_idx} is not a valid premise node")
                return None
            cur = _current_args()
            graph.arguments = cur
            G = _build_nx(graph)
            if _would_create_cycle(G, claim_idx, node_idx):
                log_error(f"expand action for claim {claim_idx}: adding node_idx={node_idx} would create a cycle")
                return None
            if node_idx not in args_dict[arg_idx].premises:
                args_dict[arg_idx].premises.append(node_idx)

        # --- add ---
        for action in adds:
            cur = _current_args()
            graph.arguments = cur
            G = _build_nx(graph)
            valid_premises = []
            for p in action.premises:
                if not _is_valid_premise(graph, p):
                    log_error(f"add action for claim {claim_idx}: premise={p} is not a valid premise node")
                    return None
                if _would_create_cycle(G, claim_idx, p):
                    log_error(f"add action for claim {claim_idx}: premise={p} would create a cycle")
                    return None
                valid_premises.append(p)
            if not valid_premises:
                log_error(f"add action for claim {claim_idx}: no valid premises after validation")
                return None
            added_args.append(Argument(claim=claim_idx, premises=valid_premises, type=EdgeType.SUPPORT))

        # --- merge (runs after expand/add so their results are included) ---
        # merged_into tracks consumed slots: merged_into[removed] = absorbing slot.
        # Follow the chain so that e.g. merge(a,b) then merge(b,c) correctly finds
        # b's replacement when processing the second merge.
        merged_into: dict[int, int] = {}

        def _resolve(idx: int) -> int:
            while idx in merged_into:
                idx = merged_into[idx]
            return idx

        for action in merges:
            if len(action.arg_idxs) < 2:
                log_error(f"merge action for claim {claim_idx}: arg_idxs must have >= 2 entries, got {action.arg_idxs}")
                return None
            resolved = [_resolve(idx) for idx in action.arg_idxs]
            for orig, res in zip(action.arg_idxs, resolved):
                if res not in args_dict or args_dict[res] is None:
                    log_error(f"merge action for claim {claim_idx}: arg_idx={orig} (resolved={res}) not found")
                    return None
                if args_dict[res].claim != claim_idx:
                    log_error(f"merge action for claim {claim_idx}: arg_idx={orig} (resolved={res}) has claim={args_dict[res].claim}, expected {claim_idx}")
                    return None
            # Union premises across all resolved slots, deduplicate, preserve order
            merged_premises: list[int] = []
            seen: set[int] = set()
            for res in resolved:
                for p in args_dict[res].premises:
                    if p not in seen:
                        merged_premises.append(p)
                        seen.add(p)
            keep = resolved[0]
            args_dict[keep] = Argument(claim=claim_idx, premises=merged_premises, type=EdgeType.SUPPORT)
            for res in resolved[1:]:
                args_dict[res] = None
                merged_into[res] = keep

    graph.arguments = [a for a in args_dict.values() if a is not None] + added_args
    return graph


class ArgumentAuditApplyProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.element_audited_graph
        audit: ArgumentAudit = annotations.argument_audit

        if graph is None or audit is None:
            return None

        return apply_argument_audit(graph, audit, log_file=logger)
