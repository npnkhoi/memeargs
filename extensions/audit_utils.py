from copy import deepcopy

from extensions.const import VertexType
from extensions.typing.graph import Argument, ElementAudit, Graph, Vertex


def audit_has_edge_cases(graph: Graph, audit: ElementAudit, log_file=None) -> bool:
    """Return True if any verdict in the audit would trigger a silently-skipped edge case."""
    def log_error(msg: str):
        if log_file is not None:
            print(f"ERROR: {msg}", file=log_file, flush=True)

    node_idxs = {v.idx for v in graph.nodes}
    claim_idxs = {arg.claim for arg in graph.arguments}
    for verdict in audit.existing:
        idx = verdict.node_idx
        if idx not in node_idxs:
            log_error(f"node_idx={idx} not found in graph nodes")
            return True
        if verdict.action == "delete" and idx in claim_idxs:
            log_error(f"node_idx={idx} is a claim node, cannot delete")
            return True
    return False


def apply_element_audit(graph: Graph, audit: ElementAudit) -> Graph:
    """Return a new Graph with ElementAudit verdicts applied.

    Actions applied:
      correct  — node unchanged
      edit     — node text replaced with new_content
      retype   — node type changed to new_type
      delete   — node removed; arguments that lose all premises are dropped

    Edge cases silently skipped (node left unchanged):
      - delete where the node is the claim of some argument
      - node_idx not present in the graph
    """
    graph = deepcopy(graph)

    node_map: dict[int, Vertex] = {v.idx: v for v in graph.nodes}
    claim_idxs: set[int] = {arg.claim for arg in graph.arguments}
    next_idx = max((v.idx for v in graph.nodes), default=-1) + 1

    to_delete: set[int] = set()

    kind_prefix = {"textual": "[Text]", "visual": "[Image]"}

    # Pass 1: process verdicts
    for verdict in audit.existing:
        idx = verdict.node_idx
        if idx not in node_map:
            continue

        action = verdict.action
        prefix = kind_prefix.get(verdict.kind, "")

        if action == "correct":
            node_map[idx].text = f"{prefix} {node_map[idx].text}"

        elif action == "edit":
            node_map[idx].text = f"{prefix} {verdict.new_content}"

        elif action == "retype":
            node_map[idx].type = VertexType(verdict.new_type)

        elif action == "delete":
            if idx in claim_idxs:
                continue  # edge case: skip
            to_delete.add(idx)

    # Pass 2: add missing element nodes (orphans)
    for missing in audit.missing:
        prefix = kind_prefix.get(missing.kind, "")
        graph.nodes.append(Vertex(idx=next_idx, text=f"{prefix} {missing.content}", type=VertexType.ELEMENT))
        next_idx += 1

    # Pass 3: remove deleted nodes
    graph.nodes = [v for v in graph.nodes if v.idx not in to_delete]

    # Pass 4: rebuild arguments
    new_arguments = []
    for arg in graph.arguments:
        if arg.claim in to_delete:
            continue
        new_premises = [p for p in arg.premises if p not in to_delete]
        if not new_premises:
            continue
        new_arguments.append(Argument(claim=arg.claim, premises=new_premises, type=arg.type))
    graph.arguments = new_arguments

    # Pass 5: compact indices
    graph.normalize()
    return graph
