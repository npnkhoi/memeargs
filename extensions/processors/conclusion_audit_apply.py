from copy import deepcopy

from pydantic import BaseModel

from extensions.base import AnnotationProcessor
from extensions.processors.claim_merger import merge_pair
from extensions.typing import Annotations
from extensions.typing.graph import ConclusionAudit, Graph


def _get_conclusion_idx(graph: Graph) -> int:
    premise_idxs = {p for arg in graph.arguments for p in arg.premises}
    node_idxs = {n.idx for n in graph.nodes}
    conclusions = node_idxs - premise_idxs
    assert len(conclusions) == 1, f"Expected 1 conclusion, got {conclusions}"
    return next(iter(conclusions))


def apply_conclusion_audit(graph: Graph, audit: ConclusionAudit, log_file=None) -> Graph | None:
    """Return transformed graph, or None if verdict is brand_new or found_in_claims with non-direct child."""
    def log_error(msg: str):
        if log_file is not None:
            print(f"ERROR: {msg}", file=log_file, flush=True)

    if audit.verdict == "brand_new":
        log_error("verdict is brand_new, skipping")
        return None

    graph = deepcopy(graph)

    if audit.verdict == "in_place":
        concl_idx = _get_conclusion_idx(graph)
        for node in graph.nodes:
            if node.idx == concl_idx:
                node.text = audit.new_conclusion
                break

    elif audit.verdict == "found_in_claims":
        concl_idx = _get_conclusion_idx(graph)
        correct_idx = audit.correct_conclusion_idx

        # collect children from all arguments leading to the conclusion
        direct_children = {p for arg in graph.arguments if arg.claim == concl_idx for p in arg.premises}

        # check if correct_idx is not a child of the conclusion. if not, fail this stage
        if correct_idx not in direct_children:
            log_error(f"correct_conclusion_idx={correct_idx} is not a direct child of the conclusion node {concl_idx}")
            return None

        # Record the child's sub-arg premise sets before merge (will be redundant after).
        child_sub_sets = {
            frozenset(arg.premises)
            for arg in graph.arguments
            if arg.claim == correct_idx
        }

        # The child (correct_idx) is the semantic winner: copy its text to the parent node.
        correct_text = next(n.text for n in graph.nodes if n.idx == correct_idx)
        for node in graph.nodes:
            if node.idx == concl_idx:
                node.text = correct_text
                break

        try:
            graph = merge_pair(graph, keep=concl_idx, remove=correct_idx)
        except ValueError as e:
            log_error(f"merge_pair failed: {e}")
            return None

    # "correct": no modification needed

    return graph


class ConclusionAuditApplyProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.graph
        audit: ConclusionAudit = annotations.conclusion_audit

        if graph is None or audit is None:
            return None

        return apply_conclusion_audit(graph, audit, log_file=logger)
