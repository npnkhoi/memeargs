"""Shared node/edge F1 helpers, used by gemini_graph_auto_eval.py (which
builds the node alignment via Gemini rather than the NLI greedy-matching this
module used to also implement)."""

from __future__ import annotations

from extensions.typing.graph import Graph
from extensions.const import VertexType


def _compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from TP/FP/FN. Returns 0.0 when undefined."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def compute_node_f1(
    matched: list[tuple[int, int]],
    n_human: int,
    n_model: int,
) -> tuple[float, float, float]:
    """Node precision, recall, F1 with merge semantics.

    Many-to-one groups (A,B,C → H) are collapsed to one effective model node.
      effective_n_model = n_model - (n_matched_model - n_matched_human)
      TP  = n_matched_human
      FP  = effective_n_model - TP  =  n_model - n_matched_model
      FN  = n_human - n_matched_human
    For 1-to-1 matching this reduces to the standard formula.
    """
    n_matched_model = len(matched)
    n_matched_human = len({h for h, _ in matched})
    tp = n_matched_human
    fp = n_model - n_matched_model
    fn = n_human - n_matched_human
    return _compute_f1(tp, fp, fn)


def compute_node_f1_by_type(
    matched: list[tuple[int, int]],
    g_human: Graph,
    g_model: Graph,
    vtype: VertexType,
) -> tuple[float, float, float, int, int, int, int]:
    """Node F1 for a specific vertex type with merge semantics.
    Returns (precision, recall, f1, n_human, n_model, n_matched_model, n_matched_human).
    """
    human_type_idxs = {v.idx for v in g_human.get_nodes_by_type(vtype)}
    model_type_idxs = {v.idx for v in g_model.get_nodes_by_type(vtype)}
    type_matched = [
        (h, m) for h, m in matched
        if h in human_type_idxs and m in model_type_idxs
    ]
    n_matched_model = len(type_matched)
    n_matched_human = len({h for h, _ in type_matched})
    n_human = len(human_type_idxs)
    n_model = len(model_type_idxs)
    tp = n_matched_human
    fp = n_model - n_matched_model
    fn = n_human - n_matched_human
    p, r, f1 = _compute_f1(tp, fp, fn)
    return p, r, f1, n_human, n_model, n_matched_model, n_matched_human


def compute_edge_f1(
    matched_nodes: list[tuple[int, int]],
    g_human: Graph,
    g_model: Graph,
) -> tuple[float, float, float, int]:
    """Edge F1 with merge-then-deduplicate semantics.

    Each matched model node is replaced by its human representative.
    Model edges are deduplicated after substitution (e.g. A→D and B→D become
    the same edge when A,B are merged).  The deduplicated set is then compared
    to the human edge set normally.
    Returns (precision, recall, f1, n_matched_edges).
    """
    model_to_human: dict[int, int] = {m: h for h, m in matched_nodes}

    # Representative for a model node endpoint:
    #   matched   → ("h", human_pos)  — can be compared to human edges
    #   unmatched → ("m", model_pos)  — distinct namespace, never matches human edge
    def rep(m_pos: int):
        if m_pos in model_to_human:
            return ("h", model_to_human[m_pos])
        return ("m", m_pos)

    human_edge_set: set[tuple[int, int]] = {
        (e.from_idx, e.to_idx) for e in g_human.flat_edges()
    }

    # Build deduplicated model edge set using representatives
    model_edge_set_dedup: set = {
        (rep(e.from_idx), rep(e.to_idx)) for e in g_model.flat_edges()
    }

    n_matched = sum(
        1 for (r_from, r_to) in model_edge_set_dedup
        if r_from[0] == "h" and r_to[0] == "h"
        and (r_from[1], r_to[1]) in human_edge_set
    )

    tp = n_matched
    fp = len(model_edge_set_dedup) - tp
    fn = len(human_edge_set) - tp
    precision, recall, f1 = _compute_f1(tp, fp, fn)
    return precision, recall, f1, n_matched
