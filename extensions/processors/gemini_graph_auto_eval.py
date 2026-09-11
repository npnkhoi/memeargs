from __future__ import annotations

from extensions.typing import Annotations
from extensions.typing.eval import GeminiGraphAutoEvalResult, MatchingResult
from extensions.base import AnnotationProcessor
from extensions.processors.graph_auto_eval import (
    compute_node_f1,
    compute_node_f1_by_type,
    compute_edge_f1,
)
from extensions.const import VertexType
from extensions.typing.graph import Graph


# ---------------------------------------------------------------------------
# Helper: translate MatchingResult per-type sublist indices → positional indices
# ---------------------------------------------------------------------------

def _matching_result_to_pairs(
    matching: MatchingResult,
    g_human: Graph,
    g_model: Graph,
) -> list[tuple[int, int]]:
    """
    Translate per-type sublist indices from MatchingResult into positional
    indices in g.nodes.  Supports many-to-one (multiple model nodes → one
    human node).  Drops out-of-bounds entries; each model node at most once.
    """
    matched: list[tuple[int, int]] = []

    type_pairs = [
        (VertexType.ELEMENT, matching.element_matches),
        (VertexType.PREMISE, matching.premise_matches),
        (VertexType.BACKGROUND, matching.background_matches),
        (VertexType.CLAIM, matching.claim_matches),
    ]

    seen_model: set[int] = set()

    for vtype, pairs in type_pairs:
        human_nodes = g_human.get_nodes_by_type(vtype)
        model_nodes = g_model.get_nodes_by_type(vtype)

        for pair in pairs:
            hi = pair.human_idx
            if hi < 0 or hi >= len(human_nodes):
                continue
            h_pos = human_nodes[hi].idx

            for mi in pair.model_idxs:
                if mi < 0 or mi >= len(model_nodes):
                    continue
                m_pos = model_nodes[mi].idx
                if m_pos in seen_model:
                    continue
                matched.append((h_pos, m_pos))
                seen_model.add(m_pos)

    return matched


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class GeminiGraphAutoEvalProcessor(AnnotationProcessor):
    """
    Deterministic processor that computes GeminiGraphAutoEvalResult from a
    MatchingResult (produced by an LLMProcessor step) and the two graphs.
    """

    def process(self, annotations: Annotations, logger=None) -> GeminiGraphAutoEvalResult | None:
        g_model = annotations.graph_by_model
        g_human = annotations.graph_by_human
        matching = annotations.matching_result
        if g_model is None or g_human is None or matching is None:
            return None

        g_human.normalize()
        g_model.normalize()
        g_human.derive_premise_types()
        g_model.derive_premise_types()

        matched = _matching_result_to_pairs(matching, g_human, g_model)
        print(f"Found {len(matched)} matched nodes.")

        node_p, node_r, node_f1 = compute_node_f1(
            matched, len(g_human.nodes), len(g_model.nodes)
        )
        print(f"Node F1: precision={node_p:.3f}, recall={node_r:.3f}, f1={node_f1:.3f}")

        elem_p, elem_r, elem_f1, n_h_elem, n_m_elem, n_match_model_elem, n_match_human_elem = compute_node_f1_by_type(
            matched, g_human, g_model, VertexType.ELEMENT
        )
        bg_p, bg_r, bg_f1, n_h_bg, n_m_bg, n_match_model_bg, n_match_human_bg = compute_node_f1_by_type(
            matched, g_human, g_model, VertexType.BACKGROUND
        )
        claim_p, claim_r, claim_f1, n_h_claim, n_m_claim, n_match_model_claim, n_match_human_claim = compute_node_f1_by_type(
            matched, g_human, g_model, VertexType.CLAIM
        )
        prem_p, prem_r, prem_f1, n_h_prem, n_m_prem, n_match_model_prem, n_match_human_prem = compute_node_f1_by_type(
            matched, g_human, g_model, VertexType.PREMISE
        )

        edge_p, edge_r, edge_f1, n_matched_edges = compute_edge_f1(
            matched, g_human, g_model
        )
        print(f"Edge F1: precision={edge_p:.3f}, recall={edge_r:.3f}, f1={edge_f1:.3f}")

        return GeminiGraphAutoEvalResult(
            node_precision=node_p,
            node_recall=node_r,
            node_f1=node_f1,
            element_node_precision=elem_p,
            element_node_recall=elem_r,
            element_node_f1=elem_f1,
            background_node_precision=bg_p,
            background_node_recall=bg_r,
            background_node_f1=bg_f1,
            claim_node_precision=claim_p,
            claim_node_recall=claim_r,
            claim_node_f1=claim_f1,
            premise_node_precision=prem_p,
            premise_node_recall=prem_r,
            premise_node_f1=prem_f1,
            edge_precision=edge_p,
            edge_recall=edge_r,
            edge_f1=edge_f1,
            graph_edit_distance=None,
            node_threshold=0.0,
            n_human_nodes=len(g_human.nodes),
            n_model_nodes=len(g_model.nodes),
            n_matched_model_nodes=len(matched),
            n_matched_human_nodes=len({h for h, _ in matched}),
            n_human_element_nodes=n_h_elem,
            n_model_element_nodes=n_m_elem,
            n_matched_model_element_nodes=n_match_model_elem,
            n_matched_human_element_nodes=n_match_human_elem,
            n_human_background_nodes=n_h_bg,
            n_model_background_nodes=n_m_bg,
            n_matched_model_background_nodes=n_match_model_bg,
            n_matched_human_background_nodes=n_match_human_bg,
            n_human_claim_nodes=n_h_claim,
            n_model_claim_nodes=n_m_claim,
            n_matched_model_claim_nodes=n_match_model_claim,
            n_matched_human_claim_nodes=n_match_human_claim,
            n_human_premise_nodes=n_h_prem,
            n_model_premise_nodes=n_m_prem,
            n_matched_model_premise_nodes=n_match_model_prem,
            n_matched_human_premise_nodes=n_match_human_prem,
            n_human_edges=len(list(g_human.flat_edges())),
            n_model_edges=len(list(g_model.flat_edges())),
            n_matched_edges=n_matched_edges,
            matched_pairs=matched,
        )
