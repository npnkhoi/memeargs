from __future__ import annotations

from extensions.typing import Annotations
from extensions.typing.eval import MatchingGEDResult
from extensions.base import AnnotationProcessor
from extensions.processors.gemini_graph_auto_eval import _matching_result_to_pairs


class MatchingGEDProcessor(AnnotationProcessor):
    """
    Compute matching-aware GED given a node alignment (MatchingResult).

    Given the alignment M: model_node → human_node, the edit sequence is:
      1. Merge many-to-one model nodes into a single node (free step; their edges are unioned).
      2. Delete truly unmatched model nodes (and their incident edges).
      3. Insert human nodes absent from the alignment.
      4. Delete projected model edges that have no counterpart in the human graph.
      5. Insert human edges absent from the projected model edge set.

    This is deterministic and O(N + E) once the matching is known.
    """

    def process(self, annotations: Annotations, logger=None) -> MatchingGEDResult | None:
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

        # --- node operations ---
        # Many-to-one groups are merged into a single node (free preprocessing).
        # Only truly unmatched model nodes incur a deletion cost.
        model_to_human: dict[int, int] = {m: h for h, m in matched}
        model_matched = set(model_to_human.keys())
        human_matched = {h for h, _ in matched}

        n_node_del = len(g_model.nodes) - len(model_matched)
        n_node_ins = len(g_human.nodes) - len(human_matched)

        # --- edge operations ---
        # After merging, all matched model nodes (including surplus) contribute
        # their edges to the projected set — duplicates collapse via set dedup.
        human_edge_set = {(e.from_idx, e.to_idx) for e in g_human.flat_edges()}

        n_edge_del = 0
        projected_edge_set: set[tuple[int, int]] = set()

        for edge in g_model.flat_edges():
            u, v = edge.from_idx, edge.to_idx
            if u in model_to_human and v in model_to_human:
                projected_edge_set.add((model_to_human[u], model_to_human[v]))
            else:
                n_edge_del += 1  # edge of a truly unmatched node

        n_edge_del += len(projected_edge_set - human_edge_set)
        n_edge_ins = len(human_edge_set - projected_edge_set)

        ged = n_node_del + n_node_ins + n_edge_del + n_edge_ins
        print(
            f"Matching-aware GED: {ged} "
            f"(node_del={n_node_del}, node_ins={n_node_ins}, "
            f"edge_del={n_edge_del}, edge_ins={n_edge_ins})"
        )

        return MatchingGEDResult(
            ged=ged,
            n_node_del=n_node_del,
            n_node_ins=n_node_ins,
            n_edge_del=n_edge_del,
            n_edge_ins=n_edge_ins,
        )
