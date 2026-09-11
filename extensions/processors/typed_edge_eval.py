"""TypedEdgeEvalProcessor: edge P/R/F1 split into *logos* and *trigger* buckets.

The Trigger edge type from the paper is not stored in the graph; it is derived:
an edge is a **trigger** edge iff its source (premise) node is an Element, and a
**logos** edge otherwise (a Claim/Background premise; support and attack merged).
We score the two buckets separately, the same way node F1 is computed per node
type and then macro-averaged.

Matching mirrors ``compute_edge_f1``: model edges are projected through the node
alignment (from ``MatchingResult``) and deduplicated, then compared to the gold
edge set within each bucket. If no ``MatchingResult`` is present (edge
construction, where nodes are gold), the alignment is the identity on shared
node indices.
"""

from extensions.base import AnnotationProcessor
from extensions.const import VertexType
from extensions.typing import Annotations
from extensions.typing.eval import TypedEdgeEvalResult
from extensions.processors.gemini_graph_auto_eval import _matching_result_to_pairs


def _prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _bucket(node_type) -> str:
    return "trigger" if node_type == VertexType.ELEMENT else "logos"


def typed_edge_scores(matched_pairs, g_human, g_model) -> dict[str, tuple]:
    """Return {'logos': (p, r, f1, tp, n_pred, n_true), 'trigger': (...)}.

    ``matched_pairs``: list[(human_node_idx, model_node_idx)].
    """
    model_to_human = {m: h for h, m in matched_pairs}
    h_type = {v.idx: v.type for v in g_human.nodes}
    m_type = {v.idx: v.type for v in g_model.nodes}

    human = {"logos": set(), "trigger": set()}
    for e in g_human.flat_edges():
        human[_bucket(h_type.get(e.from_idx))].add((e.from_idx, e.to_idx))

    def rep(m: int):
        return ("h", model_to_human[m]) if m in model_to_human else ("m", m)

    proj = {"logos": set(), "trigger": set()}
    for e in g_model.flat_edges():
        rf, rt = rep(e.from_idx), rep(e.to_idx)
        src_type = h_type.get(rf[1]) if rf[0] == "h" else m_type.get(rf[1])
        proj[_bucket(src_type)].add((rf, rt))

    out = {}
    for b in ("logos", "trigger"):
        tp = sum(
            1 for (rf, rt) in proj[b]
            if rf[0] == "h" and rt[0] == "h" and (rf[1], rt[1]) in human[b]
        )
        n_pred, n_true = len(proj[b]), len(human[b])
        out[b] = (*_prf(tp, n_pred, n_true), tp, n_pred, n_true)
    return out


class TypedEdgeEvalProcessor(AnnotationProcessor):
    """Compute logos / trigger edge P/R/F1 (+ their macro average)."""

    def process(self, annotations: Annotations, logger=None) -> TypedEdgeEvalResult | None:
        g_human = annotations.graph_by_human
        g_model = annotations.graph_by_model
        if g_human is None or g_model is None:
            return None

        g_human.normalize()
        g_model.normalize()
        g_human.derive_premise_types()
        g_model.derive_premise_types()

        matching = annotations.matching_result
        if matching is not None:
            matched = _matching_result_to_pairs(matching, g_human, g_model)
        else:  # edge construction: nodes are gold, alignment is the identity
            shared = {v.idx for v in g_human.nodes} & {v.idx for v in g_model.nodes}
            matched = [(i, i) for i in shared]

        s = typed_edge_scores(matched, g_human, g_model)
        lp, lr, lf, l_tp, l_np, l_nt = s["logos"]
        tp_, tr, tf, t_tp, t_np, t_nt = s["trigger"]
        macro_p, macro_r, macro_f = ((lp + tp_) / 2, (lr + tr) / 2, (lf + tf) / 2)
        print(
            f"Logos edge: P={lp:.3f} R={lr:.3f} F1={lf:.3f}  |  "
            f"Trigger edge: P={tp_:.3f} R={tr:.3f} F1={tf:.3f}"
        )

        return TypedEdgeEvalResult(
            logos_edge_precision=lp, logos_edge_recall=lr, logos_edge_f1=lf,
            trigger_edge_precision=tp_, trigger_edge_recall=tr, trigger_edge_f1=tf,
            macro_edge_precision=macro_p, macro_edge_recall=macro_r, macro_edge_f1=macro_f,
            n_human_logos_edges=l_nt, n_model_logos_edges=l_np, n_matched_logos_edges=l_tp,
            n_human_trigger_edges=t_nt, n_model_trigger_edges=t_np, n_matched_trigger_edges=t_tp,
        )
