from pydantic import BaseModel

from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.eval import EdgeEvalResult
from extensions.typing.graph import Graph


def _prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class EdgeEvalProcessor(AnnotationProcessor):
    """Compute set-based P/R/F1 on directed edges between human and model graphs."""

    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        human_graph: Graph = annotations.graph_by_human
        model_graph: Graph = annotations.graph_by_model
        if human_graph is None or model_graph is None:
            return None

        human_edges = human_graph.flat_edges()
        model_edges = model_graph.flat_edges()

        human_set = {(e.from_idx, e.to_idx) for e in human_edges}
        model_set = {(e.from_idx, e.to_idx) for e in model_edges}

        human_typed = {(e.from_idx, e.to_idx, e.type) for e in human_edges}
        model_typed = {(e.from_idx, e.to_idx, e.type) for e in model_edges}

        n_correct = len(human_set & model_set)
        n_correct_typed = len(human_typed & model_typed)

        prec, rec, f1 = _prf(n_correct, len(model_set), len(human_set))
        tprec, trec, tf1 = _prf(n_correct_typed, len(model_typed), len(human_typed))

        return EdgeEvalResult(
            edge_precision=prec,
            edge_recall=rec,
            edge_f1=f1,
            edge_type_precision=tprec,
            edge_type_recall=trec,
            edge_type_f1=tf1,
            n_human_edges=len(human_set),
            n_model_edges=len(model_set),
            n_correct_edges=n_correct,
            n_correct_typed_edges=n_correct_typed,
        )
