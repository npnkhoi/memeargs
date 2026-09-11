from __future__ import annotations

import networkx as nx
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from extensions.typing import Annotations
from extensions.typing.eval import GraphAutoEvalResult
from extensions.base import AnnotationProcessor
from extensions.typing.graph import Graph
from extensions.const import VertexType

# ---------------------------------------------------------------------------
# NLI model (lazy singleton)
# ---------------------------------------------------------------------------
_nli_tokenizer = None
_nli_model = None
_nli_device = None

NLI_MODEL_NAME = "potsawee/deberta-v3-large-mnli"


def _get_nli_model():
    """Lazy-load the NLI model (singleton, reused across calls)."""
    global _nli_tokenizer, _nli_model, _nli_device
    if _nli_model is None:
        _nli_device = "cuda"
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(_nli_device)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model, _nli_device


def _nli_entailment_score(premise: str, hypothesis: str) -> float:
    """Return P(entailment) for a single (premise, hypothesis) pair."""
    tokenizer, model, device = _get_nli_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # [entailment, contradiction] (no neutral head)
    probs = torch.softmax(logits, dim=-1)
    return probs[0, 0].item()  # entailment index = 0


def get_similarity_score(text_a: str, text_b: str) -> float:
    """Bidirectional NLI entailment score: avg of both directions."""
    return (_nli_entailment_score(text_a, text_b) + _nli_entailment_score(text_b, text_a)) / 2.0


# ---------------------------------------------------------------------------
# NLI similarity matrix
# ---------------------------------------------------------------------------

def build_nli_similarity_matrix(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """
    Compute pairwise bidirectional NLI entailment scores.
    Returns ndarray of shape [len(texts_a), len(texts_b)].
    """
    m, n = len(texts_a), len(texts_b)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    sim = np.zeros((m, n), dtype=np.float64)
    for i, a in enumerate(texts_a):
        for j, b in enumerate(texts_b):
            sim[i, j] = get_similarity_score(a, b)
    return sim



# ---------------------------------------------------------------------------
# Graph-based auto evaluation (greedy match + F1 + GED)
# ---------------------------------------------------------------------------

NODE_MATCH_THRESHOLD = 0.7
GED_TIMEOUT_SECONDS = 10


def greedy_match(sim: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """
    Greedy 1-to-1 matching in descending similarity order.
    Rows index human nodes, columns index model nodes.
    Returns list of (human_idx, model_idx) pairs where sim >= threshold.
    """
    if sim.size == 0:
        return []
    m, n = sim.shape
    pairs = sorted(
        ((i, j, sim[i, j]) for i in range(m) for j in range(n)),
        key=lambda x: x[2],
        reverse=True,
    )
    matched: list[tuple[int, int]] = []
    used_human: set[int] = set()
    used_model: set[int] = set()
    for i, j, score in pairs:
        if score < threshold:
            break
        if i not in used_human and j not in used_model:
            matched.append((i, j))
            used_human.add(i)
            used_model.add(j)
    return matched


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


def compute_graph_edit_distance(g_human: Graph, g_model: Graph) -> float | None:
    """
    Graph edit distance from g_model to g_human using networkx.
    Returns None if computation times out or an error occurs.
    """
    def _to_nx(g: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for v in g.nodes:
            G.add_node(v.idx)
        for e in g.flat_edges():
            G.add_edge(e.from_idx, e.to_idx)
        return G

    if not g_human.nodes and not g_model.nodes:
        return 0.0

    try:
        ged = nx.graph_edit_distance(
            _to_nx(g_human), _to_nx(g_model),
            timeout=GED_TIMEOUT_SECONDS,
        )
        return float(ged) if ged is not None else None
    except Exception:
        return None


class GraphAutoEvalProcessor(AnnotationProcessor):
    """
    Graph-based automatic evaluation using greedy NLI matching + F1 + GED.
    Fixed threshold t=0.7 for node matching.
    """

    def __init__(self, pipeline, stage_config):
        super().__init__(pipeline, stage_config)

    def process(self, annotations: Annotations, logger=None) -> GraphAutoEvalResult | None:
        g_model = annotations.graph_by_model
        g_human = annotations.graph_by_human
        if g_model is None or g_human is None:
            return None

        texts_human = [v.text for v in g_human.nodes]
        texts_model = [v.text for v in g_model.nodes]

        # Step 1: NLI similarity matrix
        sim = build_nli_similarity_matrix(texts_human, texts_model)

        # Step 2: Greedy node matching
        matched = greedy_match(sim, NODE_MATCH_THRESHOLD) # {(i, j)}, where i and j are positions in g_human.nodes and g_model.nodes, respectively
        print(f"Found {len(matched)} matched nodes.")

        # Step 3: Node F1
        node_p, node_r, node_f1 = compute_node_f1(
            matched, len(g_human.nodes), len(g_model.nodes)
        )
        print(f"Node F1: precision={node_p:.3f}, recall={node_r:.3f}, f1={node_f1:.3f}")

        # Step 3b: Per-type node F1
        elem_p, elem_r, elem_f1, n_h_elem, n_m_elem, n_match_model_elem, n_match_human_elem = compute_node_f1_by_type(matched, g_human, g_model, VertexType.ELEMENT)
        bg_p, bg_r, bg_f1, n_h_bg, n_m_bg, n_match_model_bg, n_match_human_bg = compute_node_f1_by_type(matched, g_human, g_model, VertexType.BACKGROUND)
        claim_p, claim_r, claim_f1, n_h_claim, n_m_claim, n_match_model_claim, n_match_human_claim = compute_node_f1_by_type(matched, g_human, g_model, VertexType.CLAIM)

        # Step 4: Edge F1 conditioned on node matching
        edge_p, edge_r, edge_f1, n_matched_edges = compute_edge_f1(
            matched, g_human, g_model
        )
        print(f"Edge F1: precision={edge_p:.3f}, recall={edge_r:.3f}, f1={edge_f1:.3f}")

        # Step 5: Graph edit distance
        ged = compute_graph_edit_distance(g_human, g_model)
        print(f"Graph edit distance: {ged if ged is not None else 'N/A'}")

        return GraphAutoEvalResult(
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
            edge_precision=edge_p,
            edge_recall=edge_r,
            edge_f1=edge_f1,
            graph_edit_distance=ged,
            node_threshold=NODE_MATCH_THRESHOLD,
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
            n_human_edges=len(list(g_human.flat_edges())),
            n_model_edges=len(list(g_model.flat_edges())),
            n_matched_edges=n_matched_edges,
            matched_pairs=matched,
            similarity_matrix=sim.tolist(),
        )
