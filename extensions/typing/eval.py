from pydantic import BaseModel


class NodeMatchPair(BaseModel):
    human_idx: int        # 0-based index within the per-type sublist (unique per list)
    model_idxs: list[int] # one or more model nodes that together map to this human node


class MatchingResult(BaseModel):
    """LLM-generated node matching between human and model graphs.

    Each entry maps one human node to one or more model nodes (many-to-one).
    A model node may appear in at most one entry per type list.
    """
    element_matches: list[NodeMatchPair]
    premise_matches: list[NodeMatchPair] = []
    background_matches: list[NodeMatchPair]
    claim_matches: list[NodeMatchPair]


class StructuralIssues(BaseModel):
    errors: list[str]
    warnings: list[str]



class SoftAspectEvaluation(BaseModel):
    reason: str
    score: int | None = None

class AspectEvaluation(SoftAspectEvaluation):
    # Override score to be required and stricter for the hard evaluation.
    score: int  # 0-4

ConclusionEvaluation = AspectEvaluation
ElementEvaluation = AspectEvaluation
BackgroundEvaluation = AspectEvaluation
ClaimEvaluation = AspectEvaluation
RelationEvaluation = AspectEvaluation
OverallEvaluation = AspectEvaluation

class Evaluation(BaseModel):
    element: SoftAspectEvaluation
    background: SoftAspectEvaluation
    conclusion: SoftAspectEvaluation
    claim: SoftAspectEvaluation
    relation: SoftAspectEvaluation
    overall: SoftAspectEvaluation

ConclusionEvaluationByGemini = AspectEvaluation
ElementEvaluationByGemini = AspectEvaluation
BackgroundEvaluationByGemini = AspectEvaluation
ClaimEvaluationByGemini = AspectEvaluation
RelationEvaluationByGemini = AspectEvaluation
OverallEvaluationByGemini = AspectEvaluation

class EvaluationByGemini(BaseModel):
    element: SoftAspectEvaluation
    background: SoftAspectEvaluation
    conclusion: SoftAspectEvaluation
    claim: SoftAspectEvaluation
    relation: SoftAspectEvaluation
    overall: SoftAspectEvaluation

class GraphCheckResult(BaseModel):
    errors: list[str]
    warnings: list[str]


class _GraphAutoEvalResultBase(BaseModel):
    """Shared fields for NLI-based and Gemini-based graph auto-eval results."""
    node_precision: float
    node_recall: float
    node_f1: float
    element_node_precision: float
    element_node_recall: float
    element_node_f1: float
    background_node_precision: float
    background_node_recall: float
    background_node_f1: float
    claim_node_precision: float
    claim_node_recall: float
    claim_node_f1: float
    premise_node_precision: float = 0.0
    premise_node_recall: float = 0.0
    premise_node_f1: float = 0.0
    edge_precision: float
    edge_recall: float
    edge_f1: float
    graph_edit_distance: float | None = None
    node_threshold: float = 0.7
    n_human_nodes: int = 0
    n_model_nodes: int = 0
    n_matched_model_nodes: int = 0
    n_matched_human_nodes: int = 0
    n_human_element_nodes: int = 0
    n_model_element_nodes: int = 0
    n_matched_model_element_nodes: int = 0
    n_matched_human_element_nodes: int = 0
    n_human_background_nodes: int = 0
    n_model_background_nodes: int = 0
    n_matched_model_background_nodes: int = 0
    n_matched_human_background_nodes: int = 0
    n_human_claim_nodes: int = 0
    n_model_claim_nodes: int = 0
    n_matched_model_claim_nodes: int = 0
    n_matched_human_claim_nodes: int = 0
    n_human_premise_nodes: int = 0
    n_model_premise_nodes: int = 0
    n_matched_model_premise_nodes: int = 0
    n_matched_human_premise_nodes: int = 0
    n_human_edges: int = 0
    n_model_edges: int = 0
    n_matched_edges: int = 0
    matched_pairs: list[tuple[int, int]] = []  # list of (human_node_idx, model_node_idx)
    similarity_matrix: list[list[float]] = []  # optional similarity matrix used for matching


class GraphAutoEvalResult(_GraphAutoEvalResultBase):
    """Result of graph-based automatic evaluation (greedy NLI-match F1 + GED)."""


class GeminiGraphAutoEvalResult(_GraphAutoEvalResultBase):
    """Result of graph-based automatic evaluation using Gemini-based node matching."""


# --- Rubric-based reference-free evaluation types ---

class Mistake(BaseModel):
    node_idx: int | None = None
    issue: str
    detail: str


class StageResult(BaseModel):
    score: int          # 1-5
    mistakes: list[Mistake] = []
    reason: str


class Phase0Result(BaseModel):
    passed: bool
    checks: dict[str, bool]
    errors: list[str]
    auto_edge_score: int | None = None


class RubricEvalResult(BaseModel):
    phase0: Phase0Result

    stage1_conclusion: StageResult | None = None
    stage2_elements: StageResult | None = None
    stage3_claims: StageResult | None = None
    stage4_edges: StageResult | None = None
    stage5_background: StageResult | None = None

    h_avg: float | None = None
    h_informed: StageResult | None = None
    h_blind: StageResult | None = None

    stopped_at: int | None = None
    structural_fail: bool = False


class EdgeEvalResult(BaseModel):
    edge_precision: float
    edge_recall: float
    edge_f1: float
    edge_type_precision: float
    edge_type_recall: float
    edge_type_f1: float
    n_human_edges: int
    n_model_edges: int
    n_correct_edges: int
    n_correct_typed_edges: int


class TypedEdgeEvalResult(BaseModel):
    """Edge P/R/F1 split by *derived* edge type, computed like the node-type F1s
    (per type, then macro-averaged in aggregation):

    - **logos**: the source (premise) node is a Claim or Background
      (support and attack collapsed into one bucket)
    - **trigger**: the source node is an Element

    Matching semantics mirror ``compute_edge_f1``: model edges are projected
    through the node alignment and deduplicated, then compared per bucket.
    """
    logos_edge_precision: float
    logos_edge_recall: float
    logos_edge_f1: float
    trigger_edge_precision: float
    trigger_edge_recall: float
    trigger_edge_f1: float
    # macro over the two buckets — a single summary edge number
    macro_edge_precision: float
    macro_edge_recall: float
    macro_edge_f1: float
    n_human_logos_edges: int = 0
    n_model_logos_edges: int = 0
    n_matched_logos_edges: int = 0
    n_human_trigger_edges: int = 0
    n_model_trigger_edges: int = 0
    n_matched_trigger_edges: int = 0


class MatchingGEDResult(BaseModel):
    """Matching-aware graph edit distance computed from a given node alignment."""
    ged: int
    n_node_del: int   # model nodes with no 1-to-1 human counterpart
    n_node_ins: int   # human nodes absent from the matching
    n_edge_del: int   # model edges to remove (deleted-node edges + unmatched projected edges)
    n_edge_ins: int   # human edges to insert
