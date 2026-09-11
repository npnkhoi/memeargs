from pathlib import Path
from pydantic import BaseModel
from copy import deepcopy
import re
from extensions.const import EdgeType, VertexType
from extensions.typing.eval import (
    BackgroundEvaluationByGemini,
    ClaimEvaluationByGemini,
    ConclusionEvaluationByGemini,
    ElementEvaluationByGemini,
    OverallEvaluationByGemini,
    RelationEvaluationByGemini,
    StructuralIssues,
    EvaluationByGemini,
    GraphAutoEvalResult,
    GeminiGraphAutoEvalResult,
    NodeMatchPair,
    MatchingResult,
)
from extensions.typing.graph import Vertex
from extensions.typing.graph import Edge
from extensions.typing.graph import Graph
from extensions.typing.graph import GraphByModel
from extensions.typing.graph import GraphByHuman
from extensions.typing.graph import DerivationOutput
from extensions.typing.graph import ArgumentGraphConversionResult
from extensions.typing.graph import TopDownGraph, BottomUpGraph, EdgeConstructionOutput
from extensions.typing.graph import ConclusionAudit, ElementAudit, ClaimEquivalencePairs, ClaimMergedGraph, ClaimAudit, ClaimAuditedGraph, ConclusionAuditedGraph, ElementAuditedGraph, ArgumentAudit, ArgumentAuditedGraph, UngroundedClaimsRemovedGraph, PremiseAudit, PremiseAuditedGraph, BackgroundAudit, BackgroundAuditedGraph, OrphanRemovedGraph
from extensions.typing.graph import ConclusionAudit, ElementAudit
from extensions.typing.eval import RubricEvalResult, EdgeEvalResult, MatchingGEDResult, TypedEdgeEvalResult
from extensions.typing.eval import (
    ConclusionEvaluation,
    RelationEvaluation,
    ElementEvaluation,
    BackgroundEvaluation,
    ClaimEvaluation,
    OverallEvaluation,
    Evaluation,
)

# for typing model outputs, must create new classes
# such as InterepretationOutput, that doesn't have image field


class Interpretation(BaseModel):
    essay: str
    conversation_history: list | None = None

    def naturalize(self) -> str:
        """
        Turn instances of "[4V [1] [2] abc]" into just "abc".
        the [1] [2] can appear 0 or more times.
        """
        s = self.essay
        s = re.sub(r"\[(\d+)\]", "", s, count=100)
        s = re.sub(r"\[(\d+)([EVBTD])(.*?)\]", r"\3", s, count=100)
        s = s.replace("  ", " ")
        return s


class SegmentsState(BaseModel):
    image: str  # image filename
    segments: list[str]


class Annotations(BaseModel):
    id: str
    image: Path | str | None = None
    context: str | None = "Unspecified"
    interpretation: Interpretation | None = None
    graph: Graph | None = None
    graph_by_model: GraphByModel | None = None
    graph_by_human: GraphByHuman | None = None
    structural_issues: StructuralIssues | None = None

    interpretation_text: str | None = None
    premises: list[Vertex] | None = None
    conclusion: Vertex | None = None
    derivation_output: DerivationOutput | None = None
    premises_and_conclusion: dict | None = None
    trigger_edges: list[Edge] | None = None

    argument_graph_conversion_result: ArgumentGraphConversionResult | None = None

    conclusion_evaluation: ConclusionEvaluation | None = None
    element_evaluation: ElementEvaluation | None = None
    background_evaluation: BackgroundEvaluation | None = None
    claim_evaluation: ClaimEvaluation | None = None
    relation_evaluation: RelationEvaluation | None = None
    overall_evaluation: OverallEvaluation | None = None
    evaluation: Evaluation | None = None

    conclusion_evaluation_by_gemini: ConclusionEvaluationByGemini | None = None
    element_evaluation_by_gemini: ElementEvaluationByGemini | None = None
    background_evaluation_by_gemini: BackgroundEvaluationByGemini | None = None
    claim_evaluation_by_gemini: ClaimEvaluationByGemini | None = None
    relation_evaluation_by_gemini: RelationEvaluationByGemini | None = None
    overall_evaluation_by_gemini: OverallEvaluationByGemini | None = None
    evaluation_by_gemini: EvaluationByGemini | None = None

    matching_result: MatchingResult | None = None

    graph_auto_eval_result: GraphAutoEvalResult | None = None
    gemini_graph_auto_eval_result: GeminiGraphAutoEvalResult | None = None

    top_down_graph: TopDownGraph | None = None
    bottom_up_graph: BottomUpGraph | None = None
    conclusion_audit: ConclusionAudit | None = None
    element_audit: ElementAudit | None = None
    conclusion_audited_graph: ConclusionAuditedGraph | None = None
    element_audited_graph: ElementAuditedGraph | None = None
    claim_equivalence_pairs: ClaimEquivalencePairs | None = None
    claim_merged_graph: ClaimMergedGraph | None = None
    claim_audit: ClaimAudit | None = None
    claim_audited_graph: ClaimAuditedGraph | None = None
    argument_audit: ArgumentAudit | None = None
    argument_audited_graph: ArgumentAuditedGraph | None = None
    ungrounded_claims_removed_graph: UngroundedClaimsRemovedGraph | None = None
    premise_audit: PremiseAudit | None = None
    premise_audited_graph: PremiseAuditedGraph | None = None
    background_audit: BackgroundAudit | None = None
    background_audited_graph: BackgroundAuditedGraph | None = None
    orphan_removed_graph: OrphanRemovedGraph | None = None
    rubric_eval_result: RubricEvalResult | None = None
    edge_construction_output: EdgeConstructionOutput | None = None
    edge_eval_result: EdgeEvalResult | None = None
    typed_edge_eval_result: TypedEdgeEvalResult | None = None
    matching_ged_result: MatchingGEDResult | None = None

    @property
    def top_down_graph_or_convert(self) -> "TopDownGraph | None":
        """Returns top_down_graph field if set, otherwise converts from graph."""
        if self.top_down_graph is not None:
            return self.top_down_graph
        if self.graph is None:
            return None
        from extensions.processors.graph_converter import graph_to_top_down
        return graph_to_top_down(self.graph)

    @property
    def bottom_up_graph_or_convert(self) -> "BottomUpGraph | None":
        """Returns bottom_up_graph field if set, otherwise converts from graph."""
        if self.bottom_up_graph is not None:
            return self.bottom_up_graph
        if self.graph is None:
            return None
        from extensions.processors.graph_converter import graph_to_bottom_up
        return graph_to_bottom_up(self.graph)

    def create_premises_and_conclusion(self):
        """
        Create premises and conclusion fields from graph field.
        Re-index the nodes and edges in the original graph accordingly.
        """
        assert self.graph is not None
        self.graph.normalize()
        new_full_graph = Graph(nodes=[], edges=[])

        ## Rearrange the graph to have conclusion as the first node, followed by premises, then the rest of the nodes
        ## For edges, trigger edges go first, then the rest of the edges
        # conclusion
        con_ids = self.graph.get_conclusions()
        assert len(con_ids) == 1, f"Expected 1 conclusion, got {len(con_ids)}"
        con = self.graph.nodes[con_ids[0]]
        assert con.type == VertexType.CLAIM or (
            con.type == VertexType.ELEMENT and len(self.graph.nodes) == 1
        )
        new_full_graph.nodes.append(deepcopy(con))

        # premises
        for node in self.graph.nodes:
            if node.type in [VertexType.ELEMENT, VertexType.BACKGROUND]:
                new_full_graph.nodes.append(deepcopy(node))

        # triggering edges
        for edge in self.graph.edges:
            if edge.type == EdgeType.TRIGGER:
                # check edge integrity
                from_node = self.graph.nodes[edge.from_idx]
                to_node = self.graph.nodes[edge.to_idx]
                if to_node.type != VertexType.BACKGROUND:
                    continue
                if from_node.type == VertexType.CLAIM:
                    continue
                new_full_graph.edges.append(deepcopy(edge))

        # add the rest for the new_full_graph
        for node in self.graph.nodes:  # add all D-node that is not the conclusion
            if node.type == VertexType.CLAIM and node.idx != con.idx:
                new_full_graph.nodes.append(deepcopy(node))
        for edge in self.graph.edges:  # add all the I/C edges
            if edge.type in [EdgeType.SUPPORT, EdgeType.ATTACK]:
                new_full_graph.edges.append(deepcopy(edge))

        new_full_graph.normalize()  # reindex based on positions of the nodes in the node list

        ## Collect premises and conclusion
        premises = [
            node
            for node in new_full_graph.nodes
            if node.type in [VertexType.ELEMENT, VertexType.BACKGROUND]
        ]
        conclusion = new_full_graph.nodes[0]
        trigger_edges = [
            edge for edge in new_full_graph.edges if edge.type == EdgeType.TRIGGER
        ]

        self.graph = new_full_graph
        self.premises = premises
        self.conclusion = conclusion
        self.trigger_edges = trigger_edges


class DataItem(BaseModel):
    id: str
    data: Annotations


def as_annotations(inputs: dict) -> Annotations:
    """Validate agentflow's flat inputs dict into an `Annotations` object.

    agentflow hands a processor a plain dict: human-sourced fields hold raw JSON
    (a `graph` is a dict of lists), while model-sourced fields already hold the
    previous stage's Pydantic output. `Annotations` accepts both, so this is the
    single place where the framework's dict view becomes the domain's typed view.

    Unknown keys are rejected rather than silently dropped: a stage input whose
    name has no `Annotations` field would otherwise vanish and surface much later
    as a confusing `None`.
    """
    unknown = set(inputs) - set(Annotations.model_fields)
    if unknown:
        raise KeyError(
            f"No Annotations field for stage input(s): {sorted(unknown)}. "
            f"Add the field to Annotations (extensions/typing/__init__.py) or fix "
            f"the input name in the config."
        )
    return Annotations.model_validate(inputs)
