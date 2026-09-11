## Constants

from enum import Enum


class VertexType(str, Enum):
    ELEMENT = "element"
    CLAIM = "claim"
    PREMISE = "premise"  # derived: a non-conclusion node with only-element children (or a childless claim)
    BACKGROUND = "background"
    NONE = "none"


class EdgeType(str, Enum):
    SUPPORT = "support"
    ATTACK = "attack"
    TRIGGER = "trigger"
    PARAPHRASE = "paraphrase"


class DerivationEdgeType(str, Enum):
    SUPPORT = "support"
    ATTACK = "attack"


class AnnotationType(str, Enum):
    # NOTE: The values must be actual types, and unique(?)
    GRAPH = "Graph"
    IMAGE = "Image"
    INTERPRETATION = "Interpretation"
    STRUCTURE_ISSUES = "StructuralIssues"
    GRAPH_BY_MODEL = "GraphByModel"
    GRAPH_BY_HUMAN = "GraphByHuman"
    CONTEXT = "Context"
    PREMISES = "Premises"
    CONCLUSION = "Conclusion"
    DERIVATION_OUTPUT = "DerivationOutput"
    ARGUMENT_GRAPH_CONVERSION_RESULT = "ArgumentGraphConversionResult"
    TRIGGER_EDGES = "TriggerEdges"

    CONCLUSION_EVALUATION = "ConclusionEvaluation"
    ELEMENT_EVALUATION = "ElementEvaluation"
    BACKGROUND_EVALUATION = "BackgroundEvaluation"
    CLAIM_EVALUATION = "ClaimEvaluation"
    RELATION_EVALUATION = "RelationEvaluation"
    OVERALL_EVALUATION = "OverallEvaluation"
    EVALUATION = "Evaluation"

    CONCLUSION_EVALUATION_BY_GEMINI = "ConclusionEvaluationByGemini"
    ELEMENT_EVALUATION_BY_GEMINI = "ElementEvaluationByGemini"
    BACKGROUND_EVALUATION_BY_GEMINI = "BackgroundEvaluationByGemini"
    CLAIM_EVALUATION_BY_GEMINI = "ClaimEvaluationByGemini"
    RELATION_EVALUATION_BY_GEMINI = "RelationEvaluationByGemini"
    OVERALL_EVALUATION_BY_GEMINI = "OverallEvaluationByGemini"
    EVALUATION_BY_GEMINI = "EvaluationByGemini"

    GRAPH_AUTO_EVAL_RESULT = "GraphAutoEvalResult"
    MATCHING_RESULT = "MatchingResult"

    TOP_DOWN_GRAPH = "TopDownGraph"
    BOTTOM_UP_GRAPH = "BottomUpGraph"
    EDGE_CONSTRUCTION_OUTPUT = "EdgeConstructionOutput"
    EDGE_EVAL_RESULT = "EdgeEvalResult"
    CONCLUSION_AUDIT = "ConclusionAudit"
    ELEMENT_AUDIT = "ElementAudit"
    CONCLUSION_AUDITED_GRAPH = "ConclusionAuditedGraph"
    ELEMENT_AUDITED_GRAPH = "ElementAuditedGraph"
    CLAIM_EQUIVALENCE_PAIRS = "ClaimEquivalencePairs"
    CLAIM_MERGED_GRAPH = "ClaimMergedGraph"
    CLAIM_AUDIT = "ClaimAudit"
    CLAIM_AUDITED_GRAPH = "ClaimAuditedGraph"
    ARGUMENT_AUDIT = "ArgumentAudit"
    ARGUMENT_AUDITED_GRAPH = "ArgumentAuditedGraph"
    UNGROUNDED_CLAIMS_REMOVED_GRAPH = "UngroundedClaimsRemovedGraph"
    PREMISE_AUDIT = "PremiseAudit"
    PREMISE_AUDITED_GRAPH = "PremiseAuditedGraph"
    BACKGROUND_AUDIT = "BackgroundAudit"
    BACKGROUND_AUDITED_GRAPH = "BackgroundAuditedGraph"
    RUBRIC_EVAL_RESULT = "RubricEvalResult"

class AnnotationSource(str, Enum):
    HUMAN = "human"
    MODEL = "model"


class DemoSelect(str, Enum):
    RANDOM = "random"
    SIMILAR = "similar"
    DIVERSE = "diverse"
    LONGEST = "longest"


class DemoPoolId(str, Enum):
    SEMEVAL = "semeval"
    VISARGS = "visargs"
    MEMEARGS = "memeargs"
    MEMEARGS_V5A = "memeargs_v5a"
    MEMEQA_TRAIN = "memeqa_train"
    MEMEARGS_PROCESSED_0307_TRAIN = "memeargs_processed_0307_train"

class InputFormat(str, Enum):
    CONCLUSION = "conclusion"
    ELEMENT = "element"
    BACKGROUND = "background"
    CLAIM = "claim"
    RELATION = "relation"
    NATURALIZED_ESSAY = "naturalized_essay"
    NODE_MATCHING = "node_matching"
    NODES_ALPHABETICAL = "nodes_alphabetical"
    CONCLUSION_AUDIT = "conclusion_audit"
    ELEMENT_AUDIT = "element_audit"
    CLAIM_AUDIT = "claim_audit"
    ARGUMENT_AUDIT = "argument_audit"
    PREMISE_AUDIT = "premise_audit"
    BACKGROUND_AUDIT = "background_audit"
    BOTTOM_UP_GRAPH = "bottom_up_graph"


EVAL_ASPECTS = ["node_f1", "element_node_f1", "background_node_f1", "claim_node_f1", "edge_f1", "graph_edit_distance"]
