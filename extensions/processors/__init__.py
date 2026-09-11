"""meme2graph's processors: everything the pipeline does that agentflow cannot.

`LLMProcessor` (see extensions/llm.py) covers every stage that asks a model for a
structured output. The processors here are the deterministic ones — graph
cleaning, applying an audit an LLM produced, computing evaluation metrics — plus
`RubricEvalProcessor`, which drives six prompts of its own.

They are registered in `extensions.register()`, not here; this module only
exposes the classes.
"""

from .argument_audit_apply import ArgumentAuditApplyProcessor
from .background_audit_apply import BackgroundAuditApplyProcessor
from .claim_audit_apply import ClaimAuditApplyProcessor
from .claim_merger import ClaimMergeProcessor
from .clean_graph import GraphCleaningProcessor
from .conclusion_audit_apply import ConclusionAuditApplyProcessor
from .edge_eval import EdgeEvalProcessor
from .element_audit_apply import ElementAuditApplyProcessor
from .eval_collector import EvaluationCollector
from .gemini_graph_auto_eval import GeminiGraphAutoEvalProcessor
from .graph_auto_eval import GraphAutoEvalProcessor
from .graph_converter import EdgeConstructionToModelGraphProcessor, GraphConverter
from .human_graph import HumanGraphProcessor
from .matching_ged import MatchingGEDProcessor
from .orphan_remove import OrphanRemoveProcessor
from .paraphrase import ParaphraseProcessor
from .premise_audit_apply import PremiseAuditApplyProcessor
from .rubric_eval import RubricEvalProcessor
from .structure import StructureProcessor
from .typed_edge_eval import TypedEdgeEvalProcessor
from .ungrounded_claims_remove import UngroundedClaimsRemoveProcessor

__all__ = [
    "ArgumentAuditApplyProcessor",
    "BackgroundAuditApplyProcessor",
    "ClaimAuditApplyProcessor",
    "ClaimMergeProcessor",
    "ConclusionAuditApplyProcessor",
    "EdgeConstructionToModelGraphProcessor",
    "EdgeEvalProcessor",
    "ElementAuditApplyProcessor",
    "EvaluationCollector",
    "GeminiGraphAutoEvalProcessor",
    "GraphAutoEvalProcessor",
    "GraphCleaningProcessor",
    "GraphConverter",
    "HumanGraphProcessor",
    "MatchingGEDProcessor",
    "OrphanRemoveProcessor",
    "ParaphraseProcessor",
    "PremiseAuditApplyProcessor",
    "RubricEvalProcessor",
    "StructureProcessor",
    "TypedEdgeEvalProcessor",
    "UngroundedClaimsRemoveProcessor",
]
