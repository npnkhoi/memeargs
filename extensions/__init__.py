"""meme2graph's domain extensions to agentflow.

agentflow knows how to run a cached, few-shot, multi-stage pipeline over a
dataset; it knows nothing about argument graphs. This package supplies that half:
the Pydantic types a stage can output, the processors that transform graphs
without calling a model, the prompt-side renderings of a graph, and the demo
selection strategy that depends on annotation length.

Everything is attached through agentflow's registration APIs, so nothing in
`agentflow/` is edited or subclassed beyond the two documented hooks.

Usage:
    import extensions
    extensions.register()          # before constructing Pipeline
"""

from pydantic import BaseModel

from agentflow.demo import DemoPool
from agentflow.pipeline import Pipeline

from extensions import input_formats, processors, typing as domain_typing
from extensions.llm import GraphLLMProcessor

# Stage names that are inputs only — supplied by the loader from the dataset, so
# they are never a stage output and need no registered type.
_HUMAN_ONLY_TYPES = frozenset(
    {"Image", "Context", "Premises", "Conclusion", "TriggerEdges"}
)

_PROCESSORS: dict[str, type] = {
    # Every stage that asks a model for a structured output.
    "LLMProcessor": GraphLLMProcessor,
    # Deterministic graph surgery.
    "GraphCleaningProcessor": processors.GraphCleaningProcessor,
    "GraphConverter": processors.GraphConverter,
    "HumanGraphProcessor": processors.HumanGraphProcessor,
    "EdgeConstructionToModelGraphProcessor": processors.EdgeConstructionToModelGraphProcessor,
    "ClaimMergeProcessor": processors.ClaimMergeProcessor,
    "OrphanRemoveProcessor": processors.OrphanRemoveProcessor,
    "UngroundedClaimsRemoveProcessor": processors.UngroundedClaimsRemoveProcessor,
    "StructureProcessor": processors.StructureProcessor,
    "ParaphraseProcessor": processors.ParaphraseProcessor,
    # Applying an audit that an LLM stage produced.
    "ConclusionAuditApplyProcessor": processors.ConclusionAuditApplyProcessor,
    "ElementAuditApplyProcessor": processors.ElementAuditApplyProcessor,
    "ClaimAuditApplyProcessor": processors.ClaimAuditApplyProcessor,
    "ArgumentAuditApplyProcessor": processors.ArgumentAuditApplyProcessor,
    "PremiseAuditApplyProcessor": processors.PremiseAuditApplyProcessor,
    "BackgroundAuditApplyProcessor": processors.BackgroundAuditApplyProcessor,
    # Evaluation.
    "GraphAutoEvalProcessor": processors.GraphAutoEvalProcessor,
    "GeminiGraphAutoEvalProcessor": processors.GeminiGraphAutoEvalProcessor,
    "TypedEdgeEvalProcessor": processors.TypedEdgeEvalProcessor,
    "MatchingGEDProcessor": processors.MatchingGEDProcessor,
    "EdgeEvalProcessor": processors.EdgeEvalProcessor,
    "EvaluationCollector": processors.EvaluationCollector,
    "RubricEvalProcessor": processors.RubricEvalProcessor,
}


def _longest_demos(pool: DemoPool, inputs: dict) -> list[dict]:
    """Demo strategy `longest`: the items with the longest interpretation essay.

    Ignores the query item, so every item gets the same demos.
    """

    def essay_length(item_id: str) -> int:
        item = pool.loader.load(item_id) or {}
        interpretation = item.get("interpretation") or {}
        if isinstance(interpretation, BaseModel):
            interpretation = interpretation.model_dump()
        return len(interpretation.get("essay") or "")

    ranked = sorted(pool.item_ids, key=essay_length, reverse=True)
    return pool.items_from_ids(ranked[: pool.config.shots])


def registered_types() -> dict[str, type[BaseModel]]:
    """Every Pydantic model `extensions.typing` exports, keyed by class name.

    A stage names its output type in the config, and that name is also the cache
    directory and part of the prompt filename — so the registry has to cover every
    model a config might name. Taking whatever `extensions.typing` exports keeps
    that automatic: a new output type is registered by being importable, with no
    second list to update. (`AnnotationType` is deliberately not the source here —
    it has drifted from the type list, missing e.g. `OrphanRemovedGraph`.)
    """
    types: dict[str, type[BaseModel]] = {}
    for name, cls in vars(domain_typing).items():
        if name.startswith("_") or name in _HUMAN_ONLY_TYPES:
            continue
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            types[name] = cls
    return types


def register() -> None:
    """Attach every domain extension to agentflow. Call before building Pipeline."""
    for name, cls in registered_types().items():
        Pipeline.register_type(name, cls)
    for name, processor_cls in _PROCESSORS.items():
        Pipeline.register_processor(name, processor_cls)
    input_formats.register()
    DemoPool.register_strategy("longest", _longest_demos)
