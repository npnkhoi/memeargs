"""meme2graph's processors: everything the pipeline does that agentflow cannot.

`LLMProcessor` (see extensions/llm.py) covers every stage that asks a model for a
structured output. The processors here are the deterministic ones: converting a
generated graph, combining edge-construction output with gold nodes, and the
three evaluation stages (node/edge F1, typed edge F1, matching-aware GED).

They are registered in `extensions.register()`, not here; this module only
exposes the classes.
"""

from .edge_eval import EdgeEvalProcessor
from .gemini_graph_auto_eval import GeminiGraphAutoEvalProcessor
from .graph_converter import EdgeConstructionToModelGraphProcessor, GraphConverter
from .human_graph import HumanGraphProcessor
from .matching_ged import MatchingGEDProcessor
from .typed_edge_eval import TypedEdgeEvalProcessor

__all__ = [
    "EdgeConstructionToModelGraphProcessor",
    "EdgeEvalProcessor",
    "GeminiGraphAutoEvalProcessor",
    "GraphConverter",
    "HumanGraphProcessor",
    "MatchingGEDProcessor",
    "TypedEdgeEvalProcessor",
]
