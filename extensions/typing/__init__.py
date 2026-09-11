from pathlib import Path
from pydantic import BaseModel

from extensions.typing.eval import (
    GeminiGraphAutoEvalResult,
    NodeMatchPair,
    MatchingResult,
    EdgeEvalResult,
    MatchingGEDResult,
    TypedEdgeEvalResult,
)
from extensions.typing.graph import Graph
from extensions.typing.graph import GraphByModel
from extensions.typing.graph import GraphByHuman
from extensions.typing.graph import TopDownGraph, BottomUpGraph, EdgeConstructionOutput


class DataItem(BaseModel):
    id: str
    data: "Annotations"


class Annotations(BaseModel):
    id: str
    image: Path | str | None = None
    graph: Graph | None = None
    graph_by_model: GraphByModel | None = None
    graph_by_human: GraphByHuman | None = None

    matching_result: MatchingResult | None = None
    gemini_graph_auto_eval_result: GeminiGraphAutoEvalResult | None = None

    top_down_graph: TopDownGraph | None = None
    bottom_up_graph: BottomUpGraph | None = None
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
