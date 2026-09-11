"""HumanGraphProcessor: expose a human-annotated graph as a typed `Graph`.

The `GraphByHuman` stage feeds the gold graph into the evaluation stages. Using
agentflow's built-in `CopyProcessor` returns the loader's raw dict, which the
output cache then fails to serialise (`dict` has no `model_dump_json`). Reading
the same input through the `Annotations` view instead yields it already
validated into a `Graph`.
"""

from pydantic import BaseModel

from extensions.base import AnnotationProcessor
from extensions.typing import Annotations


class HumanGraphProcessor(AnnotationProcessor):
    """Return the stage's single graph input as a typed `Graph`."""

    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        input_name = self._input_names_snake[0]
        return getattr(annotations, input_name)
