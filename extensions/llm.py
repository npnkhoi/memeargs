"""meme2graph's `LLMProcessor`, a thin subclass of agentflow's.

Two domain behaviours the generic processor cannot know about:

1. Demo pools are annotated with a single human `graph`. Stages that output a
   `TopDownGraph` or `BottomUpGraph` derive their demo target from that graph
   rather than expecting a field of their own.
2. When a stage sets no `input_format`, inputs are rendered from the typed
   `Annotations` view, so a graph appears as indented JSON — the form every
   prompt here was written against.

Registered under the plain name `LLMProcessor`, so configs are unaffected.

Usage:
    from extensions.llm import GraphLLMProcessor
    Pipeline.register_processor("LLMProcessor", GraphLLMProcessor)
"""

from agentflow.pipeline import Pipeline
from agentflow.processors.llm import LLMProcessor
from agentflow.typing.config import StageConfig

from extensions.input_formats import AnnotationFormatter
from extensions.typing import as_annotations


class _AnnotationInputFormater:
    """Formats stage inputs through `AnnotationFormatter`.

    Stands in for agentflow's `InputFormater` on this processor: same `format()`
    interface, but the no-format case goes through the domain renderer instead of
    agentflow's raw-dict one.
    """

    def __init__(self, input_format: str | None, input_names_snake: list[str]):
        self._formatter = AnnotationFormatter(input_format, input_names_snake)

    def format(self, inputs: dict) -> str:
        return self._formatter.format(as_annotations(inputs))


class GraphLLMProcessor(LLMProcessor):
    def __init__(self, pipeline: Pipeline, stage_config: StageConfig):
        super().__init__(pipeline, stage_config)
        self.input_formater = _AnnotationInputFormater(
            stage_config.input_format, self._input_names_snake
        )

    def _demo_output(self, demo: dict, output_name: str):
        """The demo's target output, derived from its `graph` when absent.

        Training items carry a human `graph` and nothing else, so a stage that
        outputs `TopDownGraph` or `BottomUpGraph` would otherwise find no demo
        target and silently drop every demo.
        """
        value = demo.get(output_name)
        if value is not None:
            return value
        return getattr(as_annotations(demo), f"{output_name}_or_convert", None)
