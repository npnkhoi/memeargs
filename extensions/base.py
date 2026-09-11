"""Base class for meme2graph's deterministic (non-model) processors.

agentflow calls a processor with a flat `inputs` dict; every processor here wants
the domain's `Annotations` view of it instead. `AnnotationProcessor` does that
conversion once so each processor implements `process()` and works with typed
graphs, audits, and evaluations.

Usage:
    from extensions.base import AnnotationProcessor

    class MyProcessor(AnnotationProcessor):
        def process(self, annotations, logger=None):
            return SomeOutput(...)
"""

from abc import abstractmethod
from pathlib import Path

from pydantic import BaseModel

from agentflow.processors.base import Processor

from extensions.typing import Annotations, as_annotations


class AnnotationProcessor(Processor):
    """A processor that reads `Annotations` rather than a raw dict.

    Returning `None` from `process()` marks the item as failed for this stage,
    exactly as it does in agentflow.
    """

    def __call__(
        self, inputs: dict, logger=None, output_dir: Path | None = None
    ) -> BaseModel | None:
        return self.process(as_annotations(inputs), logger)

    @abstractmethod
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        """Compute this stage's output. `logger` is the item's open run.log."""
