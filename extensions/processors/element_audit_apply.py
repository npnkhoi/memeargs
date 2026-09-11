from pydantic import BaseModel

from extensions.audit_utils import apply_element_audit, audit_has_edge_cases
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.graph import ElementAudit, Graph


class ElementAuditApplyProcessor(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None) -> BaseModel | None:
        graph: Graph = annotations.conclusion_audited_graph
        audit: ElementAudit = annotations.element_audit

        if graph is None or audit is None:
            return None

        if audit_has_edge_cases(graph, audit, log_file=logger):
            return None

        return apply_element_audit(graph, audit)
