"""Input formatting for meme2graph stages, registered with agentflow.

`LLMProcessor` turns a stage's inputs into the user message. The default is a
labelled JSON dump per input, but most stages here need a domain-specific view:
an audit stage wants the graph laid out as indexed nodes and arguments, an
evaluation stage wants human and model claims side by side. Those views are the
`input_format:` values in the configs, registered by `register()` below.

`AnnotationFormatter` holds the logic (unchanged from the pre-agentflow
`memefal.input_formater`); the registered handlers just adapt agentflow's
`(inputs: dict, input_names_snake)` calling convention to it.

Usage:
    from extensions.input_formats import register
    register()   # before constructing Pipeline
"""

from typing import Any
import warnings

import networkx
from pydantic import BaseModel

from agentflow.input_formater import InputFormater

import extensions.typing
from extensions.const import InputFormat
from extensions.typing import Annotations, as_annotations
from extensions.typing.graph import BottomUpGraph, Graph, Vertex


class AnnotationFormatter:
    def __init__(self, input_format: InputFormat | None, input_names_snake: list[str]):
        self.input_format = input_format
        self._input_names_snake = input_names_snake

    def format(self, inputs: Annotations) -> str:
        """
        Format the inputs into a user prompt string based on the specified input format.
        """
        user_prompt = ""
        def bullet_list_or_none(items: list[Any]) -> str:
            if not items:
                return "None"
            return "\n".join([f"- {item}" for item in items])
        if not self.input_format:
            for input_name in self._input_names_snake:
                if input_name == "image":
                    continue
                if user_prompt != "":
                    user_prompt += "\n\n"
                value = getattr(inputs, input_name)

                # represent input value
                if isinstance(value, BaseModel):
                    value = value.model_dump_json(indent=2)
                user_prompt += f"{input_name}:\n{value}"
        elif self.input_format == InputFormat.CONCLUSION:
            def get_conclusions(graph_field_name: str) -> list[str]:
                """
                Get a list of conclusion texts from the graph.
                """
                graph: Graph = getattr(inputs, graph_field_name)
                assert graph is not None, f"{graph_field_name} is None"
                conclusion_indices = graph.get_conclusions()
                conclusions = []
                for idx in conclusion_indices:
                    node = graph.nodes[idx]
                    conclusions.append(node.text)
                return conclusions
            human_conclusions = get_conclusions("graph_by_human")
            if len(human_conclusions) > 1:
                warnings.warn("More than one human conclusion found. Using the first one.")
            human_conclusion = human_conclusions[0]
            model_conclusions = get_conclusions("graph_by_model")
            user_prompt = "\n".join([
                f"Human Conclusion: {human_conclusion}",
                f"Model Conclusion(s):",
                bullet_list_or_none(model_conclusions)
            ])
        elif self.input_format == InputFormat.ELEMENT:
            def get_elements(graph_field_name: str) -> list[str]:
                """
                Get a list of element texts from the graph.
                """
                graph: Graph = getattr(inputs, graph_field_name)
                assert graph is not None, f"{graph_field_name} is None"
                elements = graph.get_nodes_by_type(extensions.typing.VertexType.ELEMENT)
                return [str(e) for e in elements]
            human_elements = get_elements("graph_by_human")
            model_elements = get_elements("graph_by_model")
            user_prompt = "\n".join([
                f"Human Elements:",
                bullet_list_or_none(human_elements),
                f"Model Elements:",
                bullet_list_or_none(model_elements),
            ])
        elif self.input_format == InputFormat.BACKGROUND:
            def get_backgrounds(graph_field_name: str) -> list[str]:
                """
                Get a list of background texts from the graph.
                """
                graph: Graph = getattr(inputs, graph_field_name)
                assert graph is not None, f"{graph_field_name} is None"
                backgrounds = graph.get_nodes_by_type(extensions.typing.VertexType.BACKGROUND)
                return [str(b) for b in backgrounds]
            human_backgrounds = get_backgrounds("graph_by_human")
            model_backgrounds = get_backgrounds("graph_by_model")
            user_prompt = "\n".join([
                f"Human Backgrounds:",
                bullet_list_or_none(human_backgrounds),
                f"Model Backgrounds:",
                bullet_list_or_none(model_backgrounds),
            ])
        elif self.input_format == InputFormat.CLAIM:
            def get_claims(graph_field_name: str) -> list[str]:
                """
                Get a list of claim texts from the graph.
                """
                graph: Graph = getattr(inputs, graph_field_name)
                assert graph is not None, f"{graph_field_name} is None"
                try:
                    nodes = graph.topological_order()
                except networkx.exception.NetworkXUnfeasible:
                    warnings.warn("Graph has cycles, cannot perform topological sort. Using original node order.")
                    nodes = graph.nodes
                claims = [node for node in nodes if node.type == extensions.typing.VertexType.CLAIM]
                return [str(c) for c in claims]
            human_claims = get_claims("graph_by_human")
            model_claims = get_claims("graph_by_model")
            user_prompt = "\n".join([
                f"Human Claims:",
                bullet_list_or_none(human_claims),
                f"Model Claims:",
                bullet_list_or_none(model_claims),
            ])
        elif self.input_format == InputFormat.RELATION:
            def get_relations_and_connected_components(graph_field_name: str) -> tuple[list[str], str]:
                """
                Get a list of relation texts from the graph.
                """
                graph: Graph = getattr(inputs, graph_field_name)
                assert graph is not None, f"{graph_field_name} is None"
                # relations = graph.edges  # old: direct edge access
                relations = graph.flat_edges()  # new: expand arguments to edges
                ret = []
                for edge in relations:
                    from_node = graph.nodes[edge.from_idx] if edge.from_idx < len(graph.nodes) else None
                    to_node = graph.nodes[edge.to_idx] if edge.to_idx < len(graph.nodes) else None
                    ret.append(f"[{str(from_node)}] --({edge.type.name})--> [{str(to_node)}]")
                cc = graph.connected_components()
                if len(cc) == 1:
                    ret_cc = "The graph is weakly connected."
                else:
                    ret_cc = f"The graph has {len(cc)} weakly connected components:"
                    for i, component in enumerate(cc):
                        ret_cc += f"({', '.join(node.label() for node in component)})"
                return ret, ret_cc
            human_relations, human_ret_cc = get_relations_and_connected_components("graph_by_human")
            model_relations, model_ret_cc = get_relations_and_connected_components("graph_by_model")

            user_prompt = "\n".join([
                f"Human Relations:",
                bullet_list_or_none(human_relations),
                f"Model Relations:",
                bullet_list_or_none(model_relations),
                f"Model Graph Connectivity:",
                model_ret_cc,
            ])
        elif self.input_format == InputFormat.NATURALIZED_ESSAY:
            essay = inputs.interpretation.naturalize()
            user_prompt = f"Interpretation Essay:\n{essay}"
        elif self.input_format == InputFormat.NODE_MATCHING:
            import networkx as nx
            from extensions.const import VertexType as VT
            g_human = inputs.graph_by_human
            g_model = inputs.graph_by_model
            assert g_human is not None and g_model is not None
            g_human.normalize()
            g_model.normalize()
            g_human.derive_premise_types()
            g_model.derive_premise_types()

            def _fmt(graph, vtype, label):
                nodes = graph.get_nodes_by_type(vtype)
                if not nodes:
                    return f"{label}: (none)"
                return label + ":\n" + "\n".join(f"  [{i}] {v.text}" for i, v in enumerate(nodes))

            # Forbidden pairs: model node pairs (within a type) with a directed
            # path between them — they cannot be grouped under the same human node.
            G_model = g_model._nxgraph()

            def _forbidden_pairs(vtype) -> str:
                nodes = g_model.get_nodes_by_type(vtype)
                pairs = []
                for i, vi in enumerate(nodes):
                    for j, vj in enumerate(nodes):
                        if i >= j:
                            continue
                        if nx.has_path(G_model, vi.idx, vj.idx) or nx.has_path(G_model, vj.idx, vi.idx):
                            pairs.append(f"({i}, {j})")
                return ", ".join(pairs) if pairs else "(none)"

            sections = [
                "=== HUMAN GRAPH ===",
                _fmt(g_human, VT.ELEMENT, "ELEMENT"),
                _fmt(g_human, VT.PREMISE, "PREMISE"),
                _fmt(g_human, VT.BACKGROUND, "BACKGROUND"),
                _fmt(g_human, VT.CLAIM, "CLAIM"),
                "",
                "=== MODEL GRAPH ===",
                _fmt(g_model, VT.ELEMENT, "ELEMENT"),
                _fmt(g_model, VT.PREMISE, "PREMISE"),
                _fmt(g_model, VT.BACKGROUND, "BACKGROUND"),
                _fmt(g_model, VT.CLAIM, "CLAIM"),
                "",
                "=== FORBIDDEN PAIRS ===",
                "Pairs of model nodes (by sublist index within each type) that have a directed",
                "path between them in the model graph. These CANNOT be grouped together under",
                "the same human node.",
                f"ELEMENT: {_forbidden_pairs(VT.ELEMENT)}",
                f"PREMISE: {_forbidden_pairs(VT.PREMISE)}",
                f"BACKGROUND: {_forbidden_pairs(VT.BACKGROUND)}",
                f"CLAIM: {_forbidden_pairs(VT.CLAIM)}",
            ]
            user_prompt = "\n".join(sections)
        elif self.input_format == InputFormat.CONCLUSION_AUDIT:
            graph: Graph = inputs.graph
            assert graph is not None, "graph is None"
            conclusion_idxs = graph.get_conclusions()
            assert len(conclusion_idxs) == 1
            conclusion_idx = conclusion_idxs[0]
            conclusion_node = next(v for v in graph.nodes if v.idx == conclusion_idx)
            other_claims = [
                v for v in graph.nodes
                if v.type == extensions.typing.VertexType.CLAIM and v.idx != conclusion_idx
            ]
            lines = [
                f"Current conclusion (idx={conclusion_idx}): \"{conclusion_node.text}\"",
                "",
                "Other claim nodes:",
            ]
            if other_claims:
                for v in other_claims:
                    lines.append(f"  idx={v.idx}: \"{v.text}\"")
            else:
                lines.append("  (none)")
            lines += ["", "Full graph:", graph.model_dump_json(indent=2)]
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.ELEMENT_AUDIT:
            from extensions.typing.graph import graph_to_bottom_up
            source_graph = inputs.conclusion_audited_graph
            assert source_graph is not None, "conclusion_audited_graph is None"
            bu_graph: BottomUpGraph = graph_to_bottom_up(source_graph)
            all_nodes = (
                [bu_graph.conclusion]
                + list(bu_graph.leaves)
                + [Vertex(idx=n.idx, text=n.text, type=n.type) for n in bu_graph.intermediates]
            )
            element_nodes = [v for v in all_nodes if v.type == extensions.typing.VertexType.ELEMENT]
            lines = [
                "=== CONCLUSION ===",
                f"  idx={bu_graph.conclusion.idx}: \"{bu_graph.conclusion.text}\"",
                "",
                "=== BOTTOM-UP ARGUMENT GRAPH ===",
                bu_graph.model_dump_json(indent=2),
                "",
                "=== ELEMENT NODES TO AUDIT ===",
            ]
            for v in element_nodes:
                lines.append(f"  idx={v.idx}: \"{v.text}\"")
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.CLAIM_AUDIT:
            from extensions.typing.graph import Graph as GraphType
            graph: GraphType = inputs.claim_merged_graph
            assert graph is not None, "claim_merged_graph is None"
            zero_out_degree = set(graph.get_conclusions())
            claim_conclusions = [
                v for v in graph.nodes
                if v.idx in zero_out_degree and v.type == extensions.typing.VertexType.CLAIM
            ]
            assert len(claim_conclusions) == 1, f"Expected 1 claim conclusion, got {len(claim_conclusions)}"
            conclusion_node = claim_conclusions[0]
            conclusion_idx = conclusion_node.idx
            intermediate_claims = [
                v for v in graph.nodes
                if v.type == extensions.typing.VertexType.CLAIM and v.idx != conclusion_idx
            ]
            lines = [
                "=== CONCLUSION ===",
                f"  idx={conclusion_node.idx}: \"{conclusion_node.text}\"",
                "",
                "=== FULL GRAPH ===",
                graph.model_dump_json(indent=2),
                "",
                "=== INTERMEDIATE CLAIM NODES TO AUDIT ===",
            ]
            for v in intermediate_claims:
                lines.append(f"  idx={v.idx}: \"{v.text}\"")
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.ARGUMENT_AUDIT:
            graph: Graph = inputs.element_audited_graph
            assert graph is not None, "element_audited_graph is None"
            lines = ["=== NODES ==="]
            for node in graph.nodes:
                lines.append(f"  idx={node.idx} type={node.type.value}: \"{node.text}\"")
            lines += ["", "=== ARGUMENTS (indexed) ==="]
            for i, arg in enumerate(graph.arguments):
                premise_strs = ", ".join(str(p) for p in arg.premises)
                lines.append(f"  arg_idx={i} type={arg.type.value}: premises=[{premise_strs}] -> claim={arg.claim}")
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.PREMISE_AUDIT:
            graph: Graph = inputs.ungrounded_claims_removed_graph
            assert graph is not None, "ungrounded_claims_removed_graph is None"
            lines = ["=== NODES ==="]
            for node in graph.nodes:
                lines.append(f"  idx={node.idx} type={node.type.value}: \"{node.text}\"")
            lines += ["", "=== ARGUMENTS (indexed) ==="]
            for i, arg in enumerate(graph.arguments):
                premise_strs = ", ".join(str(p) for p in arg.premises)
                lines.append(f"  arg_idx={i} type={arg.type.value}: premises=[{premise_strs}] -> claim={arg.claim}")
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.BACKGROUND_AUDIT:
            graph: Graph = inputs.premise_audited_graph
            assert graph is not None, "premise_audited_graph is None"
            lines = ["=== NODES ==="]
            for node in graph.nodes:
                lines.append(f"  idx={node.idx} type={node.type.value}: \"{node.text}\"")
            lines += ["", "=== ARGUMENTS (indexed) ==="]
            for i, arg in enumerate(graph.arguments):
                premise_strs = ", ".join(str(p) for p in arg.premises)
                lines.append(f"  arg_idx={i} type={arg.type.value}: premises=[{premise_strs}] -> claim={arg.claim}")
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.NODES_ALPHABETICAL:
            graph: Graph = inputs.graph_by_human
            assert graph is not None, "graph_by_human is None"
            graph.normalize()
            conclusions = graph.get_conclusions()
            root_idx = conclusions[0]
            root = graph.nodes[root_idx]
            others = sorted(
                [n for n in graph.nodes if n.idx != root_idx],
                key=lambda n: n.text,
            )
            lines = [
                "ROOT NODE:",
                f"[{root.idx}] ({root.type.value}) {root.text}",
                "",
                "OTHER NODES (alphabetical by text):",
            ]
            for n in others:
                lines.append(f"[{n.idx}] ({n.type.value}) {n.text}")
            user_prompt = "\n".join(lines)
        elif self.input_format == InputFormat.BOTTOM_UP_GRAPH:
            from extensions.typing.graph import graph_to_bottom_up
            graph_field = next(n for n in self._input_names_snake if n != "image")
            graph = getattr(inputs, graph_field)
            assert graph is not None, f"{graph_field} is None"
            bu_graph = graph_to_bottom_up(graph)
            user_prompt = bu_graph.model_dump_json(indent=2)
        else:
            raise NotImplementedError(f"Input format {self.input_format} not implemented.")

        return user_prompt


def _handler_for(input_format: InputFormat):
    """Adapt one format to agentflow's handler signature."""

    def handler(inputs: dict, input_names_snake: list[str]) -> str:
        return AnnotationFormatter(input_format, input_names_snake).format(
            as_annotations(inputs)
        )

    return handler


def default_format(inputs: dict, input_names_snake: list[str]) -> str:
    """The no-`input_format` rendering: one labelled block per non-image input.

    Kept out of agentflow's own default because that one sees raw dicts, which
    would print a Python repr where these prompts have always shown JSON.
    """
    return AnnotationFormatter(None, input_names_snake).format(as_annotations(inputs))


def register() -> None:
    """Register every domain input format. Call before constructing Pipeline."""
    for input_format in InputFormat:
        InputFormater.register(input_format.value, _handler_for(input_format))
