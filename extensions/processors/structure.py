"""
Detecting structural errors in argument graphs
"""

from extensions.const import EdgeType, VertexType
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
# from typing import Set, Dict, List
import networkx as nx

from extensions.typing.eval import GraphCheckResult, StructuralIssues
from extensions.typing.graph import Graph



class StructureProcessor(AnnotationProcessor):
    """
    Processor to detect structural errors in argument graphs
    """
    
    @staticmethod
    def _check_graph(graph: Graph) -> GraphCheckResult:
        errors = []
        warnings = []

        # Build lookup tables
        idx_to_vertex = {v.idx: v for v in graph.nodes}
        vertex_types = {v.idx: v.type for v in graph.nodes}

        # --- 1. Edge type checks ---
        for edge in graph.edges:
            if edge.type in (EdgeType.SUPPORT, EdgeType.ATTACK):
                # Target must be D-node
                if vertex_types.get(edge.to_idx) != VertexType.CLAIM:
                    errors.append(
                        f"{edge.type.title()} edge must target a D-node (derivation), but edge {edge.from_idx}->{edge.to_idx} targets {vertex_types.get(edge.to_idx)}."
                    )
            if edge.type == EdgeType.TRIGGER:
                # target must be B-node
                if vertex_types.get(edge.to_idx) != VertexType.BACKGROUND:
                    errors.append(
                        f"Trigger edge must target a B-node (background), but edge {edge.from_idx}->{edge.to_idx} targets {vertex_types.get(edge.to_idx)}."
                    )

        # --- 2. Build primary graph (inference + conflict edges only) ---
        # primary_edges = [
        #     (e.from_idx, e.to_idx)
        #     for e in graph.edges
        #     # if e.type in (EdgeType.INFERENCE, EdgeType.CONFLICT)
        # ]
        # G = nx.DiGraph()
        # G.add_nodes_from(idx_to_vertex.keys())
        # G.add_edges_from(primary_edges)

        edges = [
            (e.from_idx, e.to_idx)
            for e in graph.edges
        ]
        FG = nx.DiGraph()
        FG.add_nodes_from(idx_to_vertex.keys())
        FG.add_edges_from(edges)

        # --- 3. Check for DAG and connectivity ---
        if not nx.is_directed_acyclic_graph(FG):
            cycle = nx.find_cycle(FG)
            errors.append(f"The graph must be acyclic. Found cycle: {cycle}")
        # if not nx.is_weakly_connected(FG):
        #     excluded_nodes = [v.idx for v in graph.nodes if FG.out_degree(v.idx) == 0 and G.in_degree(v.idx) == 0]
        #     warnings.append(f"The graph should be connected. Did you mean to leave out the following nodes?: {excluded_nodes}")

        # --- 4. Out-degree 0 node (conclusion) ---
        out0_nodes = [n for n in FG.nodes if FG.out_degree(n) == 0 and FG.in_degree(n) != 0]
        if len(out0_nodes) != 1:
            warnings.append(f"The graph must have exactly one conclusion node (with no arrows pointing from it), found {len(out0_nodes)}: {out0_nodes}.")
        else:
            conclusion_idx = out0_nodes[0]
            if vertex_types.get(conclusion_idx) != VertexType.CLAIM:
                errors.append("The unique conclusion node must be a D-node (derivation).")


        # --- 6. In-degree checks ---
        # for idx, vtype in vertex_types.items():
        #     indeg = FG.in_degree(idx)
        #     if vtype == VertexType.DERIVATION:
        #         if indeg == 0 and FG.out_degree(idx) != 0:
        #             # errors.append(f"D-node {idx} must have positive in-degree in the primary graph.")
        #             errors.append(f"D-node {idx} is missing an arrow pointing to it.")

        # --- 7. Every B-node must be target of at least one trigger edge ---
        # b_nodes = [v.idx for v in graph.nodes if v.type == VertexType.BACKGROUND]
        # trigger_targets = {e.to_idx for e in graph.edges if e.type == EdgeType.TRIGGER}
        # for b_idx in b_nodes:
        #     if b_idx not in trigger_targets and G.out_degree(b_idx) != 0:
        #         errors.append(f"B-node {b_idx} must be the target of at least one trigger edge.")

        return GraphCheckResult(errors=errors, warnings=warnings)

    def process(self, annotations: Annotations, logger=None) -> StructuralIssues:
        input_names = self._input_names_snake
        assert len(input_names) == 1, "StructureProcessor receives != 1 input"
        input_name = input_names[0]
        assert input_name in ['graph_by_model', 'graph_by_human'], f"Unsupported input name: {input_name}"
        graph = getattr(annotations, input_name)
        graph_check_result = self._check_graph(graph)
        
        return StructuralIssues(
            errors=graph_check_result.errors,
            warnings=graph_check_result.warnings
        )
