import networkx as nx
from extensions.const import EdgeType, VertexType
from pydantic import BaseModel, Field, model_validator
from itertools import groupby

class Vertex(BaseModel):
    idx: int
    text: str
    type: VertexType

    def label(self) -> str:
        return f"{self.idx}{self.type.name[0]}"

    def __str__(self):
        return f"{self.label()} {self.text}"


class Edge(BaseModel):
    from_idx: int
    to_idx: int
    type: EdgeType

class Argument(BaseModel):
    claim: int
    premises: list[int]
    type: EdgeType


class Graph(BaseModel):
    nodes: list[Vertex]
    arguments: list[Argument] = []
    edges: list[Edge] | None = Field(default=None, exclude=True)  # legacy, auto-converted on load

    @model_validator(mode='after')
    def convert_edges_to_arguments(self):
        """Convert legacy 'edge' into 'argument' with joint/independent edge handling"""
        if self.edges and not self.arguments:
            key = lambda e: (e.to_idx, e.type)
            sorted_edges = sorted(self.edges, key=key)
            self.arguments = [Argument(claim=claim, premises=[e.from_idx for e in group], type=etype) for (claim, etype), group in groupby(sorted_edges, key=key)]
        self.edges = None
        return self

    def flat_edges(self) -> list[Edge]:
        """Expand arguments into flat Edge list for internal use."""
        return [
            Edge(from_idx=p, to_idx=arg.claim, type=arg.type)
            for arg in self.arguments
            for p in arg.premises
        ]

    def normalize(self):
        idx_map: dict[int, int] = {}
        # compact vertex indices
        for i, v in enumerate(self.nodes):
            idx_map[v.idx] = i
            v.idx = i
        for arg in self.arguments:
            arg.claim = idx_map[arg.claim]
            arg.premises = [idx_map[p] for p in arg.premises]

    def _nxgraph(self):
        # build nx.Graph
        idx_to_vertex = {v.idx: v for v in self.nodes}
        primary_edges = [
            (p, arg.claim)
            for arg in self.arguments
            for p in arg.premises
        ]
        G = nx.DiGraph()
        G.add_nodes_from(idx_to_vertex.keys())
        G.add_edges_from(primary_edges)
        return G

    def get_conclusions(self) -> list[int]:
        # build nx.Graph
        # vertex_types = {v.idx: v.type for v in self.nodes}
        G = self._nxgraph()

        # get conclusions
        concls = [n for n in G.nodes if G.out_degree(n) == 0]
        return concls

    def get_nodes_by_type(self, vtype: VertexType) -> list[Vertex]:
        return [v for v in self.nodes if v.type == vtype]

    def derive_premise_types(self) -> None:
        """In place: retype nodes to PREMISE. A node becomes a premise iff it is
        NOT the conclusion (out-degree-0 node) and either (a) all its children
        are Element nodes, or (b) it is a Claim node with no children.
        Idempotent. Call after normalize()."""
        children: dict[int, list[int]] = {}
        premised: set[int] = set()
        for arg in self.arguments:
            premised.update(arg.premises)
            children.setdefault(arg.claim, []).extend(arg.premises)
        conclusion = {v.idx for v in self.nodes if v.idx not in premised}
        ntype = {v.idx: v.type for v in self.nodes}
        for v in self.nodes:
            if v.idx in conclusion:
                continue
            ch = children.get(v.idx, [])
            if (ch and all(ntype.get(c) == VertexType.ELEMENT for c in ch)) or \
               (v.type == VertexType.CLAIM and not ch):
                v.type = VertexType.PREMISE

    def depth(self) -> int:
        G = self._nxgraph()
        return nx.dag_longest_path_length(G)

    def short_print(self) -> str:
        ret = "Nodes:\n"

        for node in self.nodes:
            ret += f"{node.idx}{node.type.name[0]} "
        ret += "\nArguments:\n"
        for arg in self.arguments:
            ret += f"{arg.premises} -> {arg.claim} {arg.type.name}\n"
        return ret

    def topological_order(self) -> list[Vertex]:
        """
        Generate a topological ordering of the vertices in the graph.

        Returns:
            list[Vertex]: A list of vertices ordered topologically.

        Raises:
            NetworkXUnfeasible: If the graph contains a cycle, making topological
                sorting impossible.
        """
        G = self._nxgraph()
        sorted_indices = list(nx.topological_sort(G))
        idx_to_vertex = {v.idx: v for v in self.nodes}
        return [idx_to_vertex[idx] for idx in sorted_indices]

    def connected_components(self) -> list[list[Vertex]]:
        """
        Get the connected components of the graph.

        Returns:
            list[list[Vertex]]: A list of connected components, each represented
                as a list of vertices.
        """
        G = self._nxgraph()
        components = list(nx.weakly_connected_components(G))
        idx_to_vertex = {v.idx: v for v in self.nodes}
        return [
            [idx_to_vertex[idx] for idx in component]
            for component in components
        ]

# Used for evaluation
# identical, but different name
GraphByModel = Graph
GraphByHuman = Graph


# --- Top-down / Bottom-up schema models (internal use in LLMProcessor) ---

class TopDownLeafParent(BaseModel):
    idx: int
    supports: list[int]   # parent claim idxs this leaf supports
    attacks: list[int]    # parent claim idxs this leaf attacks


class TopDownIntermediate(BaseModel):
    idx: int
    text: str
    type: VertexType
    supports: list[int]   # parent claim idxs this node supports
    attacks: list[int]    # parent claim idxs this node attacks


class TopDownGraph(BaseModel):
    conclusion: Vertex
    leaves: list[Vertex]
    intermediates: list[TopDownIntermediate]
    leaf_parents: list[TopDownLeafParent]


class BottomUpIntermediate(BaseModel):
    idx: int
    text: str
    type: VertexType
    supporting_children: list[int]   # child idxs that support this node
    attacking_children: list[int]    # child idxs that attack this node


class BottomUpConclusionChildren(BaseModel):
    supporting_children: list[int]
    attacking_children: list[int]


class BottomUpGraph(BaseModel):
    conclusion: Vertex
    leaves: list[Vertex]
    intermediates: list[BottomUpIntermediate]
    conclusion_children: BottomUpConclusionChildren


def top_down_graph_to_graph(td: "TopDownGraph") -> Graph:
    nodes: list[Vertex] = [td.conclusion] + list(td.leaves)
    for inter in td.intermediates:
        nodes.append(Vertex(idx=inter.idx, text=inter.text, type=inter.type))
    edge_map: dict[tuple[int, EdgeType], list[int]] = {}
    for lp in td.leaf_parents:
        for parent in lp.supports:
            edge_map.setdefault((parent, EdgeType.SUPPORT), []).append(lp.idx)
        for parent in lp.attacks:
            edge_map.setdefault((parent, EdgeType.ATTACK), []).append(lp.idx)
    for inter in td.intermediates:
        for parent in inter.supports:
            edge_map.setdefault((parent, EdgeType.SUPPORT), []).append(inter.idx)
        for parent in inter.attacks:
            edge_map.setdefault((parent, EdgeType.ATTACK), []).append(inter.idx)
    return Graph(nodes=nodes, arguments=[
        Argument(claim=claim, premises=premises, type=etype)
        for (claim, etype), premises in edge_map.items()
    ])


def bottom_up_graph_to_graph(bu: "BottomUpGraph") -> Graph:
    nodes: list[Vertex] = [bu.conclusion] + list(bu.leaves)
    for inter in bu.intermediates:
        nodes.append(Vertex(idx=inter.idx, text=inter.text, type=inter.type))
    edge_map: dict[tuple[int, EdgeType], list[int]] = {}
    for inter in bu.intermediates:
        if inter.supporting_children:
            edge_map[(inter.idx, EdgeType.SUPPORT)] = list(inter.supporting_children)
        if inter.attacking_children:
            edge_map[(inter.idx, EdgeType.ATTACK)] = list(inter.attacking_children)
    if bu.conclusion_children.supporting_children:
        edge_map[(bu.conclusion.idx, EdgeType.SUPPORT)] = list(bu.conclusion_children.supporting_children)
    if bu.conclusion_children.attacking_children:
        edge_map[(bu.conclusion.idx, EdgeType.ATTACK)] = list(bu.conclusion_children.attacking_children)
    return Graph(nodes=nodes, arguments=[
        Argument(claim=claim, premises=premises, type=etype)
        for (claim, etype), premises in edge_map.items()
    ])


def _classify_nodes(graph: Graph) -> tuple[Vertex, list[Vertex], list[Vertex]]:
    """Returns (conclusion, leaves, intermediates)."""
    conclusion_idx = graph.get_conclusions()[0]
    claim_idxs = {arg.claim for arg in graph.arguments}
    conclusion = next(v for v in graph.nodes if v.idx == conclusion_idx)
    leaves = [v for v in graph.nodes if v.idx not in claim_idxs and v.idx != conclusion_idx]
    intermediates = [v for v in graph.nodes if v.idx in claim_idxs and v.idx != conclusion_idx]
    return conclusion, leaves, intermediates


def graph_to_top_down(graph: Graph) -> "TopDownGraph":
    conclusion, leaves, intermediates = _classify_nodes(graph)
    supports_map: dict[int, list[int]] = {}
    attacks_map: dict[int, list[int]] = {}
    for arg in graph.arguments:
        for p in arg.premises:
            if arg.type == EdgeType.SUPPORT:
                supports_map.setdefault(p, []).append(arg.claim)
            else:
                attacks_map.setdefault(p, []).append(arg.claim)
    leaf_parents = [
        TopDownLeafParent(idx=v.idx, supports=supports_map.get(v.idx, []), attacks=attacks_map.get(v.idx, []))
        for v in leaves
    ]
    td_intermediates = [
        TopDownIntermediate(idx=v.idx, text=v.text, type=v.type, supports=supports_map.get(v.idx, []), attacks=attacks_map.get(v.idx, []))
        for v in intermediates
    ]
    return TopDownGraph(conclusion=conclusion, leaves=leaves, intermediates=td_intermediates, leaf_parents=leaf_parents)


def graph_to_bottom_up(graph: Graph) -> "BottomUpGraph":
    conclusion, leaves, intermediates = _classify_nodes(graph)
    supporting_children_map: dict[int, list[int]] = {}
    attacking_children_map: dict[int, list[int]] = {}
    for arg in graph.arguments:
        if arg.type == EdgeType.SUPPORT:
            supporting_children_map.setdefault(arg.claim, []).extend(arg.premises)
        else:
            attacking_children_map.setdefault(arg.claim, []).extend(arg.premises)
    bu_intermediates = [
        BottomUpIntermediate(idx=v.idx, text=v.text, type=v.type,
                             supporting_children=supporting_children_map.get(v.idx, []),
                             attacking_children=attacking_children_map.get(v.idx, []))
        for v in intermediates
    ]
    conclusion_children = BottomUpConclusionChildren(
        supporting_children=supporting_children_map.get(conclusion.idx, []),
        attacking_children=attacking_children_map.get(conclusion.idx, []),
    )
    return BottomUpGraph(conclusion=conclusion, leaves=leaves, intermediates=bu_intermediates, conclusion_children=conclusion_children)


class EdgeConstructionOutput(BaseModel):
    arguments: list[Argument]
