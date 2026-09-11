import networkx as nx
from extensions.const import DerivationEdgeType, EdgeType, VertexType
from copy import deepcopy
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



class VertexSetState(BaseModel):
    image: str  # image filename
    vertices: list[Vertex]


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
        # Changed from edges to arguments format
        # ret += "\nEdges:\n"
        # for edge in self.edges:
        #     ret += f"{edge.from_idx} {edge.to_idx} {edge.type.name}\n"
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


class GraphState(BaseModel):
    image: str  # image filename
    nodes: list[Vertex]
    edges: list[Edge]

    def normalize(self) -> None:
        """Normalize indices for nodes and edges in-place."""
        idx_map: dict[int, int] = {}
        for i, v in enumerate(self.nodes):
            idx_map[v.idx] = i
            v.idx = i
        for e in self.edges:
            if e.from_idx in idx_map:
                e.from_idx = idx_map[e.from_idx]
            if e.to_idx in idx_map:
                e.to_idx = idx_map[e.to_idx]


Premises = list[Vertex]
Conclusion = Vertex


class Claim(BaseModel):
    idx: int
    text: str


class DerivationEdge(BaseModel):
    id: int
    type: DerivationEdgeType


class Derivation(BaseModel):
    claim: Claim
    edges_from_child_nodes: list[DerivationEdge]


class DerivationOutput(BaseModel):
    derivations: list[Derivation]
    skip: bool
    comment: str


class ArgumentGraphConversionResult(BaseModel):
    argumentative: bool
    nodes: list[Vertex]
    comments: str


class Segment(BaseModel):
    id: int
    text: str
    start_idx: int # inclusive
    end_idx: int # exclusive


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


from typing import Literal

class ConclusionAudit(BaseModel):
    verdict: Literal["correct", "found_in_claims", "in_place", "brand_new"]
    # Populated only when verdict != "correct":
    reason: str | None = None
    # Populated only when verdict == "found_in_claims":
    correct_conclusion_idx: int | None = None
    # Populated only when verdict is "in_place" or "brand_new":
    # in_place  — refinement the same premises still directly support
    # brand_new — substantially different claim requiring a new argument structure
    new_conclusion: str | None = None


class ExistingElementVerdict(BaseModel):
    node_idx: int                   # references graph node idx
    kind: Literal["textual", "visual"]
    action: Literal["correct", "edit", "retype", "delete"]
    errors: list[Literal["over_inference", "wrong_cutoff", "non_descriptive"]] | None = None
    reason: str | None = None           # required for retype/delete; context for edit
    new_content: str | None = None      # for edit: replacement text
    new_type: str | None = None         # for retype: the correct VertexType string


class MissingElementNode(BaseModel):
    kind: Literal["textual", "visual"]
    content: str


class ElementAudit(BaseModel):
    existing: list[ExistingElementVerdict]  # one entry per existing element node
    missing: list[MissingElementNode]       # new element nodes to ADD


class EquivalentClaimPair(BaseModel):
    idx_a: int   # node idx of the first claim
    idx_b: int   # node idx of the second claim
    reason: str  # brief justification for the equivalence
    keep_idx: int  # idx of the node to keep as the merged representative


class ClaimEquivalencePairs(BaseModel):
    pairs: list[EquivalentClaimPair]


class ClaimParent(BaseModel):
    node_idx: int
    type: Literal["support", "attack"]  # how this new claim relates to the parent


class ClaimArgument(BaseModel):
    premises: list[int]
    type: Literal["support", "attack"]  # how these premises relate to this new claim


class MissingClaimNode(BaseModel):
    text: str
    reason: str
    parents: list[ClaimParent]      # usually 1 — where this claim plugs into above
    arguments: list[ClaimArgument]  # what supports/attacks this new claim from below


class ExistingClaimVerdict(BaseModel):
    node_idx: int
    action: Literal["correct", "edit", "delete"]
    errors: list[Literal["unresolved_reference", "meta_commentary"]] | None = None
    reason: str | None = None    # required for delete; context for edit
    new_text: str | None = None  # required for edit


class ClaimAudit(BaseModel):
    existing: list[ExistingClaimVerdict]  # one entry per intermediate claim node
    missing: list[MissingClaimNode]


class MergeAction(BaseModel):
    type: Literal["merge"]
    arg_idxs: list[int]   # >= 2 argument indices (as-is from graph.arguments)


class ExpandAction(BaseModel):
    type: Literal["expand"]
    arg_idx: int          # argument index from graph.arguments
    node_idx: int         # node idx of an existing Element, Background, or Claim node to add as premise


class AddAction(BaseModel):
    type: Literal["add"]
    premises: list[int]   # node indices (Element, Background, or Claim) for a brand-new argument


class ClaimArgumentAudit(BaseModel):
    claim_idx: int
    actions: list[MergeAction | ExpandAction | AddAction]  # empty = claim is fine


class ArgumentAudit(BaseModel):
    claims: list[ClaimArgumentAudit]


class PremiseRemoval(BaseModel):
    arg_idx: int
    remove_premises: list[int]


class PremiseAudit(BaseModel):
    arguments: list[PremiseRemoval]


class BackgroundAddAction(BaseModel):
    arg_idx: int   # argument index to add the new background node as a premise
    content: str   # text content of the new background node


class BackgroundEditAction(BaseModel):
    node_idx: int      # node idx of the existing background node
    new_content: str   # replacement text


class BackgroundRemoveAction(BaseModel):
    node_idx: int   # node idx of the background node to remove


class BackgroundAudit(BaseModel):
    add: list[BackgroundAddAction] = []
    edit: list[BackgroundEditAction] = []
    remove: list[BackgroundRemoveAction] = []


# Graph after conclusion audit applied
ConclusionAuditedGraph = Graph

# Graph after element audit applied
ElementAuditedGraph = Graph

# Graph after argument audit applied
ArgumentAuditedGraph = Graph

# Graph after ungrounded claims are removed
UngroundedClaimsRemovedGraph = Graph

# Graph after premise audit applied
PremiseAuditedGraph = Graph

# Graph after background audit applied
BackgroundAuditedGraph = Graph

# Graph after orphan nodes are removed
OrphanRemovedGraph = Graph

# Graph after claim merging — structurally identical to Graph, distinct name for pipeline caching
ClaimMergedGraph = Graph

# Graph after claim audit applied — structurally identical to Graph, distinct name for pipeline caching
ClaimAuditedGraph = Graph


def top_down_graph_to_graph(td: "TopDownGraph") -> Graph:
    from extensions.const import EdgeType
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
    from extensions.const import EdgeType
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
    from extensions.const import EdgeType
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
    from extensions.const import EdgeType
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

def merge_pair(graph: Graph, keep: int, remove: int) -> Graph:
    """
    For now, only support merging a child claim (remove) into its parent (keep), 
    which is the only case that arises in our current conclusion audit setting. 
    More general merges (e.g. siblings) would require more complex handling of the argument structure, and are left for future work.
    
    Logic:
    - Gather all unique children of REMOVE into a list
    - For every argument where REMOVE is a premise:
        - If claim is KEEP, add the children of REMOVE to the premises
        - If claim is not KEEP, raise an error due to divergent argument
    """
    graph = deepcopy(graph)

    # In nodes, remove the REMOVE node
    graph.nodes = [n for n in graph.nodes if n.idx != remove]
    
    # In arguments:
    
    # Pre-compute remove's unique children (premises of all arguments whose claim == remove),
    # remapped (remove→keep) and filtered to exclude keep itself (would be self-loop).
    children_of_remove: list[int] = []
    for arg in graph.arguments:
        if arg.claim == remove:
            children_of_remove.extend(arg.premises)
    children_of_remove = list(set(children_of_remove)) 

    new_arguments: list[Argument] = []
    for arg in graph.arguments:
        if arg.claim == remove:
            # skip all arguments where REMOVE is the claim (already removed)
            continue
        elif remove in arg.premises:
            if arg.claim == keep:
                # premises used to be {REMOVE, ...}
                # becomes {children_of_REMOVE, ...}
                new_premises = [p for p in arg.premises if p != remove]  # remove the REMOVE premise
                new_premises.extend(children_of_remove)
                new_arguments.append(Argument(claim=arg.claim, premises=new_premises, type=arg.type))
            else: # REMOVE makes a divergent argument
                raise ValueError("Cannot merge parent and child of a divergent argument")
        else: # REMOVE is not involved here
            new_arguments.append(arg)
    
    graph.arguments = new_arguments

    return graph


class EdgeConstructionOutput(BaseModel):
    arguments: list[Argument]

