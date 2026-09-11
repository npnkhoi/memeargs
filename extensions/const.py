## Constants

from enum import Enum


class VertexType(str, Enum):
    ELEMENT = "element"
    CLAIM = "claim"
    PREMISE = "premise"  # derived: a non-conclusion node with only-element children (or a childless claim)
    BACKGROUND = "background"
    NONE = "none"


class EdgeType(str, Enum):
    SUPPORT = "support"
    ATTACK = "attack"
    TRIGGER = "trigger"
    PARAPHRASE = "paraphrase"


class AnnotationSource(str, Enum):
    HUMAN = "human"
    MODEL = "model"


class DemoSelect(str, Enum):
    RANDOM = "random"
    SIMILAR = "similar"
    DIVERSE = "diverse"
    LONGEST = "longest"


class DemoPoolId(str, Enum):
    SEMEVAL = "semeval"
    VISARGS = "visargs"
    MEMEARGS = "memeargs"
    MEMEARGS_V5A = "memeargs_v5a"
    MEMEQA_TRAIN = "memeqa_train"
    MEMEARGS_PROCESSED_0307_TRAIN = "memeargs_processed_0307_train"

class InputFormat(str, Enum):
    CONCLUSION = "conclusion"
    ELEMENT = "element"
    BACKGROUND = "background"
    CLAIM = "claim"
    RELATION = "relation"
    NATURALIZED_ESSAY = "naturalized_essay"
    NODE_MATCHING = "node_matching"
    NODES_ALPHABETICAL = "nodes_alphabetical"
    CONCLUSION_AUDIT = "conclusion_audit"
    ELEMENT_AUDIT = "element_audit"
    CLAIM_AUDIT = "claim_audit"
    ARGUMENT_AUDIT = "argument_audit"
    PREMISE_AUDIT = "premise_audit"
    BACKGROUND_AUDIT = "background_audit"
    BOTTOM_UP_GRAPH = "bottom_up_graph"
