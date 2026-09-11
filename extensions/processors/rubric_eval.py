from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import networkx

from extensions.const import VertexType
from agentflow.models import BaseLLM
from agentflow.pipeline import Pipeline
from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from agentflow.typing.config import StageConfig
from extensions.typing.eval import Mistake, Phase0Result, RubricEvalResult, StageResult
from extensions.typing.graph import Graph, Vertex

"""Rubric prompts live beside the pipeline's other prompts, under `rubric/`.

This processor drives six prompts itself rather than the one `LLMProcessor`
resolves from the stage's type names, so it reads them from the pipeline's
prompt directory directly.
"""
RUBRIC_SUBDIR = "rubric"


# ---------------------------------------------------------------------------
# Phase 0 — programmatic structural checks
# ---------------------------------------------------------------------------

def _run_phase0(graph: Graph) -> Phase0Result:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    # 1. Weak connectivity
    try:
        components = graph.connected_components()
        ok = len(components) == 1
        checks["weak_connectivity"] = ok
        if not ok:
            errors.append(f"Graph has {len(components)} disconnected components")
    except Exception as e:
        checks["weak_connectivity"] = False
        errors.append(f"Connectivity check failed: {e}")

    # 2. Unique root (exactly one conclusion, every other node is a premise somewhere)
    try:
        concls = graph.get_conclusions()
        ok = len(concls) == 1
        checks["unique_root"] = ok
        if not ok:
            errors.append(f"Expected 1 root, found {len(concls)}: {concls}")
    except Exception as e:
        checks["unique_root"] = False
        errors.append(f"Root check failed: {e}")

    # 3. Leaf constraint — Element/Background nodes must not be arg.claim
    claim_idxs = {arg.claim for arg in graph.arguments}
    node_map = {v.idx: v for v in graph.nodes}
    leaf_violations = [
        idx for idx in claim_idxs
        if idx in node_map and node_map[idx].type in (VertexType.ELEMENT, VertexType.BACKGROUND)
    ]
    checks["leaf_constraint"] = len(leaf_violations) == 0
    if leaf_violations:
        errors.append(f"Element/Background nodes used as claim parents: {leaf_violations}")

    # 4. Cycle detection
    try:
        graph.topological_order()
        checks["cycle_detection"] = True
    except networkx.NetworkXUnfeasible:
        checks["cycle_detection"] = False
        errors.append("Graph contains a cycle")

    # 5. Dangling references
    node_idxs = {v.idx for v in graph.nodes}
    dangling = set()
    for arg in graph.arguments:
        if arg.claim not in node_idxs:
            dangling.add(arg.claim)
        for p in arg.premises:
            if p not in node_idxs:
                dangling.add(p)
    checks["dangling_references"] = len(dangling) == 0
    if dangling:
        errors.append(f"Dangling idx references: {sorted(dangling)}")

    # 6. Self-loops
    self_loops = [
        arg.claim for arg in graph.arguments if arg.claim in arg.premises
    ]
    checks["self_loops"] = len(self_loops) == 0
    if self_loops:
        errors.append(f"Self-loop at claim nodes: {self_loops}")

    passed = all(checks.values())

    # Auto edge score on failure
    auto_edge_score = None
    if not passed:
        if not checks.get("weak_connectivity", True) or not checks.get("cycle_detection", True):
            auto_edge_score = 1
        elif not checks.get("unique_root", True) or not checks.get("dangling_references", True):
            auto_edge_score = 2
        else:
            auto_edge_score = 3

    return Phase0Result(
        passed=passed,
        checks=checks,
        errors=errors,
        auto_edge_score=auto_edge_score,
    )


# ---------------------------------------------------------------------------
# Graph → bottom-up children JSON for prompts
# ---------------------------------------------------------------------------

def _graph_to_bottom_up_json(
    graph: Graph,
    include_types: set[VertexType] | None = None,
    include_edges: bool = True,
) -> str:
    """Convert Graph to bottom-up JSON (mirrors BottomUpGraph format).

    Children lists merge support + attack (edge type not shown to judge).
    """
    graph = deepcopy(graph)
    graph.normalize()

    concl_idxs = graph.get_conclusions()
    assert len(concl_idxs) == 1, f"Expected 1 conclusion, got {len(concl_idxs)}"
    concl_idx = concl_idxs[0]

    node_map = {v.idx: v for v in graph.nodes}

    # Build parent → children mapping
    parent_children: dict[int, list[int]] = defaultdict(list)
    for arg in graph.arguments:
        for p in arg.premises:
            if p not in parent_children[arg.claim]:
                parent_children[arg.claim].append(p)

    # Classify nodes
    concl_node = node_map[concl_idx]
    leaves: list[Vertex] = []
    intermediates: list[Vertex] = []

    for v in graph.nodes:
        if v.idx == concl_idx:
            continue
        if include_types and v.type not in include_types:
            continue
        if v.type in (VertexType.ELEMENT, VertexType.BACKGROUND):
            leaves.append(v)
        elif v.type == VertexType.CLAIM:
            intermediates.append(v)

    def _node_dict(v: Vertex) -> dict:
        return {"idx": v.idx, "type": v.type.value, "text": v.text}

    # Build output
    result: dict = {
        "conclusion": _node_dict(concl_node),
    }

    result["leaves"] = [_node_dict(v) for v in leaves]

    if include_edges:
        # Filter children to only include visible nodes
        visible_idxs = {concl_idx} | {v.idx for v in leaves} | {v.idx for v in intermediates}

        inter_list = []
        for v in intermediates:
            d = _node_dict(v)
            d["children"] = [c for c in parent_children.get(v.idx, []) if c in visible_idxs]
            inter_list.append(d)
        result["intermediates"] = inter_list

        result["conclusion_children"] = [
            c for c in parent_children.get(concl_idx, []) if c in visible_idxs
        ]
    else:
        result["intermediates"] = [_node_dict(v) for v in intermediates]

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Input masking helpers — each stage builds its user prompt
# ---------------------------------------------------------------------------

def _mask_conclusion(graph: Graph) -> str:
    concl_idxs = graph.get_conclusions()
    concl = graph.nodes[concl_idxs[0]]
    return f"Conclusion node:\n  idx: {concl.idx}, text: \"{concl.text}\""


def _mask_elements(graph: Graph) -> str:
    return _graph_to_bottom_up_json(
        graph,
        include_types={VertexType.ELEMENT, VertexType.CLAIM},
        include_edges=False,
    )


def _mask_claims(graph: Graph) -> str:
    return _graph_to_bottom_up_json(
        graph,
        include_types={VertexType.ELEMENT, VertexType.CLAIM},
        include_edges=False,
    )


def _mask_edges(graph: Graph) -> str:
    """All nodes except BK, with edges among visible nodes."""
    return _graph_to_bottom_up_json(
        graph,
        include_types={VertexType.ELEMENT, VertexType.CLAIM},
        include_edges=True,
    )


def _mask_full(graph: Graph) -> str:
    return _graph_to_bottom_up_json(
        graph,
        include_types=None,  # all types
        include_edges=True,
    )


def _mask_holistic_informed(graph: Graph, result: RubricEvalResult) -> str:
    tree_json = _mask_full(graph)
    parts = [f"Argument Tree:\n{tree_json}\n"]

    stage_data = [
        ("Stage 1 — Conclusion", result.stage1_conclusion),
        ("Stage 2 — Elements", result.stage2_elements),
        ("Stage 3 — Claims", result.stage3_claims),
        ("Stage 4 — Edges", result.stage4_edges),
        ("Stage 5 — Background", result.stage5_background),
    ]
    parts.append("Prior Stage Evaluations:")
    for label, sr in stage_data:
        if sr is None:
            parts.append(f"\n{label}: (not evaluated)")
        else:
            mistakes_str = json.dumps(
                [m.model_dump() for m in sr.mistakes], indent=2
            )
            parts.append(
                f"\n{label}:\n  score: {sr.score}\n  mistakes: {mistakes_str}\n  reason: {sr.reason}"
            )

    if result.structural_fail:
        parts.append(f"\nStructural failures: {result.phase0.errors}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class RubricEvalProcessor(AnnotationProcessor):
    """Reference-free rubric evaluation: Phase 0 + 5 LLM stages + holistic."""

    def __init__(self, pipeline: Pipeline, stage_config: StageConfig):
        super().__init__(pipeline, stage_config)
        if stage_config.model is None:
            raise ValueError(
                "RubricEvalProcessor calls a model directly, so the stage must set "
                "`model:` to one of the names declared in the config's `models:` block."
            )
        self._vlm: BaseLLM = pipeline.get_model(stage_config.model)

        # Load prompts
        self._prompts: dict[str, str] = {}
        for name in [
            "stage1_conclusion",
            "stage2_elements",
            "stage3_claims",
            "stage4_edges",
            "stage5_background",
            "stage6_holistic_informed",
        ]:
            path = pipeline.prompt_dir / RUBRIC_SUBDIR / f"{name}.md"
            self._prompts[name] = path.read_text(encoding="utf-8")

    def _run_llm_stage(
        self,
        prompt_key: str,
        image_path: Path | None,
        user_prompt: str,
    ) -> StageResult | None:
        system_prompt = self._prompts[prompt_key]
        raw = self._vlm.generate(
            system_prompt=system_prompt,
            examples=[],
            output_type=StageResult,
            image_path=image_path,
            input_text=user_prompt,
        )
        if raw is None:
            return None
        return StageResult.model_validate_json(raw)

    def process(self, annotations: Annotations, logger=None) -> RubricEvalResult | None:
        graph = annotations.graph
        if graph is None:
            return None
        graph = deepcopy(graph)
        graph.normalize()

        image_path = None
        if annotations.image is not None:
            image_path = Path(annotations.image)

        # Phase 0
        phase0 = _run_phase0(graph)

        result = RubricEvalResult(phase0=phase0)

        if not phase0.passed:
            result.structural_fail = True
            result.stage4_edges = StageResult(
                score=phase0.auto_edge_score or 1,
                mistakes=[Mistake(node_idx=None, issue="structural_fail", detail="; ".join(phase0.errors))],
                reason=f"Auto-assigned due to structural failure: {'; '.join(phase0.errors)}",
            )
            # H-informed still runs
            informed_prompt = _mask_holistic_informed(graph, result)
            result.h_informed = self._run_llm_stage(
                "stage6_holistic_informed", image_path, informed_prompt
            )
            return result

        # Stage 1 — Conclusion
        s1_prompt = _mask_conclusion(graph)
        result.stage1_conclusion = self._run_llm_stage(
            "stage1_conclusion", image_path, s1_prompt
        )
        if result.stage1_conclusion is None:
            return None
        if result.stage1_conclusion.score <= 2:
            result.stopped_at = 1
            return self._finalize_holistic(graph, image_path, result)

        # Stage 2 — Elements
        s2_prompt = _mask_elements(graph)
        result.stage2_elements = self._run_llm_stage(
            "stage2_elements", image_path, s2_prompt
        )
        if result.stage2_elements is None:
            return None
        if result.stage2_elements.score <= 2:
            result.stopped_at = 2
            return self._finalize_holistic(graph, image_path, result)

        # Stage 3 — Intermediate Claims
        s3_prompt = _mask_claims(graph)
        result.stage3_claims = self._run_llm_stage(
            "stage3_claims", image_path, s3_prompt
        )
        if result.stage3_claims is None:
            return None
        if result.stage3_claims.score <= 2:
            result.stopped_at = 3
            return self._finalize_holistic(graph, image_path, result)

        # Stage 4 — Edges
        s4_prompt = _mask_edges(graph)
        result.stage4_edges = self._run_llm_stage(
            "stage4_edges", image_path, s4_prompt
        )
        if result.stage4_edges is None:
            return None

        # Stage 5 — Background
        s5_prompt = f"Argument Tree:\n{_mask_full(graph)}"
        result.stage5_background = self._run_llm_stage(
            "stage5_background", image_path, s5_prompt
        )
        if result.stage5_background is None:
            return None

        # H-avg
        scores = [
            result.stage1_conclusion.score,
            result.stage2_elements.score,
            result.stage3_claims.score,
            result.stage4_edges.score,
            result.stage5_background.score,
        ]
        result.h_avg = round(sum(scores) / len(scores), 2)

        return self._finalize_holistic(graph, image_path, result)

    def _finalize_holistic(
        self,
        graph: Graph,
        image_path: Path | None,
        result: RubricEvalResult,
    ) -> RubricEvalResult:
        # H-informed
        informed_prompt = _mask_holistic_informed(graph, result)
        result.h_informed = self._run_llm_stage(
            "stage6_holistic_informed", image_path, informed_prompt
        )

        return result
