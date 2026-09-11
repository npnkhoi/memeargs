"""Reproduce Table 4 (main results), Table 5 (edge construction given gold
nodes), and Table 9 (ablation: the three other prompt-construction settings)
from pipeline output.

Aggregation matches the paper (§5.1):
  - Node P/R/F1 per type (Element, Premise, Claim, Background): macro,
    recomputed per graph from the raw match counts in
    GeminiGraphAutoEvalResult, then averaged across the test set.
  - Edge P/R/F1: split into logos (Claim/Background source, support+attack
    merged) and trigger (Element source) via TypedEdgeEvalResult, each macro
    the same way.
  - GED: mean of MatchingGEDResult.ged (the matching-aware graph edit
    distance defined in §5.1).
  - Concl.: read from scripts/conclusion_match.py's cache
    (tmp/concl_pairwise_cache.json) -- run that script first.

Requires `output/<config>/{GeminiGraphAutoEvalResult,TypedEdgeEvalResult,
MatchingGEDResult}/*/output.json` from having run the corresponding configs
(see the top-level README), and `tmp/concl_pairwise_cache.json` from
`scripts/conclusion_match.py`. Everything here is deterministic post-processing
over cached files -- no model calls.

Usage:
    conda run -n graph python -m scripts.results_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.conclusion_match import CACHE as CONCL_CACHE
from scripts.conclusion_match import MODELS

OUT = Path("output")

EDGE_FOLDER = {k: f"edge_construction_image_{s}" for k, s in [
    ("intern26b", "intern"), ("mistral24b", "mistral"), ("qwen32b", "qwen"),
    ("gemma3_27b", "gemma"), ("grok41fast", "grok"),
]}

SETTINGS = {  # bottomup_similar -> Table 4; the rest -> Table 9
    "bottomup_similar": "Table 4 -- similar demos, bottom-up",
    "topdown_similar": "Table 9 -- similar demos, top-down",
    "bottomup_random": "Table 9 -- random demos, bottom-up",
    "topdown_random": "Table 9 -- random demos, top-down",
}

# node groups: (label, matched-count key, human-count key, model-count key)
# in GeminiGraphAutoEvalResult
NODE_GROUPS = [
    ("Element", "n_matched_human_element_nodes", "n_human_element_nodes", "n_model_element_nodes"),
    ("Premise", "n_matched_human_premise_nodes", "n_human_premise_nodes", "n_model_premise_nodes"),
    ("Claim", "n_matched_human_claim_nodes", "n_human_claim_nodes", "n_model_claim_nodes"),
    ("Background", "n_matched_human_background_nodes", "n_human_background_nodes", "n_model_background_nodes"),
]
# edge buckets in TypedEdgeEvalResult
EDGE_BUCKETS = [
    ("Logos", "n_matched_logos_edges", "n_human_logos_edges", "n_model_logos_edges"),
    ("Trigger", "n_matched_trigger_edges", "n_human_trigger_edges", "n_model_trigger_edges"),
]


def _load(evaldir: Path) -> list[dict]:
    if not evaldir.exists():
        return []
    return [json.loads(p.read_text()) for p in evaldir.glob("*/output.json")]


def _per_graph_prf(tp: int, n_human: int, n_model: int) -> tuple[float, float, float]:
    p = tp / n_model if n_model > 0 else 1.0
    r = tp / n_human if n_human > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _macro(rows: list[dict], mk: str, hk: str, modek: str) -> np.ndarray:
    prf = [_per_graph_prf(r.get(mk, 0), r.get(hk, 0), r.get(modek, 0)) for r in rows]
    return np.mean(prf, axis=0) if prf else np.zeros(3)


def _concl_rate(key: str, suffix: str) -> float | None:
    if not CONCL_CACHE.exists():
        return None
    cache = json.loads(CONCL_CACHE.read_text())
    verdicts = [d["equivalent"] for k, d in cache.items() if k.startswith(f"{key}/{suffix}/")]
    return sum(verdicts) / len(verdicts) if verdicts else None


def _row(base: Path, key: str, suffix: str) -> str | None:
    node_dir = base / "GeminiGraphAutoEvalResult"
    nrows = _load(node_dir)
    if not nrows:
        return None
    concl = _concl_rate(key, suffix)
    cells = [f"{concl:.3f}" if concl is not None else "--"]
    for _, mk, hk, modek in NODE_GROUPS:
        cells += [f"{v:.3f}" for v in _macro(nrows, mk, hk, modek)]
    erows = _load(base / "TypedEdgeEvalResult")
    for _, mk, hk, modek in EDGE_BUCKETS:
        cells += [f"{v:.3f}" for v in _macro(erows, mk, hk, modek)] if erows else ["--", "--", "--"]
    ged = [r["ged"] for r in _load(base / "MatchingGEDResult")]
    cells.append(f"{np.mean(ged):.3f}" if ged else "--")
    return f"  n={len(nrows):3d}  " + " & ".join(cells)


def main() -> None:
    hdr = "Concl. | Element P R F | Premise P R F | Claim P R F | Background P R F | Logos-edge P R F | Trigger-edge P R F | GED"
    for suffix, title in SETTINGS.items():
        print(f"\n=== {title}  [{suffix}] ===")
        print(f"    ({hdr})")
        for key, label in MODELS:
            row = _row(OUT / f"{key}_4shot_graphbased_{suffix}", key, suffix)
            print(f"{label:18s}{row if row else '  (no output yet)'}")

    print("\n=== Table 5 -- edge construction given gold nodes  (Logos P R F | Trigger P R F) ===")
    for key, label in MODELS:
        base = OUT / EDGE_FOLDER[key]
        rows = _load(base / "TypedEdgeEvalResult")
        if not rows:
            print(f"{label:18s}  (no output yet)")
            continue
        cells = []
        for _, mk, hk, modek in EDGE_BUCKETS:
            cells += [f"{v:.3f}" for v in _macro(rows, mk, hk, modek)]
        print(f"{label:18s}  n={len(rows):3d}  " + " & ".join(cells))


if __name__ == "__main__":
    main()
