"""Conclusion Matching Rate (Concl. column in Tables 4 and 9): a direct
pairwise semantic-equivalence test between the two graphs' declared
conclusions.

  - Gold conclusion  = node idx 0 of GraphByHuman  (the annotators' conclusion)
  - Model conclusion = node idx 0 of GraphByModel  (the model's declared
                        conclusion; graphs are generated bottom-up, so the
                        conclusion is always node 0 -- see prompts/BottomUpGraph__Image.md)
  - Judge            = google/gemini-3-flash-preview via OpenRouter, given the
                       meme image and both conclusion texts in isolation, using
                       the same synonymy criteria as the node-matching prompt
                       (prompts/MatchingResult__Image_GraphByHuman_GraphByModel.md):
                       same entities, model conclusion no less specific, same
                       quality/action.

Reports, per model and per setting, the fraction of test-set memes for which
the judge rated the two conclusions equivalent. This is the ONE script in this
repo that makes live model calls -- everything else here is deterministic
post-processing over `output/`. Results are cached per (model, setting, item)
in tmp/concl_pairwise_cache.json, so re-runs after the first are free, and
scripts/results_tables.py reads that same cache for its Concl. column.

Requires OPENROUTER_API_KEY (see .env.example) and `output/<config>/{GraphByHuman,GraphByModel}/*/output.json`
already populated by running the corresponding configs (see the top-level README).

Usage:
    conda run -n graph python -m scripts.conclusion_match
    conda run -n graph python -m scripts.conclusion_match --fresh   # ignore the cache
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

OUT = Path("output")
IMG_DIR = Path("data/memeargs_img")
CACHE = Path("tmp/concl_pairwise_cache.json")
JUDGE_MODEL = "google/gemini-3-flash-preview"

MODELS = [
    ("intern26b", "InternVL2.5-26B"),
    ("mistral24b", "Mistral-Small-24B"),
    ("qwen32b", "Qwen2.5-VL-32B"),
    ("gemma3_27b", "Gemma-3-27B"),
    ("grok41fast", "Grok-4.20"),
]

# bottomup_similar feeds Table 4 (main results); the other three feed Table 9 (ablation).
SETTINGS = [
    ("bottomup_similar", "similar / bottom-up  (Table 4)"),
    ("topdown_similar", "similar / top-down  (Table 9)"),
    ("bottomup_random", "random / bottom-up  (Table 9)"),
    ("topdown_random", "random / top-down  (Table 9)"),
]

SYSTEM = (
    "You are an expert argument analyst. You are given a meme image, the "
    "GOLD conclusion of its argument (written by an expert annotator), and a "
    "MODEL conclusion (produced by a system under evaluation). Decide whether "
    "the MODEL conclusion demonstrates a correct understanding of the same main "
    "claim as the GOLD conclusion.\n\n"
    "They match if and only if ALL hold:\n"
    "1. [Same entity] They are about the same entities (different surface forms "
    "are fine; resolve pronouns/definite descriptions using the image).\n"
    "2. [No less specific] The MODEL conclusion is no less specific than the "
    "GOLD conclusion (the model conclusion entails the gold one). Extra detail "
    "in the model conclusion is allowed only if it is accurate and relevant to "
    "the meme's argument.\n"
    "3. [Same quality/action] They ascribe roughly the same quality or action "
    "to those entities.\n"
    "When in doubt, do not match. Reply with a JSON object "
    '{"equivalent": true|false, "reason": "<one sentence>"}.'
)


def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in env or .env (see .env.example)")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _img_data_uri(item_id: str) -> str | None:
    stem = item_id.split("__")[-1]
    p = IMG_DIR / f"{stem}.png"
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _node0_text(path: Path) -> str | None:
    d = json.loads(path.read_text())
    nodes = d.get("nodes") or []
    return nodes[0]["text"] if nodes else None


def _judge(client: OpenAI, img_uri: str | None, gold: str, model: str) -> bool:
    content = [
        {"type": "text",
         "text": f"GOLD conclusion:\n{gold}\n\nMODEL conclusion:\n{model}"},
    ]
    if img_uri:
        content.insert(0, {"type": "image_url", "image_url": {"url": img_uri}})
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return bool(json.loads(resp.choices[0].message.content)["equivalent"])
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    retry ({e.__class__.__name__})", flush=True)


def main() -> None:
    fresh = "--fresh" in sys.argv
    client = _client()
    cache: dict = {} if fresh else (json.loads(CACHE.read_text()) if CACHE.exists() else {})
    lock = threading.Lock()

    def work(ck: str, item_id: str, gold: str, mdl: str) -> None:
        if ck in cache:
            return
        img = _img_data_uri(item_id)
        verdict = _judge(client, img, gold, mdl)
        with lock:
            cache[ck] = {"equivalent": verdict, "gold": gold, "model": mdl}

    print(f"Conclusion Matching Rate  (judge = {JUDGE_MODEL})")
    for suffix, desc in SETTINGS:
        print(f"\n=== {desc}  [{suffix}] ===")
        for key, label in MODELS:
            base = OUT / f"{key}_4shot_graphbased_{suffix}"
            hdir = base / "GraphByHuman"
            if not hdir.exists():
                print(f"{label:18s}  (no output -- run configs/{key}_4shot_graphbased_{suffix}.yaml first)")
                continue
            jobs = []
            for hp in sorted(hdir.glob("*/output.json")):
                iid = hp.parent.name
                mp = base / "GraphByModel" / iid / "output.json"
                if not mp.exists():
                    continue
                gold, mdl = _node0_text(hp), _node0_text(mp)
                if not gold or not mdl:
                    continue
                jobs.append((f"{key}/{suffix}/{iid}", iid, gold, mdl))

            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(lambda a: work(*a), jobs))
            with lock:
                CACHE.parent.mkdir(exist_ok=True)
                CACHE.write_text(json.dumps(cache, indent=1))

            verdicts = [cache[ck]["equivalent"] for ck, *_ in jobs]
            n, m = len(verdicts), sum(verdicts)
            if n:
                print(f"{label:18s}  n={n:3d}  concl={m / n:.3f} ({m})")


if __name__ == "__main__":
    main()
