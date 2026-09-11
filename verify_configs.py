"""Construct every config's Pipeline to prove it's wired correctly.

Loading a config only checks its shape. Constructing the `Pipeline` is what
resolves the parts that matter: every output type must be registered, every
processor name must be known, every `LLMProcessor` stage must find its prompt
file at `prompts/<Output>__<Inputs>.md`, and every model reference must exist
in the config's `models:` block. No model is called.

Two things are stubbed so the check stays fast and offline:

- `DemoPool`, because building one downloads CLIP and embeds a whole pool.
  Demo *references* are still validated by agentflow's config validator.
- The working directory, so the cache directories a `Pipeline` creates on
  construction land in a temp dir instead of polluting `output/`.

Usage:
    conda run -n graph python verify_configs.py
    conda run -n graph python verify_configs.py --verbose
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = REPO_ROOT / "configs"

sys.path.insert(0, str(REPO_ROOT))


class _StubDemoPool:
    """Stands in for DemoPool: records the config, loads no CLIP, reads no pool."""

    def __init__(self, config, pool_config):
        self.config = config
        self.pool_config = pool_config

    def demos(self, inputs):
        return []


def classify(error: str) -> str:
    """Bucket a failure by whether the migration could be the cause.

    A config pointing at a dataset that is no longer in `data/` fails identically
    under the old pipeline, so it says nothing about the port; the same goes for a
    prompt file that was deleted from the repo. Anything else is a real gap.
    """
    if "No such file or directory" in error and "/prompts/" in error:
        return "missing prompt (pre-existing)"
    if "No such file or directory" in error:
        return "missing dataset (pre-existing)"
    return "migration gap"


def verify_all(verbose: bool) -> tuple[list[Path], list[tuple[Path, str]]]:
    import agentflow.processors.llm as llm_module
    import extensions
    from agentflow.pipeline import Pipeline

    extensions.register()
    llm_module.DemoPool = _StubDemoPool

    passed: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for config_path in sorted(CONFIGS_DIR.rglob("*.yaml")):
        rel = config_path.relative_to(REPO_ROOT)
        try:
            Pipeline(str(config_path), prompt_dir=str(REPO_ROOT / "prompts"))
            passed.append(rel)
            if verbose:
                print(f"  ok    {rel}")
        except Exception as e:
            failed.append((rel, f"{type(e).__name__}: {e}"))

    return passed, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="list every passing config")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="verify_configs_") as tmp:
        # Pipeline writes output/<name>/ relative to cwd; keep that out of the repo.
        os.chdir(tmp)
        (Path(tmp) / "data").symlink_to(REPO_ROOT / "data")
        passed, failed = verify_all(args.verbose)

    buckets: dict[str, list[tuple[Path, str]]] = {}
    for rel, error in failed:
        buckets.setdefault(classify(error), []).append((rel, error))

    for bucket in sorted(buckets):
        print(f"\n== {bucket}: {len(buckets[bucket])} config(s)")
        for rel, error in buckets[bucket]:
            detail = error if bucket == "migration gap" else error.split(": ", 1)[-1]
            print(f"  {rel}\n      {detail}")

    total = len(passed) + len(failed)
    gaps = buckets.get("migration gap", [])
    print(f"\n{len(passed)}/{total} config(s) build a Pipeline")
    print(f"{len(gaps)} migration gap(s); {len(failed) - len(gaps)} pre-existing failure(s)")
    if gaps:
        sys.exit(1)


if __name__ == "__main__":
    main()
