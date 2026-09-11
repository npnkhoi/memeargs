"""Run a meme2graph pipeline from a YAML config.

Registers the domain extensions (graph types, processors, input formats, demo
strategies) with agentflow, then executes the config over the dataset it names.
Output lands in `output/<pipeline-name>/<StageName>/<item_id>/`, and completed
item/stage pairs are skipped on a re-run.

Usage:
    python run.py configs/0328_audit_small.yaml
    python run.py configs/0328_audit_small.yaml --prompt-dir prompts
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Config values like `token: ${GEMINI_API_KEY}` are expanded from the environment
# when the config loads, so .env has to be read before anything imports agentflow.
# agentflow's own load_dotenv() cannot find it: called from site-packages, dotenv
# searches upward from there rather than from this project.
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

from agentflow.client import Client  # noqa: E402  (must follow load_dotenv)

import extensions  # noqa: E402

PROMPT_DIR = "prompts"


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=argparse.FileType("r"), help="YAML config file")
    parser.add_argument(
        "--prompt-dir",
        default=PROMPT_DIR,
        help=f"directory holding the prompt files (default: {PROMPT_DIR})",
    )
    return parser


def main() -> None:
    args = arg_parser().parse_args()
    extensions.register()
    Client(prompt_dir=args.prompt_dir).run(args.config)


if __name__ == "__main__":
    main()
