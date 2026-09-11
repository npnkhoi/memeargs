# MemeArgs

Code and data for **"MemeArgs: Vision-Language Argument Mining in Persuasive
Memes"** (Nguyen, Nguyen, Li, Patel, and Ng; Human Language Technology
Research Institute, University of Texas at Dallas). The paper is included in
this repo as [`paper.pdf`](paper.pdf).

This repo builds argument graphs from meme images with a configurable chain of
VLM stages, and reproduces every data-driven table in the paper (Tables 1, 2,
3, 4, 5, and 9). The pipeline machinery itself — loading, staging, caching,
few-shot demos, model backends — is
[agentflow](https://github.com/npnkhoi/agentflow); this repo holds the domain
half: the argument-graph types, the processors, the prompts, and the configs.

```
run.py             entry point: register extensions, run a config
configs/*.yaml     the 25 pipeline configs behind Tables 4, 5, and 9
prompts/           <Output>__<Input1>_<Input2>.md, resolved by stage types
data/              the MemeArgs corpus (images downloaded separately)
extensions/        the domain package registered with agentflow
scripts/           reproduces the paper's tables from pipeline output
hosting/           vLLM launch scripts for the four self-hosted models
output/            created when you run a config: <Stage>/<item_id>/output.json
```

## Setup

```bash
conda create -n memeargs python=3.11
conda activate memeargs
pip install -r requirements.txt
```

Download the [images](https://drive.google.com/file/d/1ENe98uIJZS4TZnX8G4JqpyvB8CnHH5JK/view?usp=sharing)
and put the `memeargs_img/` folder under `data/` (see `data/README.md`).

Copy `.env.example` to `.env` and fill in:

```bash
LOCAL_VLM_URL=http://<host>:8000/v1   # one self-hosted open model at a time
OPENROUTER_API_KEY=...                # Grok generation + the Gemini-3-Flash judge
```

## Running the pipeline

Five models are evaluated: **Gemma-3-27B**, **Mistral-Small-3.2-24B**,
**Qwen2.5-VL-32B**, and **InternVL2.5-26B** (self-hosted with vLLM), and
**Grok-4.20** (via OpenRouter, no hosting needed). See `hosting/*.sh` for the
vLLM launch commands (each expects 2×48GB GPUs); start the server for the
model you want, point `LOCAL_VLM_URL` at it, then:

```bash
python run.py configs/gemma3_27b_4shot_graphbased_bottomup_similar.yaml
```

Results land in `output/<config-name>/<StageName>/<item_id>/output.json`, and
every item/stage pair is cached, so a re-run only does missing work.

`configs/` has 25 files:
- `{model}_4shot_graphbased_bottomup_similar.yaml` × 5 models — the main
  end-to-end setting (Table 4).
- `{model}_4shot_graphbased_{bottomup,topdown}_{similar,random}.yaml` × 5
  models × 3 remaining combinations — the ablation over generation order and
  demonstration selection (Table 9).
- `edge_construction_image_{model}.yaml` × 5 models — edge construction given
  the gold nodes (Table 5).

Sanity-check that every config still resolves its types/processors/prompts
without calling any model:

```bash
python verify_configs.py --verbose
```

## Reproducing the tables

**Tables 1, 2, 3** (dataset statistics) need no pipeline run:

```bash
conda run -n memeargs python -m scripts.dataset_stats
```

**Tables 4, 5, 9** (model results) need the 25 configs' output first, then:

```bash
conda run -n memeargs python -m scripts.conclusion_match   # live OpenRouter judge calls, cached
conda run -n memeargs python -m scripts.results_tables     # prints Tables 4, 5, and 9
```

`scripts/conclusion_match.py` is the only script that makes live model calls
(the pairwise conclusion-equivalence judge); its results are cached to
`tmp/concl_pairwise_cache.json` so re-runs are free. Everything else in
`scripts/` is deterministic post-processing over `output/`.

Per the paper's Compute Budget section, running every experiment end-to-end
costs approximately 50 local GPU hours plus about $20 in OpenRouter costs
(Grok generation and the Gemini-3-Flash judge/matcher calls).

Table 10 (human evaluation) is not reproduced here — it depends on a
Label-Studio annotation dump that identifies the nine student annotators, and
the paper itself treats it as a secondary sanity check on the automatic
protocol, not a result to be re-derived.

## Concepts

A **config** declares the whole experiment: which dataset to load, which
models are available, and the sequence of stages.

A **stage** names its inputs, its output type, and the processor that produces
it. Inputs are either `human` (a field of the dataset item) or `model` (a
previous stage's output).

```yaml
- inputs:
  - [Image, human]
  output: BottomUpGraph
  processor: LLMProcessor
  model: local_vlm
  demo:
    select: similar
    shots: 4
    pool: memeargs_0826_fixed_train
```

A **processor** turns those inputs into the output. `LLMProcessor` asks a
model, using `prompts/<Output>__<Inputs>.md` as the system prompt; every other
processor here is deterministic — converting a generated graph, computing
node/edge F1, matching-aware graph edit distance, and so on.

## Data

See `data/README.md`. In short: `data/memeargs_0826_fixed.json` is the
corpus (614 argument graphs); `_train`/`_test` are the 348/266 split used
throughout the paper.

## Citation

The paper will appear in the main conference of EMNLP 2026. Cite it as:

```bibtex
@inproceedings{nguyen2026memeargs,
  title     = {MemeArgs: Vision-Language Argument Mining in Persuasive Memes},
  author    = {Nguyen, Khoi P. N. and Nguyen, Trung N. and Li, Terrence and Patel, Teerth and Ng, Vincent},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  note      = {To appear. See paper.pdf in this repository; update with page numbers once the proceedings are published.},
  url       = {https://github.com/npnkhoi/memeargs}
}
```

## License

MIT — see `LICENSE`.
