# Data

- `memeargs_0826_fixed.json` — the full MemeArgs corpus, 614 argument graphs
  over persuasive memes (see the paper, §4).
- `memeargs_0826_fixed_train.json` / `memeargs_0826_fixed_test.json` — the
  348/266 train/test split used throughout the paper (demo pool for few-shot
  prompting, and the evaluation set for every result table).
- `guidelines.md` — the human annotation guidelines used to build the corpus
  (paper Appendix C).

Each graph item has the shape:

```json
{
  "id": "argtree_2026__0145__149_image",
  "data": {
    "id": "...",
    "image": "149_image.png",
    "graph": {
      "nodes": [{"idx": 0, "text": "...", "type": "claim"}, ...],
      "arguments": [{"claim": 0, "premises": [1, 4, 5], "type": "support"}, ...]
    }
  }
}
```

`type` on a node is one of `element`, `claim`, `background` — `premise` is not
a raw annotation; it's derived at evaluation time by
`extensions.typing.graph.Graph.derive_premise_types()` (see Table 1 in the
paper).

## Images

Download the [images](https://drive.google.com/file/d/1ENe98uIJZS4TZnX8G4JqpyvB8CnHH5JK/view?usp=sharing)
and put the `memeargs_img/` folder under `data/`, so that e.g.
`data/memeargs_img/149_image.png` exists. Configs and scripts expect this
exact path. The images are not otherwise redistributed in this repository.
