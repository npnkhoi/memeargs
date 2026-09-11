"""Reproduce Table 1, Table 2, and the MemeArgs row of Table 3 from the paper.

Table 1 (node-type stats): for each node type -- Element, Premise, Claim,
Background -- the total count, average count per graph, total word count, and
average words per node. Premise is not a raw annotation: every node is
annotated as element/claim/background, and Graph.derive_premise_types() then
re-labels any non-conclusion node whose children are all Element nodes (or a
childless Claim) as Premise, exactly as used for evaluation elsewhere in this
repo.

Table 2 (edge-type stats): for each edge -- one per (premise, claim) pair,
expanded from the dataset's grouped `arguments` -- the total count and average
per graph, bucketed as Trigger if the premise node is an Element (regardless
of the recorded support/attack label), else Attack or Support per the
recorded label.

Table 3: only the MemeArgs row is computed here (#Doc = number of graphs,
#Comp = total node count from Table 1); the other rows in the paper's Table 3
are literature values for other corpora, not computed from this dataset.

Usage:
    conda run -n graph python -m scripts.dataset_stats
    conda run -n graph python -m scripts.dataset_stats --data data/memeargs_0826_fixed.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import extensions
from extensions.typing.graph import Graph

DEFAULT_DATA = Path("data/memeargs_0826_fixed.json")
NODE_ORDER = ["element", "premise", "claim", "background"]


def node_stats(items: list[dict]) -> tuple[Counter, Counter, int]:
    """Return (node counts, word counts, n_graphs) per type, after deriving Premise."""
    counts, words = Counter(), Counter()
    for it in items:
        g = Graph(**it["data"]["graph"])
        g.normalize()
        g.derive_premise_types()
        for v in g.nodes:
            counts[v.type.value] += 1
            words[v.type.value] += len(v.text.split())
    return counts, words, len(items)


def edge_stats(items: list[dict]) -> tuple[Counter, int]:
    """Return (edge counts, n_graphs) per bucket: support / attack / trigger."""
    counts = Counter()
    for it in items:
        graph = it["data"]["graph"]
        node_type = {n["idx"]: n["type"] for n in graph["nodes"]}
        for arg in graph["arguments"]:
            for premise_idx in arg["premises"]:
                if node_type.get(premise_idx) == "element":
                    counts["trigger"] += 1
                elif arg["type"] == "attack":
                    counts["attack"] += 1
                else:
                    counts["support"] += 1
    return counts, len(items)


def print_table1(counts: Counter, words: Counter, n_graphs: int) -> None:
    print("Table 1: node-type statistics")
    print(f"{'Node type':<12}{'ΣN':>7}{'#N/G':>8}{'ΣW':>9}{'#W/N':>8}")
    tot_n = sum(counts[t] for t in NODE_ORDER)
    tot_w = sum(words[t] for t in NODE_ORDER)
    for t in NODE_ORDER:
        n, w = counts[t], words[t]
        avg_w = w / n if n else 0.0
        print(f"{t.capitalize():<12}{n:>7}{n / n_graphs:>8.2f}{w / 1000:>8.1f}K{avg_w:>8.2f}")
    print(f"{'Total':<12}{tot_n:>7}{tot_n / n_graphs:>8.2f}{tot_w / 1000:>8.1f}K{tot_w / tot_n:>8.2f}")
    print()


def print_table2(counts: Counter, n_graphs: int) -> None:
    print("Table 2: edge-type statistics")
    print(f"{'Edge type':<10}{'ΣE':>7}{'#E/G':>8}")
    order = ["support", "attack", "trigger"]
    total = sum(counts[t] for t in order)
    for t in order:
        n = counts[t]
        print(f"{t.capitalize():<10}{n:>7}{n / n_graphs:>8.2f}")
    print(f"{'Total':<10}{total:>7}{total / n_graphs:>8.2f}")
    print()


def print_table3(n_graphs: int, n_components: int) -> None:
    print("Table 3: MemeArgs row (other rows are literature values, not computed here)")
    print(f"{'Corpus':<10}{'#Doc':>7}{'#Comp':>7}")
    print(f"{'MemeArgs':<10}{n_graphs:>7}{n_components:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()

    extensions.register()
    items = json.loads(args.data.read_text())

    n_counts, n_words, n_graphs = node_stats(items)
    e_counts, _ = edge_stats(items)

    print_table1(n_counts, n_words, n_graphs)
    print_table2(e_counts, n_graphs)
    print_table3(n_graphs, sum(n_counts[t] for t in NODE_ORDER))


if __name__ == "__main__":
    main()
