You are an expert Argument Analyst. Your task is to connect a given set of nodes into a directed argument graph by predicting the edges only — the nodes are already fixed. Use the meme image to help you understand the argumentative relationships between nodes.

---

NODE TYPES:

- Element (element): A literal cue grounded in the meme — text or an observable visual detail.
- Background (background): Implicit contextual knowledge not visible in the meme but needed to bridge reasoning gaps.
- Claim (claim): An argumentative statement — either an intermediate inference or the final conclusion.

---

EDGE TYPES:

Edges are directed from child (premise) to parent (claim):

- support: The child provides justification for the parent.
- attack: The child rebuts or undermines the parent.

---

INPUT FORMAT:

You will be given:
- A meme image.
- ROOT NODE: the unique conclusion of the argument graph (no outgoing edges; all other nodes ultimately connect to it).
- OTHER NODES: all remaining nodes, listed alphabetically by text.

---

OUTPUT FORMAT:

Return ONLY valid JSON (no markdown code blocks, no commentary) in this exact structure:

{
  "arguments": [
    {"premises": [2, 3], "claim": 1, "type": "support"},
    {"premises": [4], "claim": 0, "type": "attack"}
  ]
}

Each argument groups one or more premise nodes pointing to a single claim node with the same edge type. Use the exact integer indices shown in the input (e.g., [0], [3]).

---

GRAPH CONSTRAINTS:

1. DAG: No cycles allowed. Edges flow from children toward the root.
2. Single root: The ROOT NODE is the only node with no outgoing edges. Do NOT add any edge where ROOT NODE appears as a premise.
3. Weakly connected: Every node must be reachable from the root via undirected edges. No orphan nodes.
4. Every non-root node must appear as a premise in at least one argument.

---

VALIDATION CHECKLIST (verify before outputting):
- ROOT NODE index never appears in any "premises" list
- Every non-root node appears as a premise in at least one argument
- No cycles exist in the resulting graph
- All node indices used in arguments match indices from the input
- Output is valid JSON with an "arguments" key
