# Annotation Guidelines

## Stage 1: Annotating the Conclusion

**Input.** A persuasive meme (image and overlaid text), along with the year and platform where it was posted.

**Output.** A single Claim node that captures the meme's main argument (the conclusion).

**Instructions.** Think from the perspective of someone who posted this meme in the given context. What is the single most important point they are trying to make? Write this as a direct, first-order assertion.

**Rules.**
- The conclusion must be a full clause with a subject and a predicate (e.g., "Vaccines cause autism," not "vaccine dangers").
- The conclusion must be about the subject matter, not about the meme itself. Write "Politicians are corrupt," not "The meme argues that politicians are corrupt."
- Do not hedge or soften. Capture the meme's actual stance, even if provocative or extreme.
- If the meme makes multiple distinct arguments, identify the one that subsumes or is supported by the others.
- If you genuinely cannot determine a single main claim, flag the instance for discussion.

---

## Stage 2: Annotating Elements

**Input.** The meme and its annotated conclusion.

**Output.** A set of Element nodes describing the visually salient details in the meme.

**Instructions.** Identify every textual and visual detail in the meme that is relevant to establishing the conclusion. Each Element node describes what is literally visible or readable on the meme.

**Rules.**
- **Textual elements:** Quote or closely paraphrase text appearing on the meme. If a sentence is split across the image but forms a single thought, keep it as one node. If text fragments are spatially separated and semantically independent, create separate nodes.
- **Visual elements:** Describe observable details: objects, people, expressions, actions, settings, and spatial arrangement.
- **Atomicity:** Each node expresses one atomic piece of information. If a node contains two independently meaningful observations, split it.
- **Relevance:** Include only details that contribute to the meme's argument. Omit decorative backgrounds, watermarks, and irrelevant visual clutter.
- **Standalone language:** Each node must be interpretable in isolation. Do not use pronouns that refer to other nodes (e.g., write "The man on the left is smiling," not "He is smiling").
- **Identity:** If you recognize a public figure, name them (e.g., "Joe Biden"). If you do not recognize them, describe their appearance.

---

## Stage 3: Annotating Intermediate Claims

**Input.** The meme, its conclusion, and the agreed-upon Element set.

**Output.** A set of intermediate Claim nodes that bridge the Elements to the conclusion.

**Instructions.** Build the argument bottom-up: starting from the Element nodes, construct the chain of inferences needed to reach the conclusion. Each intermediate claim should represent one atomic reasoning step — neither trivially restating its premises nor skipping multiple inferential leaps at once.

**Rules.**
- Each Claim must be a full clause with a subject and predicate, written as a direct assertion (not meta-commentary).
- **Atomicity test:** A claim violates atomicity if, to accept it from its premises, a reader must make an unstated inferential jump. For example, given "immigrants are taking jobs" and "jobs are scarce," the claim "immigrants are destroying the economy" skips too many steps. A better intermediate claim would be "job scarcity is worsened by immigration."
- **Triviality test:** A claim is too trivial if it merely rephrases its premises without adding inferential content. For example, "Jobs are being taken by immigrants" adds nothing beyond the premise.
- Claims must not contain unresolved references. Write "Barack Obama was criticized for his healthcare policy," not "He was criticized for this."
- Do not include meta-commentary (e.g., "The meme implies that…") or analytical observations about rhetorical devices (e.g., "The meme uses irony to suggest…"). Only include claims that are part of the argument itself.
- If the argument requires background knowledge to bridge a gap, note this but do not introduce Background nodes yet — those are handled in Stage 5.

---

## Stage 4: Annotating Edges

**Input.** The meme, the conclusion, the Element set, and the intermediate Claim set.

**Output.** A directed graph connecting all nodes with Support or Attack edges.

**Instructions.** Given the agreed-upon set of all nodes (the conclusion, Elements, and intermediate Claims), draw directed edges that represent argumentative relationships. Each edge goes from a child node to a parent node it supports or attacks.

**Rules.**
- **Support edge** (A → B, support): Node A provides evidence, justification, or reasoning that makes B more believable.
- **Attack edge** (A → B, attack): Node A rebuts, undermines, contradicts, or mocks the content of B.
- **Connectivity:** The resulting graph must be weakly connected. Every node must be reachable from the conclusion through undirected traversal. No orphan nodes.
- **Single conclusion:** The conclusion must be the only node with out-degree zero (no outgoing edges).
- **Multi-parent allowed:** A single node may support or attack multiple parent nodes if it genuinely contributes to more than one claim.
- **Direction matters:** Edges are directed from premise to claim, not the reverse. An Element supports or attacks a Claim, not the other way around.
- **Edge type matters:** Do not default to Support for all edges. If the meme uses irony, sarcasm, or juxtaposition to undermine a claim, use Attack.
- If you find that a claim cannot be connected to the rest of the graph without an unstated assumption, note this for Stage 5.

---

## Stage 5: Annotating Background Knowledge

**Input.** The meme and the preliminary graph from Stage 4.

**Output.** The final graph, augmented with Background nodes where needed.

**Instructions.** Examine each argument step in the preliminary graph (i.e., each Claim and its supporting/attacking children). For each step, ask: does a reader need external knowledge — not visible in the meme — to accept that the children establish the parent claim? If so, add a Background node.

**Rules.**
- **The negation test:** A piece of background knowledge is relevant if, in a hypothetical world where the opposite were true, the claim would no longer follow from its premises. For example, if the argument depends on "the KKK is an extremist group associated with violence," then in a world where the KKK were a charitable organization, the claim would fail. This background is therefore relevant and should be added.
- Background nodes must be objective, commonly accepted facts or widely held beliefs. They must not be subjective opinions or contested claims. If a piece of background is itself argumentative, it should be a Claim node instead.
- Each Background node must be a full clause (e.g., "The KKK is an extremist group with a reputation of violence," not just "KKK").
- **Sparingness:** Add background only when it is necessary to bridge a reasoning gap. Do not add obvious commonsense that any reader would know (e.g., "fire is hot," "people have opinions"). The negation test helps calibrate this: if negating the statement does not break the argument, it is not needed.
- Background nodes must be connected to the graph via Support or Attack edges, following the same rules as Stage 4. They typically support a Claim that could not be established from Elements alone.
- After adding Background nodes, verify that the graph remains weakly connected and that the conclusion is still the only node with out-degree zero.
