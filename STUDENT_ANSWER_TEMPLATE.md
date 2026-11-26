# Student-Friendly Answer Template

This is the house style for every worked answer in `chapter_XX/exercises.md`. It
keeps two voices in one place: a **conversational** voice that carries the
reasoning, and a **formal** voice that pins down the mathematics. The goal is an
answer you can read once for understanding and reread once for the equation.

> **Design:** "Textbook-Bridge, spoken conversationally." Start where a student
> is, end at the book's notation. No pilot, no exceptions — every exercise
> answer in the guide uses this shape.

---

## The five parts of an answer

Every answer is a `### Exercise X.Y — Title` section with these five parts, in
this order. Keep the book-location / type / problem metadata the chapter already
uses, then write the answer in these parts.

### 1. Conversational hook

One or two sentences that say **what we are about to do and why it is worth
doing**. Speak directly to the reader ("Let's ...", "Here the trap is ...").
Do not announce the answer before earning it, and do not use filler such as
"this answer was written from scratch" or "the test suite verifies this".

### 2. Formal bridge

State the one equation the reasoning rests on, in the standard notation below.
Every displayed equation gets:

- a short lead-in sentence that says what the symbols mean,
- an in-words **symbol glossary** listing each symbol and its meaning,
- a stable number of the form `\tag{X.Y.n}` — chapter `X`, exercise `Y`,
  equation `n` within that answer. Numbering is **per exercise**, so adding a
  new exercise never renumbers an existing equation. Derived (non-book-numbered)
  exercises use `\tag{X.Dn.n}` — chapter `X`, derived exercise `n`, equation
  `n` within that answer.

Use an inline equation when the whole expression fits in a sentence; use a
display equation when the answer's logic turns on its shape.

### 3. Reasoning chain

Walk through the idea in conversation, referring back to the tagged equation as
the anchor ("that \(\kappa\sqrt{\tau}\) term in (8.2.1) ..."). This is where the
answer earns its length: name the intermediate steps, say what changes at each
step, and connect cause to effect. Prefer two or three short paragraphs over one
dense bullet list.

### 4. Why it makes sense

An intuition check, not a restatement. Relate the result to a limiting case, an
extreme parameter value, or a picture the reader already knows. If a sign or a
direction could plausibly be the opposite, say why it is this way and not the
other way.

### 5. Check your understanding

Ask the reader to **predict** what changes under a small perturbation, then give
the prediction immediately. This is a self-check, not a quiz: there is no
"answer me" and no grading. One or two sentences are enough.

> **Good:** "Prediction: if we double \(\kappa\), the bonus grows everywhere, so
> the agent re-tries stale actions sooner — and once every pair is fresh, the
> bonus fades and the two agents behave alike."
>
> **Bad:** "Quiz: what happens when \(\kappa\) is doubled? See Section 8.3."

---

## Equation formatting standard

- Inline math: `\(...\)`. Display math: `\[...\]`. Never raw `$...$` for math.
  A literal dollar amount such as `$10` in the car-rental problem is currency,
  not math, and stays as-is.
- Number every displayed equation with `\tag{X.Y.n}` on the same line as the
  expression. Refer to it in prose as `(X.Y.n)`.
- Define symbols in words immediately after the equation, in a compact line:
  `where \(r\) is the recorded reward, \(\tau(s,a)\) is the number of steps
  since \((s,a)\) was last tried, and \(\kappa>0\) is a small bonus weight.`
- Use the same symbol for the same concept everywhere in the chapter: \(s\)
  state, \(a\) action, \(r\) reward, \(G_t\) return, \(\gamma\) discount,
  \(\pi\) policy, \(v\) state value, \(q\) action value, \(\alpha\) step size,
  \(\tau\) staleness. Do not switch a symbol mid-chapter to be "closer" to a
  source.
- Keep equations close to the reasoning they support. An equation with no
  follow-up sentence is a formatting error, not a style choice.

## What to remove

- Self-referential process notes: "written from scratch", "cross-checked against
  public repos", "verified by pytest", "the test suite covers this".
- One-line "Explanation (student-friendly)." tags followed by a generic
  summary. The reasoning chain and the intuition check replace them.
- Verification tables that list test class names and pytest commands inside
  an answer. A short **Where this shows up in the code** pointer naming the
  API is welcome; a command listing is not.

---

## Fully worked example

### Exercise 1.4 — Learning from every move, or only from the moves taken?

- **Book location:** Section 1.5, Exercise 1.4.
- **Type:** Theory / Computation.

**Hook.** Suppose we keep exploring forever, but tighten the step size so the
estimates settle down. The question is not *whether* they settle — it is *what*
they settle to. The answer is decided entirely by which moves we update.

**Formal bridge.** If we back up every move we consider, including exploratory
ones, the estimates approach the optimal state values

\[
v_*(s) = \max_{a} \mathbb{E}\!\left[G_t \mid S_t = s, A_t = a\right].
\tag{1.4.1}
\]

If we update only the moves the behaviour policy actually executed, they
approach instead the values of that exploratory policy

\[
v_{\pi}(s) = \mathbb{E}\!\left[G_t \mid S_t = s, \pi\right].
\tag{1.4.2}
\]

where \(s\) is the current state, \(a\) is a candidate action, \(G_t\) is the
total discounted return from time \(t\), \(\pi\) is the behaviour policy that
keeps exploring, and the maximum in (1.4.1) is over all actions available in
\(s\).

**Reasoning chain.** Notice the only structural difference between
(1.4.1) and (1.4.2): one has a \(\max_a\), the other has an expectation over
\(\pi\). Updating every considered move performs that maximisation implicitly —
each action gets its own backup, and the best one wins as the estimates
converge. Updating only executed moves never asks "what if we had chosen
differently?", so the numbers converge to the return of the policy that did the
choosing, exploration and all.

This matters because exploration does not stop. Under the second scheme the
converged values still describe a wandering policy. Acting greedily from them
then throws away the very exploration that produced them. Under the first
scheme the greedy action is already the one the values were built around, so
the same greedy tie-break keeps paying off.

**Why it makes sense.** Think of the extreme: let exploration be uniform and
permanent. A value learned only from executed moves averages over all those
random choices, so it can never exceed the average of the available returns.
A value learned from all moves keeps the best return separately, untouched by
the randomness. The gap between (1.4.2) and (1.4.1) is exactly the gap between
"what a random player gets" and "what a player who picks the best move gets" —
it cannot vanish while exploration continues.

**Check your understanding.** Prediction: shrink \(\epsilon\) toward zero while
keeping the updates on executed moves. The behaviour policy becomes nearly
greedy, so (1.4.2) moves toward (1.4.1) — but the convergence is slower, because
rarely tried actions are now almost never updated. If instead we keep \(\epsilon\)
large and switch to all-move updates, the values stay optimal and only the data
collection is noisy.

---

## Code standard for `chapter_XX/solution.py`

The same house style applies to implementations:

- **One algorithm = one readable function.** Each public algorithm is an
  orchestrator; coherent phases (action selection, model update, planning
  backup, statistics recording) live in small private helpers with descriptive
  names and type hints.
- **Comments explain why loops exist**, e.g. "replay the model so value news
  spreads without new environment steps" — not what each line does.
- **Named constants replace magic numbers** that encode task settings, defined
  near use with the same value they had before. A safety cap is labelled as a
  cap, not smuggled in as an algorithm parameter.
- **No semantic changes.** Public names, signatures, defaults, return shapes,
  numeric values, tie-breaking, and the order and count of random draws are
  preserved. Every chapter's tests and the full suite must stay green.

*House style for `chapter_XX/exercises.md` and `chapter_XX/solution.py`; linked
from the repository README. Kept at the repository root because the public
snapshot ignores `docs/`.*
