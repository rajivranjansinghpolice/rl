# Chapter 13 — Policy Gradient Methods: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_13`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_13` — Policy Gradient Methods (REINFORCE `PolicyGradientAlgorithm`,
  REINFORCE with baseline `PolicyGradientWithBaseline`, actor–critic `ActorCriticAlgorithm`,
  continuing actor–critic `ContinuingActorCritic`, Gaussian policies `GaussianPolicy`, simulation
  harness `run_policy_gradient_simulation`).
- **Book chapter:** Chapter 13 — Policy Gradient Methods.
- **Book sections covered by the chapter implementation:**
  - Section 13.1 — Policy approximation and its advantages (Example 13.1).
  - Section 13.2 — The policy gradient theorem (Eq. 13.5).
  - Section 13.3 — REINFORCE: Monte Carlo policy gradient (Eq. 13.6–13.7).
  - Section 13.4 — REINFORCE with baseline.
  - Section 13.5–13.7 — Actor–critic methods; Gaussian policies (continuous actions).
- **Key book equations referenced by the exercises:**
  - Eq. (13.2) — softmax action preferences: \(\pi (a|s,\theta ) = exp(h(s,a,\theta )) / \sum _b exp(h(s,b,\theta ))\).
  - Eq. (13.5) — policy gradient theorem.
  - Eq. (13.6) — REINFORCE update: \(\theta \leftarrow \theta + \alpha \cdot G\cdot \nabla _\theta \pi (A|S,\theta ) / \pi (A|S, \theta t)\).
  - Eq. (13.7) — softmax eligibility: \(\nabla _\theta ln \pi (a|s,\theta ) = x(s,a) - \sum _b \pi (b|s,\theta ) x(s,b)\).
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Chapter 13 content spans
  roughly PDF pages 283–300; the exercises were extracted from PDF pages 284, 290, and 296.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 13. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-13 material, plus the public
  chapter-13 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem**, **Solution**, and — where a companion implementation exists — a
  **coding pointer** into `chapter_13/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations; Programming
  exercises describe an experiment to run.
- The final section maps exercises to the repository code that already implements (or supports) the
  corresponding machinery; exercises that exercise a specific implementation also carry a **Coding
  pointer** inline.

---

## Exercises

### Exercise 13.1 — Optimal probability of "right" in the short-corridor gridworld

- **Book location:** Section 13.1 / Example 13.1, PDF page 284 (Example on PDF page 285).
- **Type:** Computation.
- **Problem:** Example 13.1 (short corridor, three nonterminal states, switched actions in the
  middle state; every state looks identical to the agent: `x(s, right) = [1, 0]ᵀ`,
  `x(s, left) = [0, 1]ᵀ`; reward −1 per step; the ε-greedy right/left policies achieve values of
  about −44 and −82 respectively). Determine the exact symbolic expression for the optimal probability
  `p = P(right)` of selecting the right action in this example.

**Hook.** Here the trap is that all three states look identical to the agent, so one feature vector has
to serve states whose best actions point in opposite directions. Let's turn that limitation into a
single scalar equation and then optimize it.

**Formal bridge.** Because the features are state-independent — \(x(s,\text{right})=[1,0]^\top\)
and \(x(s,\text{left})=[0,1]^\top\) in every state — the best the agent can do is choose one
probability \(p=P(\text{right})\) shared by all three states. Their expected returns satisfy

\[
\begin{aligned}
V_1 &= -1 + p V_2 + (1-p) V_1,\\
V_2 &= -1 + p V_1 + (1-p) V_3,\\
V_3 &= -1 + (1-p) V_2.
\end{aligned}
\tag{13.1.1}
\]

where \(V_1,V_2,V_3\) are the expected returns starting from states 1, 2, 3 (state 1 is the start,
state 3 is next to the terminal cell), each step gives reward \(-1\), and \(p\) is the common
probability of selecting right.

Solving that linear system and then differentiating gives the two expressions the rest of the
reasoning rests on:

\[
V_1(p) = \frac{2(p-2)}{p(1-p)}, \qquad \frac{dV_1}{dp} = \frac{2(p^2-4p+2)}{(p-p^2)^2}.
\tag{13.1.2}
\]

where \(V_1(p)\) is the start-state value as a function of \(p\), and the numerator
\(p^2-4p+2\) decides where that value is stationary.

**Reasoning chain.** Start with the first line of (13.1.1): \(V_1 p = -1 + p V_2\), so
\(V_2 = V_1 + 1/p\). In state 2 the actions are reversed, giving
\(V_2 = -1 + p V_1 + (1-p) V_3\), and in state 3, \(V_3 = -1 + (1-p) V_2\) (right goes to the
terminal with reward \(-1\)). Substitute \(V_3\) into \(V_2\):
\(V_2 = -1 + p V_1 + (1-p)(-1 + (1-p) V_2) = (p-2) + p V_1 + (1-p)^2 V_2\), hence
\(V_2\bigl(1-(1-p)^2\bigr) = p-2 + p V_1\), or \(V_2 = (p V_1 + p - 2)/(p(2-p))\).

Now use \(V_2 = V_1 + 1/p\) a second time: \(V_1 + 1/p = (p V_1 + p - 2)/(p(2-p))\), which
simplifies to \(V_1(p) = (2p-4)/(p(1-p)) = 2(p-2)/(p-p^2)\), the first part of (13.1.2). Every
state is represented by the same feature vector, so function approximation forces a single
stochastic policy across states; this one scalar is the whole design space.

To maximize \(V_1(p)\) — that is, to make the negative return as close to zero as possible —
differentiate as in (13.1.2): \(dV_1/dp = 2[p^2-4p+2]/(p-p^2)^2 = 0\), so
\(p^2 - 4p + 2 = 0\) and \(p = 2 \pm \sqrt 2\). Only \(p^* = 2 - \sqrt 2 \approx 0.586\) is
feasible, since the other root exceeds 1. Substituting back,
\(V_1(p^*) = -2\sqrt 2\,/\,((2-\sqrt 2)(\sqrt 2 - 1)) = -6 - 4\sqrt 2 \approx -11.66\). That is
the best achievable performance: \(p \approx 0.59\), \(v \approx -11.6\), as reported in the book.

**Why it makes sense.** Push \(p\) to the extremes of (13.1.1). With \(p=1\) the agent always goes
right, so in the middle state it walks back to state 1 and bounces forever without reaching the
terminal cell; with \(p=0\) it always goes left and fares no better. Only randomization lets it
escape the trap, because the middle state's actions are reversed: the best single policy leans
slightly toward right (correct in two of the three states) while keeping enough left probability
to get past the reversal. So the optimum is interior near \(0.586\), not at a boundary, and the
value is negative — every escape still costs steps at \(-1\) each.

**Check your understanding.** Prediction: multiply every reward in the corridor by a constant
\(c>0\) (say change \(-1\) per step to \(-2\)). Every \(V_i\) scales by \(c\), but the numerator
in (13.1.2) does not change, so the maximizing \(p^* = 2-\sqrt 2\) is unchanged; only the value
\(-6-4\sqrt 2\) scales.

### Exercise 13.2 — Prove Eq. (13.7) (softmax eligibility vector)

- **Book location:** Section 13.3, PDF page 289–290.
- **Type:** Math / Proof.
- **Problem:** Prove Eq. (13.7) — \(\nabla _\theta ln \pi (a|s,\theta ) = x(s,a) - \sum _b \pi (b|s,\theta ) \cdot x(s,b)\) — using the
  definitions and elementary calculus (exponential softmax policies with linear action preferences).

**Hook.** The identity in (13.7) looks like a trick, but it is just the log-derivative of a ratio.
We will take the logarithm first, then differentiate one term at a time.

**Formal bridge.** Start from the exponential softmax policy with linear action preferences,

\[
\pi (a|s,\theta ) = \frac{\exp(h(s,a,\theta ))}{\sum _b \exp(h(s,b,\theta ))}, \qquad h(s,a,\theta ) = \theta^T x(s,a).
\tag{13.2.1}
\]

The claim to prove is the centered gradient

\[
\nabla_\theta \ln \pi (a|s,\theta ) = x(s,a) - \sum_b \pi (b|s,\theta )\, x(s,b).
\tag{13.2.2}
\]

where \(s\) is the state, \(a\) and \(b\) are actions, \(x(s,a)\) is the feature vector of action
\(a\) in state \(s\), \(\theta\) is the policy parameter, \(h(s,a,\theta )\) is that action's
preference, and the sum over \(b\) runs over all actions available in \(s\).

**Reasoning chain.** Take logs of (13.2.1). The numerator contributes its exponent, and the
denominator contributes the log of the normalizing sum:

\(\ln \pi (a|s,\theta ) = \theta^T x(s,a) - \ln \sum_b \exp(\theta^T x(s,b))\).

Now differentiate with respect to \(\theta\). The first term is linear, so it gives exactly
\(x(s,a)\). The second requires the chain rule: writing \(Z=\sum_b \exp(\theta^T x(s,b))\),
\(\nabla_\theta \ln Z = \frac{1}{Z}\sum_b \exp(\theta^T x(s,b))\, x(s,b)\). The exponential
factors divided by \(Z\) are precisely the softmax probabilities \(\pi (b|s,\theta )\) from
(13.2.1), so this term is \(\sum_b \pi (b|s,\theta ) x(s,b)\).

Subtract the two pieces and (13.2.2) follows: \(\nabla_\theta \ln \pi (a|s,\theta ) =
x(s,a) - \sum_b \pi (b|s,\theta ) x(s,b)\). The gradient is the feature of the action actually
taken minus the expected feature under the current policy — a *centered* score, sometimes called
the eligibility vector in the tabular softmax case.

**Why it makes sense.** Consider the two extremes. If action \(a\) is nearly certain,
\(\pi (a|s,\theta )\approx 1\), then the average feature in (13.2.2) is almost \(x(s,a)\) and the
gradient nearly vanishes: a confident policy barely changes when \(\theta\) moves. If all actions
are equally likely, the subtraction leaves the chosen feature minus the uniform mean, the largest
possible signal. The sign also has to be this way: when \(x(s,a)\) points above the policy's
average feature, the gradient points along \(x(s,a)\), so increasing \(\theta\) in that direction
raises \(a\)'s preference — exactly what a policy gradient should do after a good outcome.

**Check your understanding.** Prediction: add the same constant vector \(c\) to every feature,
\(x(s,a)\mapsto x(s,a)+c\). The gradient in (13.2.2) is unchanged, because \(c\) appears once with
a plus sign and once inside the average and cancels; the softmax preferences are unchanged too.
Scaling all features by \(k\), by contrast, scales the gradient by \(k\).

### Exercise 13.3 — Bernoulli-logistic unit

- **Book location:** Section 13.3 / 13.5, PDF page 296.
- **Type:** Math / Computation.
- **Problem (paraphrased):** A Bernoulli-logistic unit is a stochastic, neuron-like unit. Its input at time `t` is
  a feature vector `x(S_t)`; its output `A_t` is random, with `Pr{A_t = 1} = P_t` and `Pr{A_t = 0} =
  1 − P_t`. Let `h(s, 0, θ)` and `h(s, 1, θ)` be the preferences for the unit's two actions given
  policy parameter \(\theta\); assume the preference difference is a weighted sum of the input:
  \(h(s, 1, \theta ) - h(s, 0, \theta ) = \theta ^T x(s)\).
  (a) Show that if the exponential softmax distribution (13.2) is used to convert preferences into
  policies, then \(P_t = \pi (1|S_t, \theta _t) = 1 / (1 + exp(-\theta _t^T x(S_t)))\) (the logistic function).
  (b) What is the Monte-Carlo REINFORCE update of \(\theta _t\) to \(\theta _{t+1}\) upon receipt of return `G_t`?
  (c) Express the eligibility \(\nabla _\theta ln \pi (a|s,\theta )\) for a Bernoulli-logistic unit in terms of `a`,
  `x(s)`, and \(\pi (a|s,\theta )\) by calculating the gradient (separately for each action, then combined).

**Hook.** A Bernoulli-logistic unit is just a two-action softmax in disguise. Once we see that, the
logistic probability and its REINFORCE gradient fall out of the same algebra we used in 13.2.

**Formal bridge.** With only two actions, the softmax depends on the preference difference
\(z=\theta^T x(s) = h(s,1,\theta)-h(s,0,\theta)\), so the probability of output 1 is

\[
P_t = \pi (1|S_t,\theta_t) = \frac{\exp(h_1)}{\exp(h_0)+\exp(h_1)} = \frac{1}{1+\exp(-z)} = \sigma(z).
\tag{13.3.1}
\]

The REINFORCE update and the corresponding eligibility are

\[
\theta_{t+1} = \theta_t + \alpha\, G_t\, (A_t-P_t)\, x(S_t), \qquad
\nabla_\theta \ln \pi (a|s,\theta ) = \bigl(a - \pi (1|s,\theta )\bigr)\, x(s).
\tag{13.3.2}
\]

where \(x(s)\) is the input feature vector, \(\theta\) the parameter, \(z\) the preference gap,
\(\sigma(\cdot)\) the logistic function, \(\pi (a|s,\theta )\) the probability of action
\(a\in\{0,1\}\), \(A_t\) the sampled output, \(P_t=\Pr\{A_t=1\}\), \(G_t\) the received return, and
\(\alpha\) the step size.

**Reasoning chain.** For part (a), divide numerator and denominator of the softmax by
\(\exp(h_0)\): \(P = \exp(h_1)/(\exp(h_0)+\exp(h_1)) = 1/(1+\exp(h_0-h_1)) =
1/(1+\exp(-z))\), because \(z = \theta^T x(s) = h_1-h_0\). That is exactly the logistic function
\(\sigma(z)\).

For part (b), plug the eligibility into the general Monte-Carlo REINFORCE rule
\(\theta \leftarrow \theta + \alpha\, G_t \nabla_\theta \ln \pi (A_t|S_t,\theta )\). With
(13.3.2) this is \(\theta \leftarrow \theta + \alpha\, G_t (A_t-P_t) x(S_t)\). The boxed
REINFORCE algorithm discounts the return, giving the variant
\(\theta \leftarrow \theta + \alpha\, \gamma^t G_t (A_t-P_t) x(S_t)\).

For part (c), compute each case directly using \(\nabla_\theta P = P(1-P)x(s)\). If \(a=1\):
\(\nabla_\theta \ln P = (1/P)\nabla_\theta P = (1/P)P(1-P)x(s) = (1-P)x(s)\). If \(a=0\):
\(\nabla_\theta \ln(1-P) = (1/(1-P))(-P(1-P))x(s) = -P x(s)\). Combined, for \(a\in\{0,1\}\),
\(\nabla_\theta \ln \pi (a|s,\theta ) = (a-\pi (1|s,\theta ))x(s)\). It is positive on a "1"
sample and negative on a "0" sample — the same response-driven form as Eq. 15.3 in Chapter 15.

**Why it makes sense.** Look at the two confidence regimes in (13.3.2). If \(z\) is large and
positive, \(P\approx 1\), so a "1" sample gives almost no gradient: a nearly certain unit has
little left to learn. Near \(P=0.5\) the same sample gives its biggest step, \(0.5\,x(s)\). The sign
is forced: a rewarded "1" must push \(\theta\) along \(x(s)\) to raise \(P(1)\), while a rewarded
"0" pushes the other way. If the sign were flipped, a successful output would make itself less
likely and the unit would learn against its own reward.

**Check your understanding.** Prediction: add the same constant to both preferences \(h_0\) and
\(h_1\). The gap \(z\) is unchanged, so \(P_t\), the eligibility, and the update are all unchanged.
If instead \(x(s)\) is scaled by \(k\), then \(z\) and the gradient both scale by \(k\), and the
unit becomes more confident for the same input direction.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 13.1 | §13.1 (Example 13.1, short corridor) | `ShortCorridorEnv`, `ShortCorridorREINFORCE`, `run_short_corridor_reinforce`, `run_short_corridor_value_ascent`, `short_corridor_value` (closed form), `short_corridor_value_gradient` (analytic dV1/dp), `short_corridor_optimal_p` | `TestShortCorridor` (transitions, sigmoid policy, closed-form p*/v*, curve max, MC cross-check, REINFORCE validity), `TestValueAscent` (gradient sign, convergence to p*) | `python -m pytest chapter_13/tests/ -v` |
| 13.2 | §13.3 (Eq. 13.7, softmax eligibility) | `chapter_13/solution.py` uses the softmax score form \(onehot(a) - \pi (\cdot |s)\) in tabular updates (supporting usage, not an implementation of the Eq. 13.7 proof) | `TestPolicyGradientTheorem` (numeric policy-gradient-theorem check, not the Eq. 13.7 derivation) | `python -m pytest chapter_13/tests/ -v` |
| 13.3 | §13.3 / §13.5 (Bernoulli-logistic unit) | *(no direct implementation in this repo; repo policies are tabular softmax / Gaussian, not a sigmoid Bernoulli-logistic unit)* | *(none)* | — |

- Full chapter command: `python -m pytest chapter_13/tests/ -v`
- Lint: `ruff check chapter_13/`
- No standalone `exercise_*.py` file exists for this chapter; the policy-gradient machinery lives in
  `solution.py` and the tests group it by algorithm. Note: 13.2 remains supported conceptually by the
  tabular softmax score form but its proof is not directly tested. 13.3 remains derivation-only.
- 13.1 is now implemented in `solution.py`: the short-corridor MDP (`ShortCorridorEnv`), the closed-form
  start-state value \(V_1(p) = 2(p-2)/(p(1-p))\), the analytic gradient \(dV_1/dp = 2(p^2-4p+2)/(p(1-p))^2\),
  and exact value-ascent (`run_short_corridor_value_ascent`) which converges deterministically to p*. The
  empirical REINFORCE path (`run_short_corridor_reinforce`) is also included but is high-variance with
  all-negative rewards and may not reach p* in few episodes.

---

## Verification and Quality Notes

- The worked arithmetic in this document (e.g., Ex. 13.1: \(p* = 2 - \sqrt 2 \approx 0.586\), \(V(p*) \approx -11.66\))
  was re-derived by hand; the public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`)
  was used only as a cross-check.
- The short-corridor MDP in Ex. 13.1 is reconstructed from the book's description (Example 13.1,
  PDF pages 284–285); the derived optimal probability \(p* \approx 0.586\) matches the book's reported
  "about 0.59" and optimal value "about −11.6".

---

*Exercises document for Chapter 13 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
