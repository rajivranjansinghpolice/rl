# Chapter 17 — Frontiers: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_17`
> Book chapter: **Chapter 17** — Frontiers

*Book contains no formal numbered exercises for this chapter.* The prompts below are derived from
the frontier sections, labeled **[derived]**.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_17` — Frontiers (four-room `GridWorld`, hierarchical options
  `Option` / `default_options` / `run_option`, SMDP Q-learning `smdp_q_learning`, primitive
  Q-learning `primitive_q_learning`, and intrinsic novelty `novelty_bonus` / `q_learning_novelty`).
- **Book chapter:** Chapter 17 — Frontiers (where the field is going next).
- **Book sections covered:**
  - §17.1 General Value Functions and Auxiliary Tasks
  - §17.2 Temporal Abstraction via Options
  - §17.3 Observations and State
  - §17.4 Designing Reward Signals
  - §17.5 Remaining Issues
  - §17.6 Reinforcement Learning and the Future of Artificial Intelligence
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction, approximate):** §17.1 p. 397;
  §17.2 p. 398; §17.3 p. 401; §17.4 p. 404; §17.5 p. 406; §17.6 p. 408.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 17. Section statements/prompts below are paraphrased/derived (not verbatim book exercise
  text); short references only.
- **Source and scope:** Sutton & Barto (2nd ed.) Chapter 17; exercise statements are paraphrased
  and derivations are worked from the book's equations, with section/equation/page citations for
  traceability.

---

## Derived Exercises

### Derived 1 — General Value Functions (GVFs) and auxiliary tasks [derived]

- **Book location:** Section 17.1, PDF p. ~397.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Define a GVF-style auxiliary prediction (e.g. predicting a future signal or
  feature outcome under a behavior) and write its TD update; explain how auxiliary predictions can
  improve the representation/features used for a main task.

**Hook.** An agent can learn about its world even when the extrinsic reward is silent. Auxiliary
predictions turn other signals into learning tasks that shape the features the main task uses.

**Formal bridge.** A General Value Function predicts a discounted sum of a chosen cumulant under a
policy with a state-dependent continuation signal, and it can be learned by TD:

\[
v_{\pi,\gamma,c}(s)=\mathbb{E}\!\left[\sum_{k=0}^{\infty}\Bigl(\prod_{i=1}^{k}\gamma_{t+i}\Bigr)c_{t+k+1}\,\middle|\,S_t=s,\pi\right],
\tag{17.1.1}
\]

\[
\delta_t = c_{t+1} + \gamma_{t+1}\hat v(S_{t+1},w) - \hat v(S_t,w), \qquad
w \leftarrow w + \alpha\,\delta_t\,\nabla_w \hat v(S_t,w).
\tag{17.1.2}
\]

where \(c_{t+1}\) is the cumulant being predicted (not necessarily extrinsic reward), \(\pi\) is the
target or behavior policy, \(\gamma_t\in[0,1]\) is the continuation probability, \(\hat v(s,w)\) is
the learned predictor with weights \(w\), \(\delta_t\) is its TD error, and \(\alpha\) is the step
size.

**Reasoning chain.** A **General Value Function** predicts the expected discounted sum of a
specified **cumulant** \(c_{t+1}\) under a target/behavior policy \(\pi\), with a continuation signal
\(\gamma_t\in[0,1]\). The ordinary value function is the special case where the cumulant is the
reward. For off-policy auxiliary prediction, importance sampling or other corrections may be
needed; the simplest on-policy TD(0) update is the one in (17.1.2).

Now consider what the agent must represent to make those predictions. Auxiliary predictions — next
features, sensors, termination signals, and so on — force the agent to learn state features that
capture the environment's dynamics and structure. They therefore provide a richer representation
and more learning signal even when extrinsic reward is sparse, because every auxiliary cumulant
supplies its own error to train the shared features.

**Why it makes sense.** To predict many different future signals, a representation has to retain
information about all of them; a representation tuned only to a sparse reward can safely ignore
most of the environment. The gradient in (17.1.2) pushes the shared features toward whatever
matters for each auxiliary task, so the main task inherits a better basis. If the cumulant carried
no information at all, the auxiliary task would be harmless but useless — the improvement comes
from structure in the world, not from the extra update by itself.

**Check your understanding.** Prediction: set \(c_{t+1}=R_{t+1}\) and \(\gamma_t=\gamma\). The GVF
(17.1.1) reduces to the ordinary value function \(v_\pi\). If instead \(c\) is a sensor signal
unrelated to reward, the auxiliary prediction still shapes the features but contributes no reward
signal directly.

### Derived 2 — SMDP Q-learning [derived]

- **Book location:** Section 17.2, PDF p. ~398.
- **Type:** Math / Programming.
- **Problem (paraphrased):** Derive the SMDP Q-learning update for an option
  \(o = (I, \pi , \beta )\): \(Q(s,o) \leftarrow Q(s,o) + \alpha [r + \gamma ^k max_{o'} Q(s',o') - Q(s,o)]\). Contrast it with
  ordinary one-step Q-learning and explain the variable-duration return.

**Hook.** An option is an action that lasts several steps, so its return has to be aggregated over
that duration. The SMDP update is ordinary Q-learning with the bookkeeping adjusted for \(k\).

**Formal bridge.** An option running for \(k\) primitive steps produces an accumulated discounted
reward and a terminal state, giving the SMDP Bellman form and update

\[
r = \sum_{j=0}^{k-1}\gamma^j R_{t+1+j}, \qquad
Q^*(s,o)=\mathbb{E}\!\left[r + \gamma^k \max_{o'} Q^*(s',o')\,\middle|\,s,o\right],
\tag{17.2.1}
\]

\[
Q(s,o) \leftarrow Q(s,o) + \alpha\left[r + \gamma^k \max_{o'} Q(s',o') - Q(s,o)\right].
\tag{17.2.2}
\]

where \(o=(I,\pi,\beta)\) is the option, \(k\) is its duration in primitive steps, \(s'=s_{t+k}\)
is the state where it terminates, \(r\) is the discounted sum of rewards collected inside it,
\(\gamma^k\) discounts the future across that duration, and \(\alpha\) is the step size.

**Reasoning chain.** When option \(o\) is initiated in \(s_t\in I\) and executes for \(k\) primitive
steps until termination at \(s_{t+k}\), it produces the cumulative discounted reward in (17.2.1)
and the next decision state \(s'=s_{t+k}\). Treating the option as a temporally extended action in a
Semi-Markov Decision Process gives the SMDP Bellman optimality form in (17.2.1), and replacing the
expectation by a sample gives the update in (17.2.2).

The contrast with ordinary Q-learning is now exact: it is the special case \(k=1\), where a
primitive action lasts one step and \(r=R_{t+1}\), so \(\gamma^k=\gamma\). The option return discounts
across the option's duration and bootstraps only when the option terminates, enabling multi-step
temporal abstraction and fewer high-level decisions.

**Why it makes sense.** An option that lasts many steps is not equivalent to one primitive step:
rewards accrue inside it, and the next decision happens \(k\) steps later, so the continuation must
be discounted by \(\gamma^k\) to line up with the time at which the option ends. If we used plain
\(\gamma\), long options would be credited as if their future began immediately and would look too
valuable — the agent would prefer slow options for the wrong reason.

**Check your understanding.** Prediction: double an option's duration \(k\) while keeping the same
per-step reward. The internal sum \(r\) grows but \(\gamma^k\) shrinks, so the option's value changes
with both, and the number of high-level decisions halves. With \(\gamma=1\) the discount term no
longer penalizes the delay, so the value depends mainly on accumulated reward.

### Derived 3 — Options as temporal abstraction that accelerates learning [derived]

- **Book location:** Section 17.2, PDF p. ~398.
- **Type:** Theory / Programming.
- **Problem (paraphrased):** (a) Give the formal option definition \((I, \pi , \beta )\). (b) Explain (and relate
  to the repo demo) how subgoal options can reduce decisions and speed learning versus primitives.

**Hook.** Reusable skills let an agent choose a whole behavior at once instead of replanning every
primitive step. We need the formal definition to see exactly what an option commits to.

**Formal bridge.** A Markov option is a triple

\[
o = (I,\pi,\beta), \qquad I\subseteq S,\quad \pi(a|s)\ \text{the intra-option policy},\quad \beta(s)\in[0,1]\ \text{the termination probability}.
\tag{17.3.1}
\]

where \(I\) is the set of states in which the option may be initiated, \(\pi(a|s)\) gives the action
probabilities while it runs, and \(\beta(s)\) is the probability that it stops in state \(s\).

**Reasoning chain.** For part (a), (17.3.1) is the definition: \(o=(I,\pi,\beta)\), with \(I\subseteq S\) the
initiation set, \(\pi(a|s)\) the intra-option policy, and \(\beta(s)\in[0,1]\) the termination
condition. While the option is executing, actions follow \(\pi\), and the option stops in state \(s\)
with probability \(\beta(s)\).

For part (b), subgoal options such as "go to hallway" or "go to door" package useful multi-step
behaviors. In the repo's four-room gridworld, SMDP Q-learning over primitives plus subgoal options
(`smdp_q_learning`) can reach the goal with far fewer **macro decisions** than primitive-only
control — see `TestOptions::test_options_solve_in_few_macro_decisions` and
`test_options_agent_vs_primitive_only_agent`. The reason is that high-level choices jump between
meaningful subgoals instead of replanning at every primitive step.

**Why it makes sense.** A macro decision selects an entire policy segment, so if a subgoal routine
already solves the hallway, the agent does not need to relearn each step toward it. Fewer decisions
mean a smaller search over what to do next and shorter credit-assignment paths, which is why
learning speeds up. The tradeoff is that badly designed options can be the wrong skills and hurt
performance — the compression helps only when the packaged behavior is genuinely reusable.

**Check your understanding.** Prediction: set \(\beta(s)=1\) everywhere so options terminate
immediately. Options collapse into primitives and the macro-decision advantage disappears. Make
them far too coarse instead, and they overshoot their subgoals, degrading performance.

### Derived 4 — Observations and state [derived]

- **Book location:** Section 17.3, PDF p. ~401.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Explain why observations are not states in POMDPs, derive the belief-state
  update (Bayes filter), and discuss memory/RNNs as state estimators.

**Hook.** When the sensors hide part of the world, the latest observation is not enough to decide
what to do. The belief over hidden states is the compact summary that restores the Markov property.

**Formal bridge.** After taking action \(a_t\) and seeing observation \(o_{t+1}\), the Bayes filter
updates the belief by likelihood times prior prediction, normalized:

\[
b_{t+1}(s') = \eta\, O(o_{t+1}\mid s',a_t)\sum_s P(s'\mid s,a_t)\, b_t(s), \qquad
\eta = \frac{1}{P(o_{t+1}\mid b_t,a_t)}.
\tag{17.4.1}
\]

where \(b_t(s)=P(s_t=s\mid h_t)\) is the belief over the latent state given history \(h_t\),
\(O(o_{t+1}\mid s',a_t)\) is the observation model, \(P(s'\mid s,a_t)\) the transition model, and
\(\eta\) normalizes the result so that \(\sum_{s'}b_{t+1}(s')=1\).

**Reasoning chain.** In a **POMDP** the agent receives an observation \(o_t\) drawn from
\(O(o_t\mid s_t,a_{t-1})\), while the latent environment state \(s_t\) is not fully revealed —
different states can emit similar observations. Consequently \(o_t\) is not Markov in general:
the same observation can call for different actions depending on what came before.

A sufficient statistic is the belief \(b_t(s)=P(s_t=s\mid h_t)\) given the history \(h_t\). Bayes'
rule gives (17.4.1): multiply the predicted state distribution
\(\sum_s P(s'|s,a_t)b_t(s)\) by the likelihood of the new observation in each state \(s'\), then
normalize by \(\eta=1/P(o_{t+1}|b_t,a_t)\).

Because optimal control can depend on the full history, practical agents use **memory** — recurrent
policies/RNNs or explicit belief trackers — to approximate the latent state from sequences of
observations. The belief is the normative version of what such memory is trying to represent.

**Why it makes sense.** Imagine two hidden states that produce identical observations but require
opposite actions. Conditioning on the observation alone cannot distinguish them, so the history
must be consulted; the belief is exactly the compressed history relevant to the hidden state. The
normalizer \(\eta\) only rescales the numbers to sum to one — it does not change which state is
most likely, so it cannot change the decision. If observations were fully revealing, the belief
would collapse to a point mass and memory would add nothing.

**Check your understanding.** Prediction: make \(O\) deterministic and one-to-one so the observation
reveals the state. The belief becomes a point mass and the POMDP reduces to an MDP; introduce more
ambiguous observations and the belief spreads out, raising the value of memory/RNNs.

### Derived 5 — Designing reward signals [derived]

- **Book location:** Section 17.4, PDF p. ~404.
- **Type:** Theory.
- **Problem (paraphrased):** Discuss sparse vs dense rewards, derive potential-based shaping
  \(F(s,a,s') = \gamma Φ(s') - Φ(s)\) and show it preserves optimal policies, and note pitfalls of ad hoc
  shaping.

**Hook.** Dense rewards speed up learning but are easy to get wrong. Potential-based shaping is the
special construction that guides learning without changing what is optimal.

**Formal bridge.** Augment the reward with a difference of a potential function,

\[
F(s,a,s') = \gamma\Phi(s')-\Phi(s), \qquad R' = R + F,
\tag{17.5.1}
\]

and over a trajectory the shaped contributions telescope:

\[
\sum_t \gamma^t F(s_t,a_t,s_{t+1}) = -\Phi(s_0) + \lim_{T\to\infty}\gamma^T\Phi(s_T).
\tag{17.5.2}
\]

where \(\Phi\) is the potential assigned to states, \(\gamma\) the discount, \(s_0\) the initial
state, and the right-hand side has no dependence on the actions chosen along the way in standard
discounted or episodic settings.

**Reasoning chain.** **Sparse rewards** are easy to specify but provide little learning signal;
**dense rewards** can accelerate learning but may encode the designer's biases and create
unintended incentives. Potential-based shaping augments the reward with \(F\) from (17.5.1) for a
potential \(\Phi\).

For any policy, the shaped return telescopes, as (17.5.2) shows: the sum collapses to a term
depending on the start state plus a terminal limit, giving a constant/terminal-dependent offset
independent of action choices in standard discounted/episodic settings. Therefore action-value
advantages and optimal policies are preserved: \(Q'^*(s,a)\) differs from \(Q^*(s,a)\) by a
state-dependent term that does not change \(\arg\max_a\).

The **pitfall** is that non-potential shaping, or poorly chosen dense rewards more generally, can
change the optimal policy and produce reward hacking. The preservation argument depends exactly on
the difference form in (17.5.1).

**Why it makes sense.** Think of \(\Phi\) as altitude: the shaping reward pays you for descending,
but you must have climbed earlier to get the chance. Around a full trajectory the payments cancel,
so the agent cannot gain by looping — it is guided locally without the goal being rewritten. That
is why \(\arg\max_a\) is unchanged: the leftover offset in (17.5.2) is the same for every action.
If \(F\) depended on the action outside this difference structure, the offset would differ by
action and the optimal policy could genuinely change.

**Check your understanding.** Prediction: set \(\Phi\) to a constant. Then \(F=0\) and shaping has
no effect at all. Make \(\Phi\) depend directly on the action instead of the state, and the
telescoping argument fails — the optimal policy can change.

### Derived 6 — Intrinsic motivation and novelty bonuses [derived]

- **Book location:** Sections 17.1, 17.4, PDF pp. ~397–404.
- **Type:** Theory / Math.
- **Problem (paraphrased):** Define a novelty/intrinsic bonus (count- or prediction-error-based), show how
  it encourages visiting rare states, write the combined reward, and contrast with Dyna exploration
  bonuses (§8.3/8.4).

**Hook.** When rewards are sparse, the agent needs a reason to go somewhere new. A novelty bonus
makes uncertainty itself worth pursuing, but it is not the same mechanism as a model-based bonus.

**Formal bridge.** A count-based bonus decreases with visits and is added to the extrinsic reward,

\[
b(s)=\frac{\beta}{\sqrt{N(s)+1}}, \qquad R'_t = R_t^{\text{ext}} + b_t^{\text{int}}, \qquad
\pi^*_{\text{mod}}=\arg\max_\pi \mathbb{E}_\pi\!\left[\sum_t \gamma^t\bigl(R_t^{\text{ext}}+b_t^{\text{int}}\bigr)\right].
\tag{17.6.1}
\]

where \(N(s)\) is the visit count of state \(s\), \(\beta>0\) scales the intrinsic reward,
\(R_t^{\text{ext}}\) is the environmental reward, \(b_t^{\text{int}}\) is the novelty bonus, and
\(\pi^*_{\text{mod}}\) is the policy optimal for the combined reward.

**Reasoning chain.** A simple **count-based** novelty bonus is \(b(s)=\beta/\sqrt{N(s)+1}\), as in
`novelty_bonus`, or another decreasing function of the visit count \(N(s)\), such as one proportional
to \(1/N(s)\). Rarely visited states therefore receive larger intrinsic reward. The agent optimizes
the modified reward \(R'_t=R_t^{\text{ext}}+b_t^{\text{int}}\); in the tabular Q update this appears
as \(q \leftarrow q+\alpha[(R_{\text{ext}}+b)+\gamma\max_{a'}q(s',a')-q]\). The effective optimal
policy in (17.6.1) trades task return against exploration.

**Contrast with Dyna bonuses (§8.3/8.4):** Dyna-Q+/model-based bonuses such as \(\kappa\sqrt{\tau}\)
are typically used in the **model/planning** loop to encourage trying actions or state–action pairs
whose outcomes have not been revisited recently, which is useful under nonstationarity. Intrinsic
novelty bonuses instead augment experienced rewards directly, driving visitation and coverage
during real interaction.

**Why it makes sense.** Dividing by \(\sqrt{N+1}\) makes a first visit worth much more than a tenth
visit, so the agent is pulled toward uncovered states; once a state is well sampled the bonus fades
and the agent returns to optimizing extrinsic reward. The direction is essential: an inverted bonus
that grew with \(N\) would make the agent camp in familiar territory, the opposite of exploration.
The Dyna bonus differs because it is about stale *model* knowledge, not about how often a state has
been physically visited.

**Check your understanding.** Prediction: increase \(\beta\) in (17.6.1). Exploration dominates for
longer, coverage improves, and the final policy may sacrifice some extrinsic return. Set \(\beta=0\)
and the agent reverts to ordinary task-only RL, so rare states may never be visited.

### Derived 7 — Reinforcement learning and the future of AI [derived]

- **Book location:** Sections 17.5–17.6, PDF pp. ~406–410.
- **Type:** Theory.
- **Problem (paraphrased):** Discuss frontiers including (a) open-world/continual learning, (b) inductive
  bias and symmetry, (c) abstraction/unification, and (d) scaling; give one research direction each for
  representation, credit assignment, exploration, and scaling.

**Hook.** The frontier is about generality: methods that work in a fixed, small benchmark may fail in
an open world that keeps changing. A single objective statement shows what has to give.

**Formal bridge.** The standard objective assumes a fixed distribution, while continual learning
must handle dynamics that shift over time:

\[
J(\pi)=\mathbb{E}_{\pi,\,p_t}\!\left[\sum_t \gamma^t R_{t+1}\right], \qquad \hat v(s,w)\ \text{or}\ \phi(s)\ \text{as the learned representation}.
\tag{17.7.1}
\]

where \(\pi\) is the policy, \(p_t(s',r\mid s,a)\) is a potentially nonstationary transition/reward
distribution, \(R_{t+1}\) the reward, \(\gamma\) the discount, and \(\phi(s)\) or \(\hat v(s,w)\) the
state representation the agent must learn and reuse.

**Reasoning chain.** (a) **Open-world/continual learning:** agents must handle nonstationarity, new
objects, actions, and goals over long lifetimes without catastrophic forgetting — that is, \(p_t\)
in (17.7.1) changes while \(\pi\) must keep adapting.

(b) **Inductive bias and symmetry:** priors such as equivariance, compositionality, and physics can
improve sample efficiency and generalization, because they constrain which representations \(\phi\)
are worth trying.

(c) **Abstraction:** learning reusable temporal and state abstractions — options, GVFs, hierarchies —
supports transfer and planning.

(d) **Scaling:** larger models, data, and compute interact with stability, alignment, and efficiency
constraints.

Example research directions, one per area: **representation** — learn disentangled/compositional
state abstractions and GVF-based features; **credit assignment** — better long-horizon/causal
credit assignment beyond local TD/eligibility limits; **exploration** — principled intrinsic
motivation and information-seeking in vast state spaces; **scaling** — efficient, stable training
and deployment of RL agents that generalize and adapt continually.

**Why it makes sense.** Current benchmarks are stationary and fully specified, so the hard part is
solving a fixed MDP well. The real world changes and is enormous, so the hard part becomes
generalization and continual adaptation. Priors and symmetry reduce the samples needed to learn a
usable representation, abstraction compresses the state so planning stays tractable, and without
those, adding compute alone cannot cover an open world.

**Check your understanding.** Prediction: make the environment stationary and small. Inductive
priors and continual-learning machinery add little, because plain RL already works. Introduce
nonstationarity and a fixed policy degrades, so continual adaptation and reusable abstractions
become essential.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 1 | §17.1 | *Theory only in this repo chapter (no GVF/auxiliary-task implementation exposed)* | — | — |
| Derived 2–3 | §17.2 | `chapter_17/solution.py` (`GridWorld`, `Option`, `default_options`, `make_subgoal_option`, `run_option`, `smdp_q_learning`, `greedy_smdp_action`, `primitive_q_learning`) | `TestGridWorld`, `TestOptions` | `python -m pytest chapter_17/tests/test_solution.py::TestGridWorld chapter_17/tests/test_solution.py::TestOptions -v` |
| Derived 4–5, 7 | §17.3–17.6 | *Theory/design discussion only in this repo chapter* | — | — |
| Derived 6 | §17.1, §17.4 | `chapter_17/solution.py` (`novelty_bonus`, `q_learning_novelty`) | `TestNoveltyBonus` | `python -m pytest chapter_17/tests/test_solution.py::TestNoveltyBonus -v` |

- Full chapter command: `python -m pytest chapter_17/tests/ -v`
- Lint: `ruff check chapter_17/`

---

## What to take away

- A GVF generalizes the value function by swapping the reward for any cumulant and the discount for
  a continuation signal; ordinary \(v_\pi\) is just the case \(c=R,\ \gamma_t=\gamma\). Auxiliary
  predictions matter because every cumulant supplies an error that trains the shared features, so
  the agent can learn the structure of its world while extrinsic reward is silent.
- An option is a temporally extended action \((I,\pi,\beta)\), and SMDP Q-learning is ordinary
  Q-learning with duration bookkeeping: accumulate the reward over \(k\) steps and discount the
  continuation by \(\gamma^k\). Keeping plain \(\gamma\) instead makes slow options look
  artificially valuable; subgoal options speed learning only when the packaged skill is genuinely
  reusable.
- Observations are not states. When different hidden states emit the same observation, the belief
  \(b_t(s)=P(s_t=s\mid h_t)\) is the sufficient statistic that restores the Markov property, and
  memory or RNNs approximate it. Make observations one-to-one and the belief collapses to a point
  mass: the POMDP becomes an MDP.
- Potential-based shaping guides without rewriting the goal: \(\gamma\Phi(s')-\Phi(s)\) telescopes
  into start and end terms that do not depend on the action, so \(\arg\max_a\) is preserved. A
  count-based novelty bonus is different in kind — it rewards physical coverage — while Dyna's
  \(\kappa\sqrt{\tau}\) bonus targets stale *model* knowledge.
- The frontier question is generality, not benchmark skill: under shifting \(p_t\) a fixed policy
  degrades, so reusable abstraction, inductive biases, and continual adaptation matter more than
  extra compute on a stationary problem. This chapter carries the MDP/DP foundations of Chapters
  3–4 and the function approximation of Chapter 12 into open-world territory.

**Where the theory meets the code.** Derived 4 uses the standard Bayes-filter identity
\(b'(s')=\eta O(o_{t+1}\mid s',a_t)\sum_s P(s'\mid s,a_t)b_t(s)\). In this repo only the
options/SMDP and count-based novelty prompts have small-gridworld demonstrations; GVFs,
belief-state/RNN estimators, and potential-based shaping are treated theoretically.

---

*Chapter 17 (Frontiers) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
