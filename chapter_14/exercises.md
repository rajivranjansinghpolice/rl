# Chapter 14 — Psychology: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_14`
> Book chapter: **Chapter 14** — Psychology

*Book contains no formal numbered exercises for this chapter.* The prompts below are derived from
the sections themselves and are labeled **[derived]**.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_14` — Psychology (`RescorlaWagnerModel`, `TDConditioningModel`,
  blocking experiments, excitation curves).
- **Book chapter:** Chapter 14 — Psychology (RL as a model of animal behavior and learning).
- **Book sections covered:**
  - §14.1 Prediction and Control; §14.2 Classical Conditioning; §14.3 Instrumental Conditioning;
    §14.4 Delayed Reinforcement; §14.5 Cognitive Maps; §14.6 Habitual and Goal-directed Behavior;
    §14.7 Summary.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** §14.2 starts on PDF page 303;
  §14.3 on PDF page 315; §14.4 on PDF page 319; §14.5 on PDF page 320; §14.6 on PDF page 321;
  §14.7 (Summary) on PDF page 324.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 14. Section statements paraphrased; short verbatim references only.
- **Source and scope:** Sutton & Barto (2nd ed.) Chapter 14; exercise statements are paraphrased
  and derivations are worked from the book's equations, with section/equation/page citations for
  traceability.

---

## Derived Exercises

### Derived 1 — Rescorla–Wagner rule and blocking [derived]

- **Book location:** Section 14.2, PDF page 303 (classical conditioning).
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive the Rescorla–Wagner update \(ΔV_i = \alpha _i \beta (\lambda - \sum _j V_j)\) for
  stimulus `i`; use it to predict Kamin's blocking experiment: (a) Phase 1: CS_A alone paired with the
  US until fully conditioned; (b) Phase 2: compound CS_A + CS_B paired with the same US. Show
  `V(CS_B)` remains near 0.

**Hook.** Blocking is the classic demonstration that learning is driven by surprise, not by mere
contact between a cue and an outcome. Let's write the update rule that makes that prediction and
then run the two phases through it.

**Formal bridge.** On a compound trial, a cue only gets a weight change in proportion to the
error that is left over after all present cues have made their predictions:

\[
V_{AX} = \sum_j x_j V_j, \qquad \Delta V_i = \alpha_i \beta\,(\lambda - V_{AX}), \qquad \delta = \lambda - V_{AX}.
\tag{14.1.1}
\]

where \(V_i\) is the associative strength of cue \(i\), \(x_i\in\{0,1\}\) says whether cue \(i\) is
present, \(V_{AX}\) is the total predicted strength on the trial, \(\lambda\) is the asymptote of
US support (e.g. 1), \(\alpha_i\) is cue \(i\)'s learning rate, \(\beta\) scales the US impact, and
\(\delta\) is the prediction error that drives all learning.

**Reasoning chain.** On a compound trial, (14.1.1) gives the update
\(V_i \leftarrow V_i + \alpha_i\beta(\lambda - V_{AX})\), with \(V_{AX}=\sum_j x_j V_j\) and
\(\lambda\) the US asymptote (for example 1). Notice that the *same* error \(\delta=\lambda-V_{AX}\)
is applied to every cue present on the trial; cues do not get separate errors.

In phase 1 (A→US) only cue A is present, so its update is
\(V_A \leftarrow V_A + \alpha_A\beta(\lambda - V_A)\) and \(V_A\to\lambda\), while \(V_B=0\)
because B has never appeared. Each trial shrinks \(\lambda-V_A\), so after enough pairings A alone
accounts for the entire outcome.

In phase 2 (AX→US), \(V_{AX}=V_A+V_B\approx\lambda+0\). Hence \(\delta=\lambda-V_{AX}\approx 0\)
and \(\Delta V_B = \alpha_B\beta\delta\approx 0\): B stays near zero even though it was reliably
paired with the US. A already fully predicts the US, so no error remains to drive learning about
B — this is the blocking effect.

**Why it makes sense.** Ask what happens if phase 1 is cut short, before \(V_A\) reaches
\(\lambda\). Then a residual error survives into phase 2, and B acquires some strength — the
partial-blocking result. That direction is exactly right: if learning is driven by unexplained
outcomes, a cue can only capture the part of the outcome that the existing cue failed to explain.
If the rule instead used reward magnitude \(\lambda\) directly, B would always be learned and
blocking would be impossible.

**Check your understanding.** Prediction: give A many more phase-1 pairings so \(V_A\) sits even
closer to \(\lambda\). The phase-2 error shrinks further and B is blocked even more completely.
But if the phase-2 US is made larger than the phase-1 US, \(\delta\) becomes positive again and B
is no longer blocked — there is new surprise to explain.

### Derived 2 — TD model of classical conditioning [derived]

- **Book location:** Section 14.2, PDF page 303.
- **Type:** Math.
- **Problem (paraphrased):** Write the TD(0) update for a conditioning trial and show that (a) a
  presence trial (CS followed by US) increases `V(CS)`, (b) an extinction trial decreases `V(CS)`
  toward 0, and (c) conditioning reaches a nonzero fixed point when the US is reliably predicted.

**Hook.** One TD error has to explain acquisition, extinction, and second-order conditioning. The
trick is that the error is a *difference*, so its sign flips when the prediction changes.

**Formal bridge.** On each step the critic forms a target from the reward actually received plus
its bootstrapped estimate of the next state, and moves its current estimate toward that target:

\[
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t), \qquad V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t.
\tag{14.2.1}
\]

where \(V(S_t)\) is the predicted value of the current state, \(R_{t+1}\) is the reward (US)
received on the next step, \(\gamma\) discounts the bootstrapped next-state value, and \(\alpha\)
is the step size; \(\delta_t\) is the TD prediction error.

**Reasoning chain.** Look at (14.2.1) on a presence trial: at the CS state preceding the US we have \(R_{t+1}>0\),
or \(V(S_{t+1})\) already reflects the upcoming US. Initially \(V(S_t)\) is small, so
\(\delta_t>0\) and the update increases \(V(\text{CS})\). That is acquisition: the cue acquires
value because the outcome outstrips its current prediction.

For an extinction trial there is no US, so \(R_{t+1}=0\) and the bootstrapped target falls. Once
\(V(\text{CS})\) overpredicts what actually follows, \(\delta_t<0\), and the same update drives
\(V(\text{CS})\to 0\). Extinction is therefore not a separate mechanism; it is the same equation
running with the opposite sign of error.

The fixed point occurs when \(E[\delta_t\mid S_t]=0\), i.e.
\(V(S_t)=E[R_{t+1}+\gamma V(S_{t+1})]\). For a reliably predicted US this yields \(V(\text{CS})>0\):
the CS acquires predictive value equal to the discounted/expected outcome. Acquisition,
extinction, and second-order conditioning all fall out of this one rule, because a cue that
reliably precedes another predictor inherits value through the bootstrapped term.

**Why it makes sense.** Start from \(V(\text{CS})=0\): any upcoming reward is pure surprise, so the
error is positive and the value grows. It stops growing exactly when the prediction matches the
outcome — the error self-extinguishes. When the US is removed, the previously learned prediction
is now too high, so the error reverses sign and unwinds the learning. If the error used only
\(R_{t+1}\) and ignored \(V(S_{t+1})\), there would be no second-order conditioning, because a cue
that merely predicts another cue would receive no reward of its own.

**Check your understanding.** Prediction: increase \(\alpha\). Both acquisition and extinction
proceed faster, but individual updates become noisier. Reduce \(\gamma\) toward zero and the value
at the CS shrinks toward immediate reward only, so second-order conditioning fades because the
bootstrapped future is discounted away.

### Derived 3 — Instrumental vs classical conditioning [derived]

- **Book location:** Section 14.3, PDF page 315.
- **Type:** Theory.
- **Problem (paraphrased):** Contrast classical conditioning with instrumental conditioning and explain
  why Q-learning/Sarsa-style TD control is instrumental rather than classical.

**Hook.** The distinction is not about which algorithm you run; it is about whether the animal's own
choice changes the outcome. We can see that directly by comparing a value prediction with an
action-value maximization.

**Formal bridge.** Prediction-only learning estimates the value of a state under a policy,

\[
V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\,\middle|\,S_t=s\right],
\tag{14.3.1}
\]

whereas control learns and improves action values,

\[
Q^*(s,a) = \mathbb{E}\!\left[R_{t+1} + \gamma\max_{a'} Q^*(S_{t+1},a')\,\middle|\,s,a\right], \qquad V^*(s)=\max_a Q^*(s,a).
\tag{14.3.2}
\]

where \(s\) is the state, \(a\) the action, \(\pi\) the behaviour policy being evaluated,
\(R_{t+1}\) the reward, \(\gamma\) the discount, and \(Q^*(s,a)\) the optimal value of taking
\(a\) in \(s\). The \(\max_a\) in (14.3.2) is the mathematical trace of instrumental control.

**Reasoning chain.** Classical conditioning is stimulus–outcome prediction: a CS comes to predict a
US regardless of the animal's behavior. It is Pavlovian, and it is modeled by value prediction —
Rescorla–Wagner or TD estimation of a quantity like \(V^\pi\) in (14.3.1). There is no action in the
equation, because the outcome does not depend on what the animal does.

Instrumental (operant) conditioning is action–outcome learning: the organism's behavior changes
the probability of reinforcement, which is Thorndike's law of effect. TD control methods such as
Sarsa and Q-learning learn action values \(Q(s,a)\) and select actions to maximize return, as in
(14.3.2). Because the policy and the choice affect which outcomes occur, these methods model
instrumental learning. Prediction-only TD, estimating \(V\) without selecting among actions,
corresponds to classical conditioning.

**Why it makes sense.** Suppose reward delivery were completely independent of the animal's
actions. Then every \(Q(s,a)\) in (14.3.2) would converge to the same number and the \(\max_a\)
would be vacuous — keeping an action index would add nothing. The moment outcomes depend on the
choice, the action values separate and maximizing them becomes meaningful. That is why the
presence of choice, not the presence of a reward, is what makes a learning problem instrumental.

**Check your understanding.** Prediction: make the US independent of the animal's response while
keeping the CS–US pairing. Q-learning's action values collapse toward equality and behavior looks
Pavlovian; the classical CS–US association, however, is unaffected because it never depended on
the response.

### Derived 4 — Delayed reinforcement [derived]

- **Book location:** Section 14.4, PDF page 319.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive how eligibility traces bridge the delay between an action/state and
  a later outcome; explain the role of \(\lambda\).

**Hook.** A reward that arrives many steps late would normally train only the most recent state.
Eligibility traces keep a fading record of where the agent has been so the error can travel
backward in time.

**Formal bridge.** With linear value approximation \(\hat v(s,w)=w^T x(s)\), an accumulating
trace and the TD(λ) weight update are

\[
e_t = \gamma\lambda\, e_{t-1} + x(S_t), \qquad
\delta_t = R_{t+1}+\gamma \hat v(S_{t+1},w)-\hat v(S_t,w), \qquad
w \leftarrow w + \alpha\,\delta_t\, e_t.
\tag{14.4.1}
\]

where \(x(S_t)\) is the feature vector of the current state, \(w\) the value weights, \(e_t\) the
eligibility trace, \(\lambda\in[0,1]\) its decay rate, \(\delta_t\) the TD error, \(\gamma\) the
discount, and \(\alpha\) the step size; in the general nonlinear case \(x(S_t)\) is replaced by
\(\nabla_w \hat v(S_t,w)\).

**Reasoning chain.** With \(\hat v(s,w)=w^T x(s)\), the first line of (14.4.1) says the accumulating trace obeys
\(e_t=\gamma\lambda e_{t-1}+x(S_t)\), and the TD(λ) update is \(w\leftarrow w+\alpha\delta_t e_t\)
with \(\delta_t=R_{t+1}+\gamma\hat v(S_{t+1})-\hat v(S_t)\). Unrolling the recursion, a state
visited \(k\) steps ago still contributes \((\gamma\lambda)^k x(S_t)\) to the current trace.

So when a delayed reward arrives at time \(t+k\), \(e_{t+k}\) still contains that decayed trace
of earlier states, and the TD error is applied to all of them in proportion to their recency.
That is how credit is assigned backward across the delay without waiting for a full episode.

The parameter \(\lambda\) sets the time horizon of that backward reach. Larger \(\lambda\) (closer
to 1) keeps traces alive longer, bridging longer delays but with higher variance, because more
tentative state–outcome pairings receive credit. Smaller \(\lambda\) emphasizes recent,
bootstrapped credit and behaves more like one-step TD.

**Why it makes sense.** At \(\lambda=0\) the trace is just \(x(S_t)\), so only the state immediately
before the error is updated — no delayed reinforcement at all. At \(\lambda\to 1\) the trace decays
only by \(\gamma\), so credit flows far back toward distant causes, approaching Monte Carlo
attribution and paying for it with variance. The interpolation is natural: \(\gamma\lambda\) is the
factor by which a memory survives one step, so a delay of \(k\) steps costs a factor
\((\gamma\lambda)^k\). If that factor were larger for longer delays, unrelated early states would
get the most credit, which is backwards.

**Check your understanding.** Prediction: double the delay \(k\) between action and reward. The
credit reaching the original state shrinks by another factor \(\gamma\lambda\), so learning about
it weakens. Raising \(\lambda\) toward 1 compensates by keeping the trace alive longer, at the cost
of noisier updates.

### Derived 5 — Cognitive maps as learned models [derived]

- **Book location:** Section 14.5, PDF page 320.
- **Type:** Theory.
- **Problem (paraphrased):** Explain the difference between a latent world model and a cache of
  conditioned responses, relate cognitive maps to model-based RL, and mention place-cell evidence.

**Hook.** The same animal can follow a habit and take a brand-new shortcut. Explaining both requires
two kinds of knowledge: a cache of what worked before, and a model of how the world is arranged.

**Formal bridge.** A cached controller stores value directly, while a model-based controller
recomputes it from learned transitions,

\[
Q_{\text{cache}}(s,a) \approx \text{stored from experience}, \qquad
Q_{\text{model}}(s,a) = \sum_{s'} p(s'\mid s,a)\bigl[r(s,a,s')+\gamma V(s')\bigr].
\tag{14.5.1}
\]

where \(s,a\) are state and action, \(p(s'\mid s,a)\) is the learned transition model,
\(r(s,a,s')\) the resulting reward, \(\gamma\) the discount, and \(V(s')\) the value used for
planning; the cache has no explicit \(p\) to consult.

**Reasoning chain.** Equation (14.5.1) makes the contrast concrete. A cache of conditioned responses stores direct stimulus→response/value
associations learned from experience — this is model-free. A latent world model, or cognitive map,
represents the environment's structure and transitions, which allows planning and detour or
shortcut behavior without direct retraining. Tolman's maze results and hippocampal place cells
(spatial firing fields) support the existence of map-like representations.

In RL terms, cognitive maps correspond to model-based methods: learn a transition model and plan
with it, as in Dyna. Cached responses correspond to model-free value or policy learning. The two
can coexist, and the observable difference is what happens when the environment changes: the
model can be queried for a new route, while the cache must be updated by fresh experience.

**Why it makes sense.** Imagine a barrier moved in a familiar maze. A cached response keeps sending
the animal into the old route until it is retrained; a map lets it recompute a detour immediately
because the layout is explicitly represented. That is why map-like evidence appears in shortcut
and detour tests rather than in simple repeated-response tasks — those tasks never require using
the representation for anything but the learned response.

**Check your understanding.** Prediction: keep the animal on one fixed, unchanging route with
reliable reinforcement. The map confers no visible advantage and habitual control dominates.
Introduce a novel shortcut and the map-based agent takes it at once, while the cache-based agent
must first discover it by trial and error.

### Derived 6 — Habitual vs goal-directed behavior [derived]

- **Book location:** Section 14.6, PDF page 321.
- **Type:** Theory / Math.
- **Problem (paraphrased):** Contrast habits (model-free) with goal-directed behavior (model-based) and
  explain the reinforcer-devaluation test.

**Hook.** Devaluation is a clever experiment because it changes the *value of the outcome* without
changing any stimulus–response association. Watching whether behavior changes tells us which
system is in control.

**Formal bridge.** With a model, action value is recomputed from the current outcome value;
without one, it is a cached number:

\[
Q_{\text{goal}}(s,a)=\sum_{s'}p(s'\mid s,a)\bigl[r(s,a,s')+\gamma\max_{a'}Q(s',a')\bigr], \qquad
Q_{\text{habit}}(s,a)\leftarrow Q_{\text{habit}}(s,a)+\alpha\bigl[R+\gamma\max_{a'}Q_{\text{habit}}(S',a')-Q_{\text{habit}}(s,a)\bigr].
\tag{14.6.1}
\]

where \(p\) is the learned transition model and \(r\) the outcome reward that devaluation changes;
the cached update on the right has no separate representation of \(r\) to revise.

**Reasoning chain.** Goal-directed behavior is sensitive to the current value of outcomes and to
action contingencies — it corresponds to model-based evaluation and planning, like the first
expression in (14.6.1). Habitual behavior is triggered by stimuli via cached action tendencies —
model-free control, like the second expression.

In a reinforcer-devaluation test the outcome is devalued (by satiation or aversion) without any
further training. A goal-directed agent reduces the action because it re-evaluates the now-devalued
outcome through the model; a habitual agent keeps performing it, because the cached tendency was
never tied to the current outcome value. Overtraining tends to shift control from model-based to
model-free systems, consistent with dual-process accounts and with Dyna-like planning versus
direct TD control in RL.

**Why it makes sense.** Devaluation changes \(r\) in (14.6.1), not the stimulus–response mapping.
A model explicitly stores the contingency between action and outcome, so it has somewhere to put
the new information. A cache has compressed that contingency away into a number; there is nothing
for devaluation to act on. Hence sensitivity to devaluation is evidence that the outcome is still
being consulted, and insensitivity is evidence that control has migrated to the habit system.

**Check your understanding.** Prediction: partially devalue the outcome — reduce \(r\) only a
little. Goal-directed responding declines roughly in proportion to the change, while habitual
responding stays flat. After extensive overtraining the gap widens, because more control has moved
into the cache.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 1 | §14.2 | `chapter_14/solution.py` (`RescorlaWagnerModel`, `run_trials`, `excitation_curve`) | `TestRescorlaWagner` | `python -m pytest chapter_14/tests/ -v` |
| Derived 2 | §14.2 | `chapter_14/solution.py` (`TDConditioningModel`, `presence_trial`, `csc_trial`, `run_blocking_experiment`, `train_presence`) | `TestTDConditioningModel`, `TestBlockingExperiment` | `python -m pytest chapter_14/tests/ -v` |
| Derived 3–6 | §14.3–14.6 | *Theory only in this repo chapter (classical-conditioning models implemented above; no instrumental control, Dyna/maps, or devaluation routines)* | — | — |

- Full chapter command: `python -m pytest chapter_14/tests/ -v`
- Lint: `ruff check chapter_14/`

- Full chapter command: `python -m pytest chapter_14/tests/ -v`
- Lint: `ruff check chapter_14/`

---

## What to take away

- Every effect here is driven by a prediction error, not by raw reward magnitude. The
  Rescorla–Wagner \(\delta\) explains blocking: a cue that already predicts the US leaves no
  surprise for a new cue to capture, so \(V_B\) stays near zero even though B was reliably paired
  with the outcome. Cutting phase 1 short leaves residual error and gives partial blocking.
- Acquisition and extinction are one TD update with the error's sign reversed. Value rises while
  the outcome exceeds the prediction, then unwinds when the prediction overshoots; the fixed point
  is where \(E[\delta]=0\). If the teaching signal were \(\lambda\) alone, blocking would be
  impossible, and if the bootstrapped term were dropped, so would second-order conditioning.
- Delayed reinforcement is a credit-assignment problem. The accumulating trace
  \(e_t=\gamma\lambda e_{t-1}+x(S_t)\) keeps earlier states eligible so a later error can reach
  them; \(\lambda\) interpolates from one-step bootstrapping at \(0\) to nearly Monte Carlo
  attribution near \(1\), buying reach at the cost of variance.
- Habit versus goal-directed behavior is model-free versus model-based. A cached value cannot
  respond to devaluation or a moved barrier, while a learned transition model can recompute a
  route or an outcome value on the spot; devaluation sensitivity and shortcut tests reveal which
  system is in control, and overtraining migrates control toward the cache. Chapter 15 retells
  this same divide in neural terms.

**Where the theory meets the code.** The prompts in this chapter are derived from the sections
rather than numbered book exercises, and only the conditioning models have a direct implementation
here; instrumental control, cognitive maps, and devaluation remain theoretical. The claims are
qualitative predictions of behavior — concentration-style bounds play no role in them.

---

*Chapter 14 (Psychology) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
