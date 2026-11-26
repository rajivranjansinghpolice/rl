# Chapter 2 — Multi-Armed Bandits: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_02`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_02` — Multi-Armed Bandits (Algorithms 2.1–2.5: bandit simulation
  environment, action-value estimation / sample averages, ε-greedy selection, UCB selection, gradient
  bandit preferences; plus simulations for non-stationary and optimistic-initial-value settings, and
  the exploration strategies folded in from the old repository Chapter 11: UCB action selection,
  optimism in the face of uncertainty, and the descending bonus).
- **Book chapter:** Chapter 2 — Multi-Armed Bandits.
- **Book sections covered by the chapter implementation:**
  - Section 2.1 — The k-armed bandit problem.
  - Section 2.2 — Action values and the sample-average method (estimator algorithms).
  - Section 2.3 — ε-greedy action selection.
  - Section 2.4 — Incremental implementation of sample averages (and non-stationary step sizes);
    Section 2.5 — Tracking a non-stationary problem.
  - Section 2.6 — Optimistic initial values; Section 2.7 — Upper-confidence-bound action selection;
    Section 2.8 — Gradient bandit algorithms; Section 2.9 — Associative search / contextual bandits.
- **Key book equations referenced by the exercises:**
  - Eq. (2.1) — sample average: \(Q_t(a) = (\sum _{i=1}^{t} R_i \cdot I_i(a)) / (\sum _{i=1}^{t} I_i(a))\) with
    `I_i(a) = 1/n` for the sample-average case.
  - Eq. (2.2) — greedy: `A_t = argmax_a Q_t(a)`.
  - Eq. (2.8) — UCB1: \(A_t = argmax_a [Q_t(a) + c\cdot \sqrt (ln(t) / N_t(a))]\).
  - Eq. (2.10) — gradient bandit preference updates.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** Chapter 2 content spans
  roughly PDF pages 30–52; the exercises were extracted from PDF pages 39, 41, 44, 45, 47, 50, and 52.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 2. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`,
  `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-2 material, plus the public
  chapter-2 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem**, **Solution**, and — where a companion implementation exists — a
  **coding-exercise pointer** into `chapter_02/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations; Programming
  exercises describe an experiment to run.
- The final section maps exercises to the repository code that already implements (or supports) the
  corresponding machinery; exercises that exercise a specific implementation also carry a
  **Coding pointer** inline.

---

## Exercises

### Exercise 2.1 — Probability of selecting the greedy action under ε-greedy

- **Book location:** Section 2.2, PDF page 39.
- **Type:** Computation.
- **Problem:** In ε-greedy action selection, for the case of two actions and ε = 0.5, what is the
  probability that the greedy action is selected?
**Hook.** ε-greedy has two separate routes to the same action. Let's count both routes, because the
random route can land on the greedy action too — an easy term to forget.

**Formal bridge.** The greedy action is chosen either by the greedy branch or by the random branch
hitting it by chance:

\[
P(\text{greedy}) = (1-\epsilon)\cdot 1 + \epsilon\cdot\frac{1}{k}.
\tag{2.1.1}
\]

where \(\epsilon\) is the probability of taking the random branch, \(k\) is the number of actions,
and the \(1/k\) term is the chance that a uniformly random pick is the greedy action.

**Reasoning chain.** Start with the first branch. With probability \(1-\epsilon\) the rule is
greedy, and then the greedy action is selected with certainty — that contributes
\((1-\epsilon)\cdot 1\). With probability \(\epsilon\) we ignore the estimates and draw uniformly
among all \(k\) actions. That draw is not guaranteed to be non-greedy; it picks the greedy action
with probability \(1/k\), contributing \(\epsilon/k\).

For \(k=2\) and \(\epsilon=0.5\) the two contributions are \(0.5\) and \(0.5\times 0.5=0.25\), so
(2.1.1) gives \(1-0.5+0.25=0.75\). In other words, three times out of four the greedy action is
played, even though half the time the rule "decided" to explore. The original student-friendly
summary has the direction backwards: small \(\epsilon\) means mostly exploitation, large
\(\epsilon\) means more exploration.

**Why it makes sense.** Check the extremes of (2.1.1). At \(\epsilon=0\) we get \(1\): always
greedy. At \(\epsilon=1\) we get \(1/k\): pure uniform random play, so with two actions the greedy
one appears half the time, not zero. The answer must lie between those extremes, and \(0.75\) does.
If the formula lacked the \(\epsilon/k\) term, the \(\epsilon=1\) case would incorrectly give zero,
which would mean a random player never selects the best action — clearly impossible.

**Check your understanding.** Prediction: raise \(\epsilon\) from \(0.5\) to \(0.9\) while keeping
\(k=2\). Equation (2.1.1) gives \(0.1+0.45=0.55\), so the greedy action is played much less often;
with many actions (large \(k\)), the same \(\epsilon\) hurts exploitation more because the random
branch is unlikely to stumble onto the best arm.

### Exercise 2.2 — Which steps involved the ε-case?

- **Book location:** Sections 2.2–2.3, PDF page 39.
- **Type:** Computation.
- **Problem:** k = 4 bandit, ε-greedy, sample-average estimates, initial `Q(a) = 0` for all actions.
  Observed sequence: `A1 = 1, R1 = 1; A2 = 2, R2 = 1; A3 = 2, R3 = 2; A4 = 2, R4 = 2; A5 = 3, R5 = 0`.
  On which time steps did the ε-case definitely occur? On which could it possibly have occurred?
**Hook.** Here we play detective: from a short log of actions and rewards, infer when the random
branch of ε-greedy fired. The trick is to reconstruct the estimates step by step.

**Formal bridge.** Each action's sample average updates only when that action is chosen,

\[
Q_{n+1}(a) = Q_n(a) + \frac{1}{n}\left(R_{n+1} - Q_n(a)\right),
\tag{2.2.1}
\]

and the greedy branch selects \(\arg\max_a Q_t(a)\), so any taken action that is not a maximiser
must have come from the ε branch.

where \(a\) is an action, \(n\) is the number of times \(a\) has been taken so far, \(Q_n(a)\) is
its current sample-average estimate, and \(R_{n+1}\) is the reward received on that play.

**Reasoning chain.** Begin with every \(Q(a)=0\). On step 1 all estimates tie, so action 1 is an
arbitrary pick among maximisers — no exploration is forced. Its reward \(R_1=1\) gives
\(Q(1)=1\).

Step 2: \(Q(1)=1 > Q(2)=0\), so the greedy choice is action 1, but action 2 was taken. A greedy
branch could not have done that, so the ε-case definitely occurred. Action 2's reward gives
\(Q(2)=1\), creating a tie with action 1.

Step 3: with \(Q(1)=Q(2)=1\), action 2 is itself a maximiser. The pick may have been greedy (tie
broken toward action 2) or it may have come from the ε branch, which can also land on action 2;
this step is ambiguous. Its reward \(R_3=2\) moves \(Q(2)\) to \((1+2)/2=1.5\).

Step 4: \(Q(2)=1.5 > Q(1)=1\), and action 2 was taken. The action agrees with the greedy choice,
so there is no evidence of an ε-case here — the log is consistent with no exploration, which is
how the original classification treats this step. Strictly, an ε draw could also have landed on
action 2, so agreement cannot prove the ε branch was absent; only disagreement proves it was
present. The reward leaves \(Q(2)=(1+2+2)/3=2\).

Step 5: \(Q(2)=2 > Q(3)=0\), yet action 3 was taken. That is another definite ε-case, followed by
\(R_5=0\). So the definite ε steps are 2 and 5; step 3 could have been one; step 4 shows none.

**Why it makes sense.** Think of the anchor in (2.2.1): the only way to see a non-greedy action is
for the ε branch to fire, because the greedy branch is deterministic given the estimates. That
makes non-greedy moves perfect evidence — a one-way detector. The reverse is not true: greedy-looking
moves are weak evidence, because a random draw can coincide with the best action (recall Exercise
2.1, where that happens 25% of the time here). This asymmetry is why steps 2 and 5 are certain but
step 4 cannot be certified.

**Check your understanding.** Prediction: if the same log had action 1 taken on step 2 instead,
step 2 would no longer be a definite ε-case because action 1 was greedy there. If all estimates
were tied at every step (all rewards equal), no step could ever be classified as definitely
exploratory, since every action would be a maximiser.

### Exercise 2.3 — Best method in the 10-armed testbed (Figure 2.2)

- **Book location:** Section 2.3, PDF page 41.
- **Type:** Theory / Computation.
- **Problem:** In the comparison shown in Figure 2.2, which method will perform best in the long run in
  terms of cumulative reward and probability of selecting the best action? How much better? Express
  your answer quantitatively.
**Hook.** A learning curve is two stories at once: how much reward was collected, and how often the
best arm was chosen. Let's read both from the same quantity so the comparison is quantitative rather
than a list of impressions.

**Formal bridge.** At step \(t\), expected reward and best-action probability are the same average
under different weights:

\[
\mathbb{E}[R_t] = \sum_{a} \pi_t(a)\, q(a),
\qquad
P(\text{best at }t) = \pi_t(a_*),
\tag{2.3.1}
\]

and the cumulative comparison over a horizon \(T\) is the area under that curve,
\(\sum_{t=1}^{T}\mathbb{E}[R_t]\).

where \(\pi_t(a)\) is the probability that the method selects action \(a\) at step \(t\), \(q(a)\) is
the true expected reward of \(a\), \(a_*\) is the best action (\(\arg\max_a q(a)\)), and \(T=1000\)
is the horizon of Figure 2.2.

**Reasoning chain.** Equation (2.3.1) shows exactly why exploration strategy decides the result.
The greedy method has \(\pi_t(a)\) concentrated on whatever currently looks best. If the first
sample of the true best arm is unlucky, greedy never returns to it, so \(\pi_t(a_*)\) stays low and
the average reward stays well below optimal for the whole 1000 steps — it plateaus instead of
learning. With 2000 independent tasks averaged, this is a systematic effect, not noise.

ε-greedy with moderate \(\epsilon\) keeps a floor of probability on every action, so eventually
the optimal arm is sampled, its estimate rises, and \(\pi_t(a_*)\) climbs toward \(1-\epsilon
+\epsilon/k\). Once the optimum is found, the method recovers quickly and its reward approaches the
optimal level, though it never quite reaches it while \(\epsilon>0\).

UCB uses a deterministic bonus instead of random exploration: actions with high uncertainty are
preferred until the evidence is sufficient. In the Figure 2.2 comparison it attains a high
probability of selecting the best action and the best asymptotic performance among the compared
methods. Quantitatively, ranking the methods by the average reward over the 1000-step horizon —
the area under the learning curve — gives greedy well below optimal, moderate-ε greedy near
optimal, and UCB best of the group. The exact gap depends on the sampled true values, so the
figure, not a single closed-form number, is the quantitative statement.

**Why it makes sense.** Imagine the extreme where a task's best arm is never sampled early. A
greedy method can then never revise that arm's estimate, so the deficit in (2.3.1) persists for
all \(T\) steps and the area under its curve can only get worse as \(T\) grows. Any method with a
nonzero exploration mechanism eventually samples it and removes the deficit. The ordering cannot
be reversed in the long run: exploration costs something early, but it is the only way to make
\(\pi_t(a_*)\) large, and large \(\pi_t(a_*)\) is what average reward rewards.

**Check your understanding.** Prediction: extend the horizon from 1000 to 10,000 steps. Greedy's
average reward per step stays roughly at its plateau, while ε-greedy and UCB improve further, so
the cumulative gap widens; if exploration is removed from those methods after convergence, their
best-action probability stops improving as well.

### Exercise 2.4 — General step-size weighting

- **Book location:** Section 2.4, PDF page 44.
- **Type:** Math.
- **Problem:** If the step-size parameters `I_n` are not constant, then the estimate `Q_n` is a weighted
  average of previously received rewards with a weighting different from that given by Eq. (2.1). What
  is the weighting on each prior reward for the general case, analogous to Eq. (2.1)?
**Hook.** The incremental update hides a weighted average. Unrolling it once shows how a step-size
sequence decides how much the past still counts.

**Formal bridge.** Writing the step size as \(\alpha_n\) (the problem's \(I_n\)), the recursion
\(Q_{n+1}=Q_n+\alpha_n(R_{n+1}-Q_n)\) unrolls to

\[
Q_{n+1} = \left(\prod_{j=1}^{n}(1-\alpha_j)\right)Q_1
        + \sum_{i=1}^{n} \alpha_i\left(\prod_{j=i+1}^{n}(1-\alpha_j)\right)R_{i+1}.
\tag{2.4.1}
\]

where \(\alpha_i\in(0,1]\) is the step size used on the \(i\)-th update, \(R_{i+1}\) is the reward
received then, \(Q_1\) is the initial estimate, and the products are over all later updates.

**Reasoning chain.** Work from the special cases. With a constant step size \(\alpha_i=\alpha\), the
product over later updates becomes \((1-\alpha)^{n-i}\), so every reward \(R_{i+1}\) gets weight
\(\alpha(1-\alpha)^{n-i}\) and the initial estimate keeps weight \((1-\alpha)^n\). This is the
geometric weighting from the original solution: recent rewards count more, and the weights sum to
\(1-(1-\alpha)^n\), with the remainder sitting on \(Q_1\).

With the sample-average step size \(\alpha_i=1/i\) (the document's \(I_i=\delta/N_i\) with
\(\delta=1\)), the telescoping product \(\prod_{j=i+1}^{n}(1-1/j)=i/n\) cancels the \(1/i\), leaving
weight \(1/n\) on every reward and zero on the initial estimate. That is Eq. (2.1)'s uniform
average.

The general answer is therefore the product formula in (2.4.1): reward \(R_{i+1}\) is weighted by
its own step size \(\alpha_i\) times the survival factor \((1-\alpha_j)\) for every subsequent
update. A large step size now amplifies that reward but also shrinks the weight of everything that
came before. The second convergence condition on the step-size sequence — roughly, that the steps
shrink but their sum diverges — is what makes these weights spread out enough to average away
noise while still letting old data be forgotten.

**Why it makes sense.** Picture the two extremes. Constant \(\alpha\) is a leaky bucket: old rewards
are diluted by a fixed factor at every step, so the estimate forever partially forgets its past —
exactly what is wanted when rewards drift. Sample averages are a sealed ledger: every observation
keeps weight \(1/n\) forever, which is ideal when the target is fixed but disastrous when the
target moves, because ancient data never loses influence. The product in (2.4.1) interpolates
between these pictures by choosing how quickly old weights decay.

**Check your understanding.** Prediction: switch from \(\alpha_i=1/i\) to a constant \(\alpha=0.1\).
Old rewards now decay geometrically rather than persisting at \(1/n\), so the estimate responds
faster to a recent change in rewards but carries a bias toward the initial value that vanishes only
as \((1-\alpha)^n\). Letting \(\alpha\to 0\) recovers near-uniform weighting.

### Exercise 2.5 — (Programming) Non-stationary experiment

- **Book location:** Section 2.5, PDF page 44.
- **Type:** Programming.
- **Problem:** Design and conduct an experiment to demonstrate the difficulties that sample-average
  methods have for non-stationary problems. Use a modified version of the 10-armed testbed in which all
  `Q(a)` start equal and then take independent random walks (e.g., add a N(0, 0.01) increment to all
  `Q(a)` each step). Prepare plots like Figure 2.2 for an action-value method using sample averages,
  incrementally computed, and another using a constant ε = 0.1 (for the ε-greedy method). Use ε = 0.1
  and longer runs, e.g., 10,000 steps.

**Hook.** We want to see a failure, not just hear about it. The experiment gives the sample-average
method a moving target and lets it expose how badly it holds on to obsolete data.

**Formal bridge.** Make the task non-stationary by giving every true value an independent random
walk,

\[
q_{t+1}(a) = q_t(a) + \mathcal{N}(0,\,0.01),
\tag{2.5.1}
\]

and compare two estimators of the same drifting target:

\[
Q_{n+1}(a) = Q_n(a) + \alpha\left(R_{n+1} - Q_n(a)\right),
\qquad \alpha = \frac{1}{n}\ \text{(sample average)} \quad\text{vs.}\quad \alpha\ \text{constant}.
\tag{2.5.2}
\]

where \(q_t(a)\) is the true expected reward of action \(a\) at time \(t\), \(\mathcal{N}(0,0.01)\) is
a Gaussian increment with mean 0 and variance 0.01, \(Q_n(a)\) is the learned estimate after \(n\)
plays of \(a\), and \(\alpha\) is the step size — \(1/n\) for sample averages, a constant for the
tracker.

**Reasoning chain.** Build a 10-armed testbed with all \(q_0(a)\) equal, so no action starts with an
advantage. At every step, before choosing an action, add an independent \(\mathcal{N}(0,0.01)\)
increment to each \(q(a)\); the drift is small per step but accumulates over a long run. Generate
many independent tasks (the book uses 2000 for its testbeds) and average within each step, so the
curves reflect the algorithm rather than one lucky walk.

Run both action-value methods with the same ε-greedy action selection, \(\epsilon=0.1\), so the only
difference is the estimator in (2.5.2). The first uses the incrementally computed sample average
(\(\alpha=1/n\)); the second uses a constant step size. Extend each run to about 10,000 steps —
long enough for the random walks to move well away from their initial values — and plot the two
curves in the style of Figure 2.2: average reward versus step, and preferably also the percentage
of times the currently best action is chosen.

The expected picture follows from Exercise 2.4's weighting. The sample-average estimate keeps a
uniform average over a reward history whose target has changed, so it is dominated by old data and
lags behind the drift; its best-action percentage degrades as the run goes on. The constant-step
estimate discounts old rewards and follows the moving target, so its curves stay noticeably higher.
That gap, reproduced across many tasks, is the demonstration.

**Why it makes sense.** A sample average is an average over "what this arm used to pay", with every
observation carrying equal weight no matter how old. If the arm's true value wanders, the newest
observation is the only one describing the present, yet it gets the same \(1/n\) weight as data
from thousands of steps ago. The direction of the error is therefore a lag: the estimate trails the
current value, and the longer the run, the more obsolete data it must drag along. A constant step
size gives the newest reward disproportionate influence, biasing the estimate toward the present —
the right direction for a drifting target.

**Check your understanding.** Prediction: set the drift variance to zero so the task becomes
stationary. The constant-step method no longer has an advantage; the sample average becomes the
more accurate estimator (unbiased with lower asymptotic variance), and the two curves converge as
the run grows.

### Exercise 2.6 — Mysterious spikes in optimistic method

- **Book location:** Section 2.6, PDF page 45.
- **Type:** Theory.
- **Problem:** The results in Figure 2.3 should be reliable (averages over 2000 tasks), so why are
  there oscillations and spikes in the early part of the curve for the optimistic method?
**Hook.** Averaging over 2000 tasks removes random noise — but not a systematic transient that all
2000 tasks share. Let's find out why the spikes survive the averaging.

**Formal bridge.** With an optimistic start \(Q_1(a)=+5\), the expected estimate after \(n\) plays of
an action with true value \(q(a)\) is

\[
\mathbb{E}[Q_{n+1}(a)] - q(a) = (1-\alpha)^n\left(Q_1(a)-q(a)\right),
\tag{2.6.1}
\]

so early estimates remain above the true values and decline geometrically as evidence arrives.

where \(Q_1(a)\) is the initial optimistic estimate, \(q(a)\) is the true action value, \(\alpha\) is
the step size (or \(1/n\) for sample averages), and \(n\) is the number of times \(a\) has been tried.

**Reasoning chain.** Setting every \(Q_1(a)=+5\), far above the roughly unit-scale rewards of the
10-armed testbed, makes a greedy action selector behave like an explorer. Any untried or sparsely
sampled action looks better than one whose estimate has already been pulled down by a real reward,
so the learner cycles through actions, trying each until disappointment lowers its estimate.
Equation (2.6.1) is the mechanism: the initial gap shrinks only as \((1-\alpha)^n\), which is slow
when few samples have been collected.

Why does this produce *spikes* in an average over 2000 tasks? Because the optimism is shared. All
tasks start from the same \(+5\), so they enter the disappointment phase at roughly the same early
steps and their estimates are all based on only a handful of noisy rewards. At those steps the
curve is an average of high-variance, partially synchronised transitions — some tasks have already
found a good action and others have not — so the mean oscillates and spikes rather than being
smoothed away. Later, as estimates settle near their true values, the tasks desynchronise and the
curve flattens.

The exploration drive is temporary by construction: once all estimates have dropped below the
initial optimism, there is no remaining incentive to try actions just because they look untested.
Performance therefore oscillates early and then stabilises — the spikes are a transient of
learning, not a defect in the averaging.

**Why it makes sense.** Imagine setting \(Q_1\) enormously high, say \(+10^6\). Every first action
looks equally marvellous, so the learner samples all of them before any can look bad; the early
phase becomes a long, highly synchronised scan with wild swings as huge estimates collapse one by
one. Now set \(Q_1\) equal to the true mean of each action: (2.6.1) has zero initial gap, no
disappointment phase, and no spikes. The spikes are therefore the visible footprint of a large
initial gap being paid off with little data — they cannot appear once the gap is zero.

**Check your understanding.** Prediction: lower the optimistic start from \(+5\) to \(+1\). The early
spikes shrink and the initial exploration phase shortens, because (2.6.1) starts with a smaller gap;
if \(Q_1\) is raised instead, the spikes grow and occur later, since more samples are needed to
dispel the optimism.

### Exercise 2.7 — Softmax equals logistic

- **Book location:** Section 2.8, PDF page 47.
- **Type:** Math / Proof.
- **Problem:** Show that in the case of two actions, the softmax distribution is the same as the
  logistic/sigmoid function used in statistics and neural networks.
**Hook.** Softmax and sigmoid look like different formulae from different subjects. With two
actions they are the same object written twice — let's make the algebra explicit.

**Formal bridge.** The softmax probability of action \(a_1\) equals the logistic function of the
preference difference:

\[
\frac{e^{H(a_1)}}{e^{H(a_1)}+e^{H(a_2)}}
= \frac{1}{1+e^{-\left(H(a_1)-H(a_2)\right)}}
= \sigma\!\left(H(a_1)-H(a_2)\right).
\tag{2.7.1}
\]

where \(H(a)\) is the learned preference (numeric propensity) for action \(a\), and
\(\sigma(x)=1/(1+e^{-x})\) is the logistic/sigmoid function.

**Reasoning chain.** Start from the softmax expression for two actions. Divide numerator and
denominator by \(e^{H(a_1)}\). The numerator becomes \(1\), and the denominator becomes
\(1+e^{H(a_2)-H(a_1)} = 1+e^{-\left(H(a_1)-H(a_2)\right)}\). That is precisely
\(\sigma(H(a_1)-H(a_2))\). The probability of \(a_2\) is the complementary
\(\sigma(H(a_2)-H(a_1))=1-p(a_1)\), so the whole distribution is captured by one difference.

This also explains a useful invariance: adding the same constant \(c\) to both preferences leaves
\(H(a_1)-H(a_2)\) unchanged, so softmax depends only on relative preferences, not their absolute
level. In the gradient bandit algorithm this is why preferences can drift without changing the
action probabilities at all.

**Why it makes sense.** Look at the extremes of (2.7.1). If \(H(a_1)=H(a_2)\), the difference is
zero and \(\sigma(0)=1/2\): with no evidence favouring either action, softmax is a fair coin. If
\(H(a_1)\) exceeds \(H(a_2)\) by a large amount, \(e^{-(H(a_1)-H(a_2))}\) tends to zero and the
probability tends to \(1\), approaching greedy selection smoothly. The direction is right: raising
one preference raises its probability, and the sigmoid is exactly the smooth S-curve that
interpolates between "coin flip" and "certain choice".

**Check your understanding.** Prediction: add \(+10\) to both \(H(a_1)\) and \(H(a_2)\). Nothing
changes, because only the difference enters (2.7.1). Multiply both preferences by \(2\), however,
and the difference doubles, so the distribution becomes more decisive (closer to a greedy choice).

### Exercise 2.8 — Associative search / contextual bandits

- **Book location:** Section 2.9, PDF page 50.
- **Type:** Theory.
- **Problem:** 2-armed bandit whose true action values change randomly (case A: actions 1/2 = 0.1/0.2;
  case B: 0.9/0.8). (a) Without cues: best expectation of success and how to behave? (b) With case
  labels each step: best expectation of success and how to behave?
**Hook.** The same two arms flip which one is better depending on a hidden case. The question is how
much a cue is worth — that is, how much information changes what any policy can achieve.

**Formal bridge.** With cases \(c\in\{A,B\}\) and a policy \(\pi\) that may condition on the observed
cue,

\[
\mathbb{E}[\text{success}] = \sum_{c} p(c)\, q\!\left(c, \pi(c)\right),
\tag{2.8.1}
\]

where \(q(A,1)=0.1,\ q(A,2)=0.2,\ q(B,1)=0.9,\ q(B,2)=0.8\), and \(\pi(c)\) is the action chosen
when the cue for case \(c\) is present (a single fixed action when no cue is available).

**Reasoning chain.** Part (a): with no cue, \(\pi\) must pick one action regardless of the case.
If the two cases are equally likely, action 1 gives \(\tfrac12(0.1)+\tfrac12(0.9)=0.5\) and action
2 gives \(\tfrac12(0.2)+\tfrac12(0.8)=0.5\), so the best achievable expectation of success is
\(0.5\). More generally, with case probability \(p(A)=p\), the best fixed action maximises
\(p\,q(A,a)+(1-p)\,q(B,a)\); without a cue the agent cannot do better than this prior-weighted
choice, because the reward stream mixes the two cases indistinguishably. It cannot identify which
case it faces on a given step, so no case-specific behaviour is possible.

Part (b): with the case label shown each step, the agent can condition its action on \(c\). In case
A, action 2 (\(0.2\)) beats action 1 (\(0.1\)); in case B, action 1 (\(0.9\)) beats action 2
(\(0.8\)). Choosing \(\pi(A)=2\) and \(\pi(B)=1\) gives, for equally likely cases,
\(\tfrac12(0.2)+\tfrac12(0.9)=0.55\). That is the optimal expected success: each case is now a
small stationary 2-armed bandit, and the learner can estimate and exploit the better arm within
each context. The cue has turned a non-stationary bandit into a contextual bandit.

**Why it makes sense.** No algorithm can create information it never receives. Without cues, the
observation distribution is a mixture of two regimes, and a single action is a compromise between
them — here exactly \(0.5\) by symmetry. With cues, the stream is partitioned into two clean
sub-streams, so the learner can be locally greedy in each; the gain \(0.55-0.5=0.05\) per step is
the value of knowing which regime is active. If the cases were not equally likely, the
no-cue optimum would shift toward the action best in the more common case, while the cued optimum
would keep both correct choices — information never makes the best achievable value smaller.

**Check your understanding.** Prediction: make case B rare, say \(p(B)=0.1\). Without cues, action 2
becomes better (\(0.9\times0.1+0.2\times0.9=0.27\) vs. action 1's \(0.18\)), while with cues the
optimal policy is unchanged: action 2 in A, action 1 in B. The value of the cue shrinks because the
rare case contributes little.

### Exercise 2.9 — (Programming) Figure for the non-stationary case

- **Book location:** Section 2.9, PDF page 52.
- **Type:** Programming.
- **Problem:** Make a figure analogous to Figure 2.6 for the non-stationary case outlined in Exercise
  2.5. Include the constant-step-size ε-greedy algorithm with ε = 0.1. Use runs of 200,000 steps and,
  as the performance measure for each algorithm and parameter setting, use the average reward over the
  last 100,000 steps.

**Hook.** Figure 2.6 sweeps algorithm parameters; here we repeat that sweep on a moving target. The
late-window average keeps the comparison honest by throwing away the early transient.

**Formal bridge.** For each algorithm and parameter setting, the plotted performance is

\[
\bar R = \frac{1}{100{,}000}\sum_{t=100{,}001}^{200{,}000} R_t,
\tag{2.9.1}
\]

where \(R_t\) is the reward at step \(t\), averaged over many independent runs of the drifting
10-armed testbed from Exercise 2.5, with each arm following \(q_{t+1}(a)=q_t(a)+\mathcal{N}(0,0.01)\).

**Reasoning chain.** Reuse the non-stationary testbed of Exercise 2.5: ten arms starting equal, each
value taking an independent \(\mathcal{N}(0,0.01)\) random walk every step. Sweep the algorithm
parameters as Figure 2.6 does — for ε-greedy, a range of \(\epsilon\) values and step sizes
\(\alpha\); for UCB, a range of the exploration coefficient \(c\); for gradient bandits, a range of
\(\alpha\) — and include the constant-step-size ε-greedy algorithm at \(\epsilon=0.1\) explicitly,
as required.

Run each setting for 200,000 steps, repeated over many independent tasks. Discard the first 100,000
steps entirely and compute (2.9.1) on the remaining half. Plot this late-window average on the
vertical axis against the swept parameter on the horizontal axis, one curve per algorithm, matching
Figure 2.6's style. Keeping the same random-walk tasks across settings makes the curves comparable.

The expected result mirrors Exercise 2.5 but more sharply: sample-average methods (\(\alpha=1/n\))
sit low because their estimates keep averaging over obsolete rewards; constant-step methods reach
higher late-window averages because they keep tracking the current values; and the best parameter
is a nonzero \(\alpha\) rather than the smallest one. This is how the figure demonstrates that
non-stationary problems favour constant step sizes.

**Why it makes sense.** In a drifting problem there is no convergence, so any performance measure
that includes the start of the run is dominated by the initial learning transient — the wrong
thing to compare when the question is "which method tracks best once it has warmed up?" Taking
the average over the *last* 100,000 steps answers that question. The direction of the result is
also forced: a sample average's weights do not decay (Exercise 2.4), so its estimate always lags a
moving target; a constant step size gives recent rewards more weight, and recency is exactly what a
drifting target rewards.

**Check your understanding.** Prediction: shrink the measurement window to the last 1,000 steps.
The ordering of the methods should stay the same, but every curve becomes noisier because fewer
rewards enter (2.9.1); if the drift variance is set to zero, the advantage of constant step sizes
disappears and small step sizes or sample averages do best.

---

## Exploration Strategies and Derived Exercises (book §§2.6–2.7)

> The exploration material that previously lived in the old repository
> Chapter 11 (Exploration) is folded into this chapter: the repo has no
> separate book chapter for "exploration", so book Chapter 2 (multi-armed
> bandits) is the reference implementation home. The derived exercises below
> cover optimistic initial values (book §2.6), UCB action selection
> (book §2.7), and the optimism-in-the-face-of-uncertainty family
> (Exercise 11.1 / descending bonus).

### Derived 1 — Bias of sample-average vs constant-step-size estimators [derived]

- **Book location:** Section 2.6, PDF pages 44–45.
- **Type:** Math / Theory.
- **Problem:** Sample-average returns treat the past uniformly; the constant-step-size estimator
  weights recent rewards more. Show that sample-average estimates are unbiased once every action has
  been selected at least once, while constant-step-size estimates carry bias that decreases with time.
**Hook.** Let's take the incremental update apart and see whose data it is really averaging. The surprise
is that the two step-size choices differ less in *what* they average than in how much weight the starting
guess keeps.

**Formal bridge.** Under stationary rewards every reward has the same mean \(q_*(a)\), so the sample
average is unbiased as soon as one reward exists:

\[
\mathbb{E}[Q_n(a)] = \frac{1}{n}\sum_{i=1}^{n}\mathbb{E}[R_i] = q_*(a).
\tag{2.D1.1}
\]

The constant-step recursion \(Q_{k+1}=Q_k+\alpha(R_k-Q_k)\) unrolls differently:

\[
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i,
\qquad
\mathbb{E}[Q_{n+1}]-q_*(a) = (1-\alpha)^n\left(Q_1-q_*(a)\right).
\tag{2.D1.2}
\]

where \(Q_n(a)\) is the estimate after \(n\) plays of action \(a\), \(q_*(a)\) is its true expected reward,
\(R_i\) is the reward received on the \(i\)-th update, \(Q_1\) is the initial estimate, \(\alpha\in(0,1)\)
is the constant step size, and \(n\) is the number of updates so far.

**Reasoning chain.** For the sample average, the first update cancels \(Q_1\) and every one of the \(n\)
rewards then carries weight \(1/n\). Since each has mean \(q_*(a)\), their average has that mean too —
that is (2.D1.1), unbiased for every \(n\ge 1\). No initial guess survives.

For constant \(\alpha\), the weights on past rewards are geometric: \(\alpha(1-\alpha)^{n-i}\) for reward
\(R_i\). They sum to \(1-(1-\alpha)^n\), so the remaining weight \((1-\alpha)^n\) still sits on \(Q_1\).
Taking expectations makes the reward part equal to \(q_*(a)\) times the summed weights, which leaves
exactly the initial-error term in (2.D1.2). The bias is not in the rewards — it is the starting guess
refusing to die.

That term is nonzero for any finite \(n\) whenever \(Q_1\neq q_*(a)\), but it decays geometrically. More
recent rewards also get more weight, so the estimate never becomes a uniform average of the whole
history. This recency weighting is precisely why constant \(\alpha\) tracks nonstationary problems
better than sample averages, which keep averaging over obsolete data.

**Why it makes sense.** Push \(\alpha\) to an extreme. As \(\alpha\to 1\), the estimate becomes just the
latest reward: the bias \((1-\alpha)^n(Q_1-q_*(a))\) vanishes after one step, but variance is high. As
\(\alpha\to 0\), the estimate barely moves and the initial error persists almost forever. The bias must
*decay* rather than grow, because each new reward is unbiased and gets positive weight while the old
guess is multiplied by \((1-\alpha)<1\) every time. Its sign is the sign of \(Q_1-q_*(a)\): an optimistic
start biases the estimate upward, a pessimistic one downward.

**Check your understanding.** Prediction: double the initial gap \(Q_1-q_*(a)\) and the bias in (2.D1.2)
doubles at every \(n\), but its decay rate is unchanged; halve \(\alpha\) and the gap closes roughly twice
as slowly. If \(Q_1=q_*(a)\) exactly, the constant-step estimator is unbiased from the very first update.

### Derived 2 — Experiment isolating optimistic initialization [derived]

- **Book location:** Section 2.6, PDF page 44.
- **Type:** Programming / Theory.
- **Problem:** Design an experiment to separate the effect of optimistic initialization from other
  exploration mechanisms. Predict which early steps' performance will differ between the optimistic
  greedy method and an ε-greedy method on the 10-armed testbed.
**Hook.** To isolate a mechanism, change as little else as possible. Here we keep the testbed and the
estimator fixed, and vary only *where the exploration comes from*: a tilted starting value under greedy
selection, or a random branch added to greedy selection.

**Formal bridge.** Both methods use the same incremental estimator but different starting values,
and the optimistic method's extra pull is just its initial error decaying:

\[
Q_{n+1}(a) = Q_n(a)+\alpha\left(R_{n+1}-Q_n(a)\right),
\qquad
\mathbb{E}[Q_{n+1}(a)]-q(a) = (1-\alpha)^n\left(Q_1(a)-q(a)\right).
\tag{2.D2.1}
\]

To compare them fairly, average reward in an early window and a late window separately:

\[
\bar R_{1:m} = \frac{1}{m}\sum_{t=1}^{m}\mathbb{E}[R_t],
\qquad
\bar R_{m+1:T} = \frac{1}{T-m}\sum_{t=m+1}^{T}\mathbb{E}[R_t].
\tag{2.D2.2}
\]

where \(q(a)\) is the true value of action \(a\), \(Q_n(a)\) is the estimate after \(n\) plays of it,
\(Q_1(a)\) is the starting value (\(+5\) for the optimistic method, \(0\) for ε-greedy), \(\alpha\) is the
step size, \(R_t\) is the reward at step \(t\), \(m\) is the end of the early window, and \(T\) is the run
length.

**Reasoning chain.** Run a stationary 10-armed testbed with the same estimator for both methods. Method A
is greedy action selection with every \(Q_1(a)=+5\); method B is ε-greedy with \(\epsilon=0.1\) and
\(Q_1(a)=0\). Identical tasks and run lengths make the comparison about the exploration mechanism alone.
Compute (2.D2.2) over the first few dozen steps and over the final stretch.

The two mechanisms act at different times — optimism is front-loaded. With \(Q_1=+5\), every untried arm
looks marvellous, and a tried arm's estimate is pulled toward ordinary reward scale after its first
sample, so greedy selection sweeps the untried arms before revisiting any of them. That is a systematic
early scan, driven by the decaying term in (2.D2.1). Once the initial gap has been paid off, the optimism
is spent and the method behaves like plain greedy, with no further exploration budget.

ε-greedy does not spend a budget: it takes a random branch with fixed probability \(\epsilon\) at every
step, so its exploration cost is spread evenly across the whole run. The prediction is therefore that
the two curves differ mainly early — where the optimistic method's forced scan changes both reward and
information gain — while late in the run ε-greedy keeps paying a steady exploration cost and the
optimistic method has reverted to greedy. The early-window and late-window averages in (2.D2.2) expose
that difference; a single end-of-run average would hide it.

**Why it makes sense.** Take the extremes. Set \(Q_1\) enormously high and greedy selection must sample
every arm many times before any estimate falls back to reality — a long, synchronized scan. Set \(Q_1=0\)
and the optimistic method becomes ordinary greedy, which can lock onto a lucky first arm forever. Neither
changes ε-greedy's behaviour, because its exploration is generated step by step. The direction follows:
optimism cannot help late in a stationary run, since (2.D2.1) guarantees its extra push only shrinks;
random exploration can.

**Check your understanding.** Prediction: lower \(Q_1\) from \(+5\) toward the reward scale and the early
difference between the two methods shrinks and disappears sooner. Set \(\epsilon=0\) and ε-greedy becomes
greedy on zero-initialized values, losing its late-run exploration while the optimistic method still
gets its initial scan.

### Derived 3 — UCB index from a confidence argument [derived]

- **Book location:** Section 2.7, PDF pages 45–46.
- **Type:** Math / Proof.
- **Problem:** Derive the UCB index \(Q_t(a) + c\cdot sqrt(ln t / N_t(a))\) from a Chernoff/Hoeffding-style
  confidence bound. Explain why `ln t` (and not `t`) appears and why untried actions are preferred.
**Hook.** The UCB bonus looks like a heuristic until we ask what it would take to be *confident*. Let's
build it from a tail bound and watch the logarithm appear on its own.

**Formal bridge.** For rewards bounded in \([0,1]\), Hoeffding's inequality limits how far the sample
mean can stray from the truth:

\[
P\!\left(\left|\hat Q_t(a)-q_*(a)\right|>\varepsilon\right)
\le 2\exp\!\left(-2N_t(a)\varepsilon^2\right).
\tag{2.D3.1}
\]

Set a per-\((a,t)\) failure budget \(\delta_t\) by making the right-hand side equal to \(\delta_t\).
Solving for the confidence radius gives the UCB index:

\[
\varepsilon_t(a)=\sqrt{\frac{\ln(2/\delta_t)}{2N_t(a)}}
\;\xrightarrow{\;\delta_t\propto 1/t^2\;}\;
O\!\left(\sqrt{\frac{\ln t}{N_t(a)}}\right),
\qquad
U_t(a)=\hat Q_t(a)+c\sqrt{\frac{\ln t}{N_t(a)}}.
\tag{2.D3.2}
\]

where \(\hat Q_t(a)\) is the sample mean of action \(a\) after \(N_t(a)\) plays, \(q_*(a)\) is its true
mean, \(\varepsilon\) and \(\varepsilon_t(a)\) are confidence radii, \(\delta_t\) is the allowed tail
probability at step \(t\), \(t\) is the time step, \(c\) is an exploration coefficient absorbing the
constant factors, and \(U_t(a)\) is the index the agent maximises, \(A_t=\arg\max_a U_t(a)\).

**Reasoning chain.** Start with (2.D3.1): the probability of a large estimation error shrinks
exponentially in the number of pulls \(N_t(a)\). We want a one-sided upper bound on \(q_*(a)\) that holds
except on rare failures, so we declare a budget \(\delta_t\), set the tail probability equal to it, and
solve. The result is the radius in (2.D3.2): it falls like \(1/\sqrt{N_t(a)}\) because more data means a
tighter interval.

The choice \(\delta_t\propto 1/t^2\) is what produces the logarithm. Then \(\ln(2/\delta_t)=\ln 2+2\ln t
=\Theta(\ln t)\), and the constant factor is folded into \(c\). Because \(\sum_t\delta_t<\infty\), a union
bound over time and actions makes the total failure probability finite; by Borel–Cantelli, only finitely
many confidence failures occur, so \(\hat Q_t(a)+\varepsilon_t(a)\) covers \(q_*(a)\) uniformly well
enough. A suboptimal arm then stops being attractive once its bonus cannot bridge its gap, and standard
analysis bounds its pulls by \(O(\log t)\). UCB therefore commits to exploring each action only
\(O(\log t)\) times, giving near-optimal regret on stationary problems.

Why \(\ln t\) and not \(t\)? The budget \(\delta_t\) must be summable, and \(1/t\) is not — it would let
failures keep occurring forever. What survives summation is \(\ln t\), which grows slowly enough that the
bonus stays small as data accumulates, but still grows without bound so a rarely tried arm is eventually
rescued. Actions with \(N_t(a)=0\) have no bound at all; by convention they are treated as maximising,
which forces the initial sample before (2.D3.2) is even defined.

**Why it makes sense.** Compare the alternatives. If the bonus grew like \(\sqrt{t}\) instead of
\(\sqrt{\ln t}\), uncertainty would increase almost as fast as time passes and the agent would keep
re-sampling clearly bad arms — regret would grow far too quickly. If it stayed constant, some arms might
never be sampled and a better arm could be missed forever. The signs in (2.D3.2) are also forced: the
bonus must *shrink* with \(N_t(a)\) because data removes uncertainty, and *grow* with \(t\) because more
steps mean more chances for a rare miss to matter.

**Check your understanding.** Prediction: raise \(c\) and the intervals widen, so the agent re-tries stale
arms more often and converges later; set \(c=0\) and the rule collapses to greedy, which can miss the best
arm entirely. Replace \(\delta_t\propto 1/t^2\) with \(\delta_t\propto 1/t\): \(\ln t\) still appears but
\(\sum_t\delta_t\) diverges, so the uniform confidence guarantee is lost.

### Derived 4 — Every action is explored infinitely often under UCB, but rarely on average [derived]

- **Book location:** Section 2.7, PDF page 46.
- **Type:** Math / Theory.
- **Problem:** Show that under the UCB rule every action is selected infinitely often, but the index
  gap for suboptimal actions grows so that their selection frequency decreases to zero.
**Hook.** These two claims sound contradictory — try every action forever, yet try each one less and less.
The resolution is that the bonus depends on a *ratio*: a numerator that creeps up and a denominator that
can only grow when the action is actually played.

**Formal bridge.** The index driving both effects is

\[
U_t(a)=Q_t(a)+c\sqrt{\frac{\ln t}{N_t(a)}}.
\tag{2.D4.1}
\]

For a suboptimal action \(a\) with gap \(\Delta_a=q_*-q(a)>0\), once its count is large enough the bonus
can no longer bridge the gap:

\[
N_t(a)\gtrsim \frac{4c^2\ln t}{\Delta_a^2}
\;\Longrightarrow\;
U_t(a)\le q(a)+\frac{\Delta_a}{2}<q_*\le U_t(a_*),
\tag{2.D4.2}
\]

so \(N_t(a)=O\!\left(\frac{\ln t}{\Delta_a^2}\right)\) while \(N_t(a)\to\infty\).

where \(Q_t(a)\) is the current estimate of action \(a\), \(N_t(a)\) is its pull count, \(c\) is the
exploration coefficient, \(t\) is the time step, \(q_*=\max_b q(b)\) is the best true value, \(q(a)\) is
the true value of \(a\), \(\Delta_a>0\) is its gap, and \(a_*\) is a best action.

**Reasoning chain.** First, every action is tried infinitely often. Suppose some action \(a\) were pulled
only finitely many times, so \(N_t(a)=N\) stayed fixed. The \(\sqrt{\ln t/N}\) term in (2.D4.1) is
unbounded as \(t\to\infty\), while the indices of frequently played actions have growing denominators and
bounded estimates. Eventually \(U_t(a)\) must dominate, forcing another pull — a contradiction. Hence
\(N_t(a)\to\infty\) for every \(a\). That is the sense in which UCB never gives up on an arm.

Second, suboptimal actions are rare on average. Each pull of \(a\) raises \(N_t(a)\) and shrinks its bonus.
For an arm with gap \(\Delta_a\), its index can beat the best arm's only while the bonus is at least of
order \(\Delta_a\). Inequality (2.D4.2) shows that this stops once \(N_t(a)\) reaches roughly
\(4c^2\ln t/\Delta_a^2\). Since the confidence argument of Derived 3 fails only finitely often, this holds
eventually, so \(N_t(a)\) grows like \(\log t\), not like \(t\). Dividing by \(t\) gives
\(N_t(a)/t\to 0\): selection frequency vanishes even though the count diverges.

**Why it makes sense.** Picture time passing. \(\ln t\) creeps upward, while the best arm's count grows
roughly linearly, so its bonus tends to zero and it is chosen almost always. A neglected arm keeps a
small denominator, so its bonus grows without bound and it gets rescued. The two forces balance exactly
at \(N\propto\ln t/\Delta^2\). If the bonus shrank like \(1/N\) instead of \(1/\sqrt{N}\), a neglected arm
would never be rescued; if it grew like \(\sqrt{t}\), suboptimal arms would be pulled a constant fraction
of the time. The direction of (2.D4.1) is the only one that gives both behaviours at once.

**Check your understanding.** Prediction: halve a suboptimal arm's gap \(\Delta_a\) and its pull count grows
about fourfold, since the threshold in (2.D4.2) scales as \(1/\Delta_a^2\); raise \(c\) and it is pulled more
often. If all arms are equally good (\(\Delta_a=0\)), none is suboptimal and counts spread evenly.

### Derived 5 — Optimistic initialization vs UCB: practical trade-offs [derived]

- **Book location:** Sections 2.6–2.7, PDF pages 44–46.
- **Type:** Theory.
- **Problem:** Compare optimistic initialization and UCB on (a) stationary vs nonstationary problems,
  (b) simplicity/intuition, (c) when each fails. Which would you use, and when?
**Hook.** Both methods explore by making uncertain actions look good, but they spend that optimism very
differently. Let's compare them on what they guarantee, what they cost to use, and where each breaks.

**Formal bridge.** Optimistic initialization adds a decaying initial-error term to the estimate,

\[
\mathbb{E}[Q_{n+1}(a)]-q(a)=w_n\left(Q_1(a)-q(a)\right),
\quad
w_n=\frac{1}{n+1}\ \text{(sample average)}\quad\text{or}\quad (1-\alpha)^n\ \text{(constant }\alpha\text{)},
\tag{2.D5.1}
\]

whereas UCB keeps a standing bonus that shrinks only as its visit count grows:

\[
U_t(a)=Q_t(a)+c\sqrt{\frac{\ln t}{N_t(a)}}.
\tag{2.D5.2}
\]

where \(q(a)\) is the true value of action \(a\), \(Q_1(a)\) is the starting estimate, \(Q_{n+1}(a)\) is the
estimate after \(n\) plays, \(\alpha\) is a constant step size, \(Q_t(a)\) and \(N_t(a)\) are the current
estimate and pull count, \(c\) is the exploration coefficient, and \(t\) is the time step.

**Reasoning chain.** On stationary problems, UCB is the method with a proof. Its confidence argument
(Derived 3) bounds suboptimal pulls by \(O(\log t)\), so regret grows only logarithmically. Optimistic
initialization is a heuristic by comparison: it pushes exploration early and then stops, with no
corresponding guarantee, but it often works well when \(Q_1\) is set above the plausible reward range.

In simplicity, optimism wins. It is a one-line change to \(Q_1\) plus the convention that untried actions
are maximising; there are no counters, logarithms, or coefficients to tune. UCB requires maintaining
\(N_t(a)\), the \(\ln t\) factor, and a choice of \(c\), though its confidence-interval interpretation is
still intuitive.

Each fails differently. Setting \(Q_1\) too high wastes many steps cycling through disappointing arms
and produces early spikes (Exercise 2.6); setting it too low collapses to greedy and risks locking onto a
lucky suboptimal arm. UCB fails when \(c\) is badly chosen, with many arms relative to the gaps, and on
nonstationary tasks: \(N_t(a)\) never discounts old data, so an arm that was once good keeps a large count
and a small bonus in (2.D5.2) even after its value drifts down. The optimism in (2.D5.1) only decays.
Neither is suited to nonstationary problems without modification — a sliding window, discounted counts,
or a constant step size is needed.

So: use UCB for stationary problems when a regret guarantee matters and the bookkeeping is affordable;
use optimistic initialization for a quick baseline, a short run, or when a constant step size is already
handling drift.

**Why it makes sense.** Optimism is a one-time investment: the weight \(w_n\) in (2.D5.1) can only shrink,
so the method cannot reopen exploration when the world changes. UCB's bonus also shrinks with \(N_t(a)\),
and \(\ln t\) never signals that an arm's value has moved. Both methods "believe" old evidence more as
time passes — the right behaviour for a fixed target, the wrong one for a drifting target. To adapt, the
mechanism must re-open uncertainty rather than spend a fixed budget early.

**Check your understanding.** Prediction: make the rewards drift and both methods degrade — optimism's
early push is unchanged but its later greedy phase acts on stale estimates, while UCB's small bonuses keep
it from revisiting drifted arms. Shrink the horizon to a few dozen steps and optimism becomes relatively
more attractive, because its front-loaded exploration is exactly what a short run can use.

### Derived 6 — Why UCB does not extend trivially to large state spaces / function approximation [derived]

- **Book location:** Section 2.7, PDF page 46.
- **Type:** Theory / Math.
- **Problem:** Explain why UCB1 is hard to extend beyond bandits (large state spaces, function
  approximation). Describe the general principle of optimistic value functions (OFU — optimism in the
  face of uncertainty) and its common tabular/linear instantiation (LinUCB); give its confidence set.
**Hook.** UCB1's bonus is built from a count, and counts are exactly what a large state space cannot
provide. Let's see what has to replace the count when many states share one learner.

**Formal bridge.** Tabular UCB scores each \((s,a)\) using its own visit count,

\[
U_t(s,a)=Q_t(s,a)+c\sqrt{\frac{\ln t}{N_t(s,a)}}.
\tag{2.D6.1}
\]

With a linear model \(\mathbb{E}[r\mid s,a]=x(s,a)^\top\theta_*\), evidence from different states informs
the same parameter vector, and uncertainty lives in a covariance matrix:

\[
\hat\theta_t=D_t^{-1}\sum_{i<t}x_i r_i,
\qquad
D_t=\lambda I+\sum_{i<t}x_i x_i^\top.
\tag{2.D6.2}
\]

Optimism in the face of uncertainty (OFU) then chooses the action best under the most favourable
parameter in the confidence set \(\mathcal{C}_t\):

\[
\mathcal{C}_t=\left\{\theta:(\theta-\hat\theta_t)^\top D_t(\theta-\hat\theta_t)\le\beta_t^2\right\},
\qquad
A_t=\arg\max_a\max_{\theta\in\mathcal{C}_t}x(s_t,a)^\top\theta
=\arg\max_a\left[x(s_t,a)^\top\hat\theta_t+\beta_t\sqrt{x(s_t,a)^\top D_t^{-1}x(s_t,a)}\right].
\tag{2.D6.3}
\]

where \(s\) is the state, \(a\) the action, \(x(s,a)\in\mathbb{R}^d\) its feature vector, \(x_i\) the
feature vector of the \(i\)-th observed transition with reward \(r_i\), \(\theta_*\) the unknown
parameter, \(\hat\theta_t\) the ridge estimate, \(D_t\) the feature covariance (normal) matrix,
\(\lambda>0\) the regularization, \(\beta_t\) the confidence radius set by the tail bound, \(c\) the
exploration coefficient, and \(N_t(s,a)\) the number of visits to \((s,a)\).

**Reasoning chain.** In tabular UCB, the bonus in (2.D6.1) is meaningful only because the same
\((s,a)\) is revisited: \(N_t(s,a)\) grows and the interval tightens. In a large or continuous state
space nearly every pair is visited once or not at all. The count is then either zero — undefined, handled
only by the untried-action convention — or one, giving a huge, uninformative bonus. Tabular methods also
share no evidence across states, so nothing learned in one state can guide a similar new one.

Function approximation changes what carries the uncertainty. With rewards modelled as
\(x(s,a)^\top\theta_*\), every observation updates one shared \(\theta\), and \(D_t\) in (2.D6.2)
records how much information has accumulated along each feature direction. Directions with little data
make \(D_t^{-1}\) large, and Cauchy–Schwarz turns that into the prediction uncertainty
\(\beta_t\sqrt{x^\top D_t^{-1}x}\) in (2.D6.3).

OFU says: act as if the unknown parameter were the most favourable one still consistent with the data.
The inner maximization over the ellipsoid \(\mathcal{C}_t\) has exactly the closed form shown in
(2.D6.3) — this is LinUCB, the common tabular/linear instantiation of the optimistic-value principle.
With one-hot features, \(D_t\) is diagonal with entries \(N_t(s,a)+\lambda\), so the square-root term
becomes \(1/\sqrt{N_t(s,a)+\lambda}\) and we recover the count-based bonus of (2.D6.1). LinUCB is a
generalization of UCB1, not a different idea.

**Why it makes sense.** In the one-hot case the ellipsoid is an axis-aligned box around each arm's mean —
literally UCB1's interval. With rich features, data from one state shrinks uncertainty in the shared
direction, so a similar unseen state automatically gets a smaller bonus than a genuinely novel one; that
sharing is the whole point of approximation. The direction in (2.D6.3) must be right: the bonus is larger
along poorly observed directions (large \(x^\top D_t^{-1}x\)) and smaller along well observed ones. If it
were reversed, the agent would keep re-testing familiar states and never explore new ones.

**Check your understanding.** Prediction: using one-hot features reduces (2.D6.3) to tabular UCB up to
the regularization \(\lambda\). Increasing \(\lambda\) shrinks \(D_t^{-1}\), so exploration weakens and the
estimates lean toward zero; with high-dimensional features and little data, \(D_t\) is nearly singular,
bonuses blow up and exploration dominates until regularization bounds it.

### Derived 7 — The "disappointment" dynamic of optimistic initialization [derived]

- **Book location:** Section 2.6, PDF pages 44–45.
- **Type:** Math / Theory.
- **Problem:** Quantify how many steps optimistic initialization drives exploration: with `Q1 = +5`
  and `q*(a) ~ N(0, 1)`, every greedy pick starts above the true value, so the learner switches until
  rewards reveal lower values; estimate how long this lasts and why.
**Hook.** Optimistic initialization is advertised as "automatic exploration", so let's put a number on it.
How many steps does an inflated start actually keep the learner switching arms?

**Formal bridge.** With \(Q_1=+5\) and sample averages, the excess over the true value after \(n\) plays
of an arm is a shrinking fraction of the initial gap:

\[
\mathbb{E}[Q_{n+1}(a)]-q(a)=\frac{Q_1-q(a)}{n+1},
\qquad\text{or}\qquad
(1-\alpha)^n\left(Q_1-q(a)\right)\ \text{with constant }\alpha.
\tag{2.D7.1}
\]

The exploration push fades once that excess is comparable to the reward noise \(\sigma\):

\[
\frac{Q_1-q(a)}{n+1}\lesssim\sigma
\;\Longrightarrow\;
n\gtrsim\frac{Q_1-q(a)}{\sigma},
\qquad
n\gtrsim\frac{1}{\alpha}\ln\!\frac{Q_1-q(a)}{\sigma}\ \text{(constant }\alpha\text{)}.
\tag{2.D7.2}
\]

where \(q(a)\) is the true value of action \(a\) (here drawn from \(\mathcal{N}(0,1)\)), \(Q_1=+5\) is the
optimistic start, \(Q_{n+1}(a)\) is the estimate after \(n\) plays, \(\sigma\) is the reward-noise scale
(about \(1\) for a unit-variance testbed), and \(\alpha\) is a constant step size.

**Reasoning chain.** Every estimate starts at \(+5\), far above arms whose true values are near zero. A
greedy selector therefore keeps meeting disappointment: after one reward \(r\), a tried arm's sample
average drops to roughly \((5+r)/2\), which is below the untouched \(+5\) of an untried arm, so the
learner switches. Each switch spends a little of the initial gap, as shown in (2.D7.1) — the excess is
divided by \(n+1\) for sample averages, or multiplied by \((1-\alpha)^n\) for constant \(\alpha\). The
drive is strongest at the very start and weakens with every observation.

To estimate the length, ask when the inflated estimate stops beating arms that already report honest
values. That happens when the remaining excess in (2.D7.1) is no larger than the noise \(\sigma\) that
makes estimates fluctuate anyway. Setting the excess to \(\sigma\) gives the order-of-magnitude rule in
(2.D7.2): with \(Q_1-q(a)\approx 5\) and \(\sigma\approx 1\), about five pulls per arm — a few tens of
steps across a 10-armed testbed, not thousands. With constant \(\alpha\), the logarithmic expression
stretches the phase by roughly \(1/\alpha\); the precise length depends on the gaps and noise, but it is
always set by how long the initial gap takes to decay.

On a stationary task the process is self-terminating. Once all estimates have dropped to true-value
scale, no arm looks "untested and wonderful" any more, the drive for exploration disappears, and greedy
selection settles down. Optimism is a front-loaded burst, not a permanent exploration policy.

**Why it makes sense.** If \(Q_1\) equalled each arm's true mean, the excess in (2.D7.1) would be zero from
step one: no disappointment, no switching, plain greedy. If \(Q_1\to\infty\), every arm must be sampled
many times before its estimate falls back to reality, giving a long, highly synchronized scan and the
early spikes of Exercise 2.6. The direction is right — more optimism means more exploration — and the
extra exploration must end, because each new reward is unbiased and dilutes a fixed initial error.

**Check your understanding.** Prediction: raise \(Q_1\) from \(+5\) to \(+10\) and the per-arm exploration
length roughly doubles under sample averages, since the excess in (2.D7.2) scales with \(Q_1-q(a)\); set
\(Q_1\) to the true mean and the burst vanishes. Switching to a small constant \(\alpha\) stretches the
phase by about \(1/\alpha\).

### Derived 8 — The tie-breaking rule for untried actions [derived]

- **Book location:** Section 2.7, PDF pages 45–46.
- **Type:** Theory.
- **Problem:** In the book's UCB rule, actions with `N_t(a) = 0` are considered maximizing actions.
  Explain why this rule is necessary and what could go wrong without it.
**Hook.** The UCB formula divides by the number of pulls, so what should it say about an arm nobody has
tried? The book's convention looks like a numerical trick, but it is really a statement about ignorance.

**Formal bridge.** Define the index piecewise so that an untried action is maximising:

\[
U_t(a)=
\begin{cases}
+\infty, & N_t(a)=0,\\[2pt]
Q_t(a)+c\sqrt{\dfrac{\ln t}{N_t(a)}}, & N_t(a)\ge 1,
\end{cases}
\qquad
A_t=\arg\max_a U_t(a).
\tag{2.D8.1}
\]

If instead an untried arm were given a finite default value, it could be skipped forever:

\[
U_t(a)=0\ \text{for }N_t(a)=0
\ \text{and}\
\max_{b:\,N_t(b)>0}Q_t(b)>0
\ \Longrightarrow\ a\ \text{is never selected}.
\tag{2.D8.2}
\]

where \(U_t(a)\) is the UCB index at step \(t\), \(N_t(a)\) is the number of pulls of action \(a\),
\(Q_t(a)\) is its current estimate, \(c\) is the exploration coefficient, and \(A_t\) is the chosen action.

**Reasoning chain.** The index contains \(\sqrt{\ln t/N_t(a)}\), which has no finite value at
\(N_t(a)=0\). The book resolves this by declaring untried actions maximising, as in (2.D8.1). That is more
than a patch for division by zero: an untried arm has no estimate, so assigning it an ordinary number
would silently claim knowledge we do not have. Giving it \(+\infty\) says "unknown means possibly best",
and forces one pull of every arm before any comparison on evidence is allowed.

Without the rule, a tempting fallback is to use the default estimate \(Q=0\) and no bonus. In a testbed
whose true values are positive, tried arms' estimates rise above zero, so the untried arm's index stays at
zero and it is never chosen — exactly (2.D8.2). The learner can then lock onto a suboptimal arm with
linear regret, and the confidence argument of Derived 3, which needs each arm's interval to be valid
after sampling, no longer applies. With the convention, \(N_t(a)\ge 1\) after the initial sweep, the
logarithm is finite, and the ordinary bonus takes over; since \(\ln t\) grows without bound, a neglected
arm is still rescued later (Derived 4). Every action gets at least one trial — the minimum needed to
learn anything about it.

**Why it makes sense.** Picture untried arms as sealed envelopes. An envelope has no measured value, so a
rational explorer treats it as possibly holding the best prize — that is an infinite index, not a zero.
Setting it to zero means asserting "untried arms are worthless", the opposite of what ignorance means.
In the extreme where the best arm is the one never opened, no cleverness with the other arms can recover
its reward, so the tie-break protects against the worst case.

**Check your understanding.** Prediction: replace \(+\infty\) with a large finite index \(M\) and arms are
still tried, but only before arms whose indices exceed \(M\); set \(M\) too low relative to tried arms'
values and some arms are skipped again. A pseudo-count \(N_t(a)=1\) instead gives a bonus \(c\sqrt{\ln t}\),
which grows with \(t\), so every arm is eventually tried — just later than with immediate first-sweep
selection.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 2.1–2.2 | §2.2–2.3 | `chapter_02/solution.py` (`BanditProblem`, `ActionValueEstimator`) | `TestBanditProblem`, `TestActionValueEstimator` | `python -m pytest chapter_02/tests/ -v` |
| 2.3 | §2.3 | `select_epsilon_greedy` | `TestEpsilonGreedy` | `python -m pytest chapter_02/tests/ -v` |
| 2.4 | §2.4 | `ActionValueEstimator`, `simulate_nonstationary` | `TestActionValueEstimator`, `TestNonstationary` | `python -m pytest chapter_02/tests/ -v` |
| 2.5 | §2.5 | `chapter_02/solution.py` (`simulate_nonstationary`) | `TestNonstationary` | `python -m pytest chapter_02/tests/ -v` |
| 2.6 | §2.6 | `simulate_optimistic_initial_values` | `TestOptimisticInitialValues` | `python -m pytest chapter_02/tests/ -v` |
| 2.7 | §2.8 | `chapter_02/solution.py` (`select_softmax`, `GradientBanditAgent`) | `TestGradientBandit` | `python -m pytest chapter_02/tests/ -v` |
| 2.8 | §2.9 | `ContextualBandit`, `simulate_*` | `TestContextualBandit` | `python -m pytest chapter_02/tests/ -v` |
| 2.9 | §2.9 | `chapter_02/solution.py` (`simulate_nonstationary`) | `TestNonstationary` | `python -m pytest chapter_02/tests/ -v` |
| Derived 1–2, 7–8 | §2.6 | `chapter_02/solution.py` (`BanditProblem`, `run_optimistic_greedy_bandit`, `run_epsilon_greedy_bandit`) | `TestBanditProblem`, `TestOptimisticInitialValues`, `TestBanditExplorationRunners` | `python -m pytest chapter_02/tests/ -v` |
| Derived 3–4 | §2.7 | `chapter_02/solution.py` (`ucb_bonus`, `select_ucb`, `run_ucb_bandit`) | `TestUCBBonus`, `TestBanditExplorationRunners` | `python -m pytest chapter_02/tests/ -v` |
| Derived 5–6 | §2.6–2.7 | `chapter_02/solution.py` (`q_learning_ucb`, `q_learning_optimistic`, `q_learning_descending_bonus`) | `TestQLearningExploration`, `TestDescendingBonus` | `python -m pytest chapter_02/tests/ -v` |

- Full chapter command: `python -m pytest chapter_02/tests/ -v`
- Lint: `ruff check chapter_02/`
- No standalone `exercise_*.py` bandit file exists; the chapter implements the bandit machinery in
  `solution.py` (Algorithms 2.1–2.5, plus the folded exploration strategies).

---

## Verification and Quality Notes

- The worked arithmetic in this document (e.g., Ex. 2.1: `P(greedy) = 0.75`) was re-derived by hand;
  the public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`) was used only as a cross-check.
- Programming-exercise answers (2.5, 2.9) describe experiments to run; the underlying code paths are
  mapped to the repository simulations.
- Derived exploration answers were re-derived from first principles; public solution material used
  only as a cross-check.

---

*Exercises document for Chapter 2 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
