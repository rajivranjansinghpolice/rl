# Chapter 3 — Finite Markov Decision Processes: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_03`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_03` — Finite MDPs (Algorithms 3.1–3.4).
- **Book chapter:** Chapter 3 — Finite Markov Decision Processes.
- **Book sections covered by the chapter implementation:**
  - Section 3.1–3.2 — The Agent–Environment Interface; the MDP dynamics `p(s', r | s, a)`.
  - Section 3.3 — The Reinforcement Learning Problem; returns and reward-to-go.
  - Section 3.4 — Policies and Value Functions (softmax/Gibbs preferences).
  - Section 3.5 — Bellman Equations (expectation and optimality; backups and exact solve).
- **Key book equations referenced by the exercises:**
  - Eq. (3.2) — `p(s', r | s, a) = Pr{St+1 = s', Rt+1 = r | St = s, At = a}`.
  - Eq. (3.3) — \(V\pi (s) = \sum _a \pi (a|s) \sum _{s',r} p(s', r | s, a) [r + \gamma V\pi (s')]\).
  - Eqs. (3.8)–(3.10) — discounted return \(Gt = \sum _{k=0}^\infty \gamma ^k Rt+k+1\); episodic return;
    constant-reward return \(G = R / (1 - \gamma )\).
  - Eq. (3.14) — Bellman expectation/optimality equations.
  - Eqs. (3.16)–(3.17) — backup diagrams rooted at a state / action.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** the chapter opens at
  PDF page 40; its exercises and examples were extracted from PDF pages 58–72 (examples 3.1–3.6,
  exercises 3.1–3.26).

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 3. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`,
  `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction` chapter-3 material, plus the public
  chapter-3 solution PDFs. Any disagreement is resolved by independent derivation; discrepancies
  found during cross-checks are noted in the individual exercise entries.
- All solutions in this document are **rewritten from scratch** for student comprehension.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`),
  **Problem**, **Solution**, and — where a companion implementation exists — a **coding-exercise
  pointer** into `chapter_03/`.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations.
- The final section maps exercises to the repository code that already implements (or supports)
  the corresponding machinery.

---

## Exercises

### Exercise 3.1 — Devising example MDP tasks

- **Book location:** Section 3.1 (agent–environment interface), around PDF page 58.
- **Type:** Theory.
- **Problem:** Devise three example tasks of your own that fit into the MDP framework, identifying
  for each its states, actions, and rewards. Make the three examples as different from each other
  as possible. The framework is abstract and flexible, so stretch its limits in at least one of the
  examples.
**Hook.** Let's practice the act of modelling: turning a messy real task into the four things an MDP
asks for. The framework is more flexible than it first looks, so we will deliberately push it.

**Formal bridge.** A task is an MDP once its dynamics can be written as a probability of the next
state and reward given the current state and action,

\[
p(s',r \mid s,a) = \Pr\{S_{t+1}=s',\,R_{t+1}=r \mid S_t=s,\,A_t=a\}.
\tag{3.1.1}
\]

where \(s\) is the current state, \(a\) is the chosen action, \(s'\) is the next state, \(r\) is the
immediate reward, and \(p\) gives the joint probability of \((s',r)\).

**Reasoning chain.** The recipe is always the same four steps: list the states \(S\), list the
actions available in each state, define \(p(s',r \mid s,a)\), and say what return is being maximised.
Once those are written down, the task is an MDP no matter what its subject matter is.

For a **thermostat**, states are room temperature levels (a scalar), actions are {heat, cool,
idle}, and the reward is minus the squared deviation from the target temperature. The action set is
discrete; the state is a number. For a **dog feeding scheduler**, states are {hungry, full,
diabetic?} or a single satiety level, actions are {feed, wait}, and rewards are \(+1\) per
well-timed meal with a small penalty for overfeeding. For a **video-game enemy**, states are
{patrol zone, chase, cooldown} or health/armor scalars, actions are {move, attack, reload}, and
rewards are \(+1\) per hit landed and \(-0.1\) per missed shot. These three differ in time scale,
in whether the state is categorical or numeric, and in whether reward arrives from a physical
process, a schedule, or combat.

To stretch the framework, let temperature be an actual real number, or let the action be a
continuous steering torque. Then \(S\) or \(A\) is no longer finite, so a lookup table cannot hold
\(p\), but (3.1.1) still describes the task: the model is a density rather than a table. That is the
sense in which the MDP frame is abstract and flexible.

**Why it makes sense.** Any of these tasks becomes learnable only when "what happens next" depends
on what the agent did, and when the goal can be expressed as accumulated reward. The thermostat's
negative squared deviation is a compact way of saying "small errors are tolerable, large ones are
expensive", which is exactly the shape a quadratic reward gives. Taking the extreme of a
continuous state shows the boundary of the *tabular* methods, not of the MDP idea itself.

**Check your understanding.** Prediction: replace the thermostat's discrete actions with a
continuous heat setting. Equation (3.1.1) is unchanged, but the agent can no longer compare a
finite list of actions, so it needs a policy that outputs a number and a way to generalise across
settings.

### Exercise 3.2 — Is the MDP framework adequate?

- **Book location:** Section 3.1, around PDF page 58.
- **Type:** Theory.
- **Problem:** Is the MDP framework adequate to usefully represent all goal-directed learning
  tasks? Can you think of any clear exceptions?
**Hook.** It is tempting to assume every goal-directed task is an MDP. Let's test that assumption by
looking for tasks where the state-and-reward description breaks down.

**Formal bridge.** The MDP promise is that the current state and action contain everything needed
to predict the next outcome and reward, and that the goal is an additive return,

\[
p(s',r \mid s,a) = \Pr\{S_{t+1}=s',R_{t+1}=r \mid \text{full history}, S_t=s, A_t=a\},
\qquad
J(\pi)=\mathbb{E}\!\left[\sum_{k\ge 0}\gamma^k R_{t+k+1}\right].
\tag{3.2.1}
\]

where \(s,a,s',r\) are state, action, next state and reward, \(\pi\) is the policy, and \(\gamma\) is
the discount factor.

**Reasoning chain.** For the large majority of goal-directed sequential problems the answer is
yes. If the goal can be written as a sum of rewards along a trajectory and the dynamics depend only
on the current state and action, then (3.2.1) is exactly the right language, and value functions
and Bellman equations apply.

The exceptions are instructive. Tasks with **intrinsic motivation** — curiosity or novelty bonuses
— add reward signals that depend on the learner's own knowledge or history, which a stationary
external \(p(s',r \mid s,a)\) does not supply. **Multi-agent** credit assignment breaks the single
learner's perspective: one agent cannot cleanly write a scalar reward that explains another
agent's contribution, so the joint dynamics and rewards need a game-theoretic frame. Tasks where
the agent must **discover its own action effects**, as in meta-learning or structure exploration,
violate the assumption that \(p\) is already fixed and stationary. Finally, with **partial
observability**, the observable \(S_t\) does not summarise the past, so the Markov equality in
(3.2.1) fails unless the state is enlarged to include history or a belief.

Notice that several exceptions are really "plain MDP plus one ingredient", not a rejection of the
framework: add the belief state and the problem is again an MDP, just over a bigger state space;
add explicit second-agent rewards and one has a multi-agent MDP. Recognising which ingredient is
missing is part of the modelling skill.

**Why it makes sense.** Consider the extreme of a partially observed task where the only useful
information is what happened two steps ago. If \(S_t\) omits it, two histories with the same \(S_t\)
demand different actions, so no policy \(\pi(a \mid s)\) can be optimal for both — the MDP promise
cannot hold, and that is precisely the failure mode we listed. Conversely, once that history is
folded into the state, the Markov condition is restored: the representation, not the formalism,
was the problem.

**Check your understanding.** Prediction: give the agent a memory channel that records the last two
observations. The formerly partially observable task becomes an MDP over an enlarged state, so the
Bellman machinery applies again, at the cost of many more states.

### Exercise 3.3 — Where to draw the agent–environment line?

- **Book location:** Section 3.1, around PDF page 58–59.
- **Type:** Theory.
- **Problem:** For driving, actions could be defined at the accelerator/steering/brake level, at
  the tire-torque level, at the muscle level, or at the "where to drive" policy level. Where is the
  right place to draw the line?
**Hook.** Before writing \(p(s',r \mid s,a)\) we must decide what counts as an action. Driving makes
this decision vivid, because the same behaviour can be described at four very different levels.

**Formal bridge.** The division is acceptable exactly when the chosen state and action restore the
Markov property,

\[
\Pr\{S_{t+1}=s',R_{t+1}=r \mid S_t,A_t\} =
\Pr\{S_{t+1}=s',R_{t+1}=r \mid \text{all prior events}\}.
\tag{3.3.1}
\]

where \(S_t\) and \(A_t\) are the state and action at the chosen level of abstraction.

**Reasoning chain.** There is no fundamental answer: any consistent level where (3.3.1) holds and
the reward is meaningful is a valid MDP. The choice is a modelling tradeoff. At the muscle or
tire-torque level the agent has fine control, but the state must now include body dynamics, wheel
slip and road friction, so the dynamics are high-dimensional. At the accelerator/steering/brake
level, the car's controllers absorb the low-level detail, the state stays manageable, and the
actions still map directly onto driving.

At the "where to drive" policy level, the reward is cleanest — destination reached, traffic
rules obeyed — but control is coarse: the learner no longer decides how to execute a route, only
which route to request. Each level simply moves part of the control problem from the agent into
the environment, which is why the book describes the line as a modelling choice.

The practical rule follows from (3.3.1): put the boundary where the learner can actually act and
where the remaining dynamics are tractable enough to model. If pushing the boundary down makes the
state explode, stop; if pushing it up hides a decision the agent must learn, stop earlier.

**Why it makes sense.** Take the extreme of muscle-level control: the agent can in principle
produce any steering motion, but it must now learn biomechanics and tire physics before it can
learn to drive — the extra expressiveness is paid for with enormous state and data. Take the
opposite extreme of a pure destination selector: learning is cheap but the agent cannot react to a
skid. Neither extreme is "more correct"; the right level is where the Markov description is both
true and small enough to use.

**Check your understanding.** Prediction: move the boundary from steering-wheel angle down to tire
torque. Equation (3.3.1) can be restored only by adding slip and friction to the state, so control
improves while the number of states and the sample requirement grow sharply.

### Exercise 3.4 — Expected one-step reward

- **Book location:** Section 3.2 (equation 3.2), PDF page 59.
- **Type:** Computation.
- **Problem:** If the current state is `St` and actions are selected according to a stochastic
  policy \(\pi\), what is the expectation of `Rt+1` in terms of the four-argument function
  `p(s', r | s, a)`?
**Hook.** We often need just the immediate expected reward, without tracking where the next state
lands. Let's extract it from the full dynamics by summing out the things we do not care about.

**Formal bridge.** Averaging first over the world's outcome and then over the policy's action gives

\[
\mathbb{E}[R_{t+1}\mid S_t=s]
= \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\,r,
\tag{3.4.1}
\]

with the inner sum alone giving \(\mathbb{E}[R_{t+1}\mid S_t=s,A_t=a]\).

where \(\pi(a\mid s)\) is the probability of choosing action \(a\) in state \(s\), \(p(s',r\mid s,a)\)
is the dynamics, \(r\) is the immediate reward, and \(s'\) is the next state that gets summed out.

**Reasoning chain.** The law of total expectation says: condition on the action, average over all
possible \((s',r)\) pairs weighted by \(p\), then average over the actions chosen by \(\pi\). Fixing
\(a\), the conditional expectation is \(\sum_{s',r}p(s',r\mid s,a)\,r\); every term is a reward value
times the probability of the outcome that produced it.

For the recycling robot in state `high` taking `search`, the original calculation is
\(\alpha\,r_{\text{search}} + (1-\alpha)\,r_{\text{search}} = r_{\text{search}}\): the expected
reward while searching is the expected number of cans collected times the unit reward per can.
Notice that the state outcome (`high` or `low`) does not change this reward term in that example;
the probability \(\alpha\) appears in the transition part of \(p\) rather than in the reward.

This quantity is the building block of every tabular backup: the Bellman equations replace the
raw reward \(r\) with exactly this expectation, then add the discounted expected future value.

**Why it makes sense.** Think of the extreme of a deterministic policy: the outer sum collapses to
the single chosen action, so the expectation is just that action's expected reward. Think also of a
reward that does not depend on \(s'\): then summing over \(s'\) merely redistributes the same \(r\)
across outcomes and leaves the answer unchanged. The direction is intuitive — averaging cannot
produce a value outside the range of rewards the chosen actions can actually yield.

**Check your understanding.** Prediction: make \(\pi\) greedy instead of stochastic. The outer sum
in (3.4.1) becomes a single term for the maximising action; if instead the reward were identical
for all actions, the policy would stop mattering altogether.

### Exercise 3.5 — Dynamics table for the recycling robot

- **Book location:** Section 3.2 (Table 3.1), PDF page 59.
- **Type:** Computation.
- **Problem:** Give a table analogous to Table 3.1, but for `p(s', r | s, a)`, with columns for
  `s, a, s', r, p(s', r | s, a)`, and a row for every combination with `p > 0`.
**Hook.** Here the abstract \(p(s',r \mid s,a)\) becomes a concrete instruction manual for the
recycling robot. Writing out every positive-probability row makes the model testable.

**Formal bridge.** Each row of the table assigns a probability to one triple \((s',r)\) under a
given \((s,a)\),

\[
p(s',r \mid s,a) = \Pr\{S_{t+1}=s',\,R_{t+1}=r \mid S_t=s,\,A_t=a\},
\qquad \sum_{s',r} p(s',r \mid s,a)=1.
\tag{3.5.1}
\]

where \(\alpha\) is the probability of staying `high` after `search` from `high`, \(\beta\) is the
probability of staying `low` after `search` from `low` (so rescue probability is \(1-\beta\)),
\(r_{\text{search}}\) and \(r_{\text{wait}}\) are the corresponding rewards, and the penalty for a
rescue is \(-3\).

**Reasoning chain.** Read the table row by row. From `high`, `search` keeps the robot `high` with
probability \(\alpha\) or drops it to `low` with probability \(1-\alpha\), collecting
\(r_{\text{search}}\) either way. From `low`, `search` either survives with probability \(\beta\)
and collects \(r_{\text{search}}\), or the battery dies with probability \(1-\beta\), the robot is
rescued to `high`, and the reward is \(-3\). `wait` is deterministic: `high` stays `high` and `low`
stays `low`, both with reward \(r_{\text{wait}}\). `recharge` from `low` deterministically returns
the robot to `high` with reward \(0\).

For each fixed \((s,a)\), the listed probabilities sum to one over next states and rewards — for
example \(\alpha+(1-\alpha)=1\) and \(\beta+(1-\beta)=1\). Rows with zero probability are omitted
because they can never occur; leaving them out is exact bookkeeping, not an approximation. That is
what makes the table a complete dynamics specification.

**Why it makes sense.** Look at the extremes of the two parameters. If \(\alpha=1\), searching from
`high` can never drain the battery, so the `high → low` row disappears in effect. If \(\beta=1\),
searching from `low` never needs rescue, so the \(-3\) outcome never occurs. The table's probabilities
directly encode how risky the robot judges searching to be; when the risk parameters move, the
expected reward and the optimal action move with them in the obvious direction.

**Check your understanding.** Prediction: set \(\beta=0\), so a low battery always triggers rescue.
The `low → low` search row vanishes, the \(-3\) outcome occurs on every low-search, and the robot
should prefer `wait` or `recharge` until it is high again.

**Where this shows up in the code.** `FiniteMDP.expected_reward` and `FiniteMDP.transition_probs`
in `chapter_03/solution.py`; tests `TestFiniteMDP.test_expected_reward` and
`TestFiniteMDP.test_transition_probs`.

### Exercise 3.6 — Bellman equation for episodic tasks

- **Book location:** Section 3.3 (equations 3.3–3.9), PDF page 62–63.
- **Type:** Math.
- **Problem:** The equations in Section 3.1 (agent–environment interface) and Section 3.3 (returns)
  are written for the continuing case and need slight modification to apply to episodic tasks. Give
  the modified version of (3.3).
**Hook.** Episodic tasks are not a new theory; they are the same equations with a stopping rule. Let's
pin down the one modification that makes the return finite.

**Formal bridge.** With episode length \(T\), the return is a finite sum with \(G_T=0\), and the
Bellman expectation equation keeps its shape but assigns terminal states value zero:

\[
G_t = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1},
\qquad
v_\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_\pi(s')\right].
\tag{3.6.1}
\]

where \(T\) is the terminal time, \(G_T=0\) is the terminal return, \(\gamma\) is the discount
factor, \(\pi\) is the policy, and \(v_\pi(s')=0\) whenever \(s'\) is terminal.

**Reasoning chain.** In the continuing case the return is an infinite discounted sum, and the
Bellman equation follows from \(G_t = R_{t+1}+\gamma G_{t+1}\). For an episode of length \(T\), set
\(G_T=0\); unrolling the same recursion from \(t\) stops after \(T-t\) rewards, giving the finite sum
in (3.6.1). Nothing else about the dynamics changes: \(p(s',r \mid s,a)\) is still the same joint
distribution.

The modification to (3.3) is therefore a boundary condition, not a new operator. When the backup
lands in a terminal state, the continuation term \(\gamma v_\pi(s')\) is zero because there are no
more rewards; elsewhere the expression is exactly the continuing one. The recursion \(G_t =
R_{t+1}+\gamma G_{t+1}\) still holds for every non-terminal \(t\), which is why the same fixed-point
equations can be solved by the same methods.

**Why it makes sense.** Take \(\gamma=1\): the finite sum becomes the total reward until
termination, which is finite precisely because the episode stops. Now take the opposite extreme of
forgetting to set terminal value to zero: the backup would keep adding a fictitious future beyond
the episode, overvaluing states near the exit. The zero boundary condition is what lets the
same algebraic identity describe both continuing and episodic returns.

**Check your understanding.** Prediction: shorten the episode (smaller \(T\)). Fewer rewards enter
the sum in (3.6.1), so values near the start drop; letting \(T\to\infty\) recovers the continuing
discounted return.

### Exercise 3.7 — Pole-balancing: episodic return with discounting

- **Book location:** Section 3.3 (equations 3.7–3.10), PDF page 62.
- **Type:** Computation.
- **Problem:** Treat pole-balancing episodically but also use discounting; all rewards zero except
  \(-1\) upon failure. What is the return at each time, and how does it differ from the discounted,
  continuing formulation?
**Hook.** Here we combine two mechanisms that might seem to compete — discounting and episode
termination. With a single delayed penalty we can see exactly how they interact.

**Formal bridge.** Compare the two return definitions,

\[
G_t^{\text{ep}} = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1},
\qquad
G_t^{\text{cont}} = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
\tag{3.7.1}
\]

When the only nonzero reward is \(-1\) at failure, \(m\) steps after \(t\), both reduce to
\(G_t=-\gamma^{m}\).

where \(T\) is the failure time, \(m=T-t\) counts the steps from \(t\) to the failure, \(\gamma\) is
the discount factor, and all rewards before and after the failure are zero.

**Reasoning chain.** In the episodic formulation, the sum in (3.7.1) stops at \(T-1\). The only
contributing term is the failure reward \(-1\) at offset \(m\), so \(G_t=\gamma^m(-1)=-\gamma^m\).
Before failure, every step's return is the discounted value of the approaching penalty; at and
after failure the return is zero.

In the continuing formulation, the sum is infinite, but every reward after failure is zero, so
the extra tail contributes nothing. The same single term survives and gives the same
\(-\gamma^m\). Thus the two formulations agree on the numbers: discounting and termination are
orthogonal — one weights rewards by distance, the other decides where rewards stop.

The difference is only in horizon handling. The episodic sum is explicitly finite with \(G_T=0\),
so states after failure have no value to compute; the continuing sum is formally infinite and
relies on the zero rewards and discounting to converge. In this example the two bookkeeping
conventions therefore produce identical returns.

**Why it makes sense.** Consider \(\gamma=1\): then \(G_t=-1\) no matter how far away failure is,
because an undiscounted penalty is felt in full; both formulations agree. With \(\gamma<1\), the
further away the failure the smaller \(-\gamma^m\), which is the correct sign and direction:
avoiding failure for longer makes the current situation look better. If discounting made distant
penalties *larger*, the agent would be perversely attracted to imminent failure — the exponential
decay is exactly what prevents that.

**Check your understanding.** Prediction: add a reward of \(+1\) on every surviving step. The
continuing return gains the infinite tail \(1/(1-\gamma)\), while the episodic return gains only the
finite surviving prefix; the two are no longer numerically identical, because termination now
truncates nonzero rewards.

### Exercise 3.8 — Maze task and stagnation

- **Book location:** Section 3.3, PDF page 63.
- **Type:** Theory.
- **Problem:** Maze robot gets `+1` for escaping, `0` otherwise; treated as episodic; after training
  shows no improvement. What is going wrong?
**Hook.** A agent that receives a reward for escaping but never improves is a debugging puzzle.
Let's look at the specification of the objective rather than at the learning algorithm.

**Formal bridge.** The episodic objective is to maximise the expected return

\[
\mathbb{E}[G_t] = \mathbb{E}\!\left[\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}\right],
\tag{3.8.1}
\]

where the \(+1\) reward occurs at escape and is zero at every other step, and \(T\) is the episode's
terminal time.

where \(G_t\) is the episodic return, \(T\) is termination, \(\gamma\) is the discount factor, and
\(R_{t+k+1}\) is the reward on each step.

**Reasoning chain.** The reward specification looks adequate on paper: \(+1\) for escaping, \(0\)
otherwise, with a finite episode. But the agent can only optimise (3.8.1) if it knows the episode
boundaries and treats the escape reward as the terminal signal. If termination is not signalled,
the learner treats the final transition as just another step; if it frames the problem as
undiscounted while the implementation discounts (or vice versa), the values it learns do not
correspond to the objective it is scored on.

With only a sparse \(+1\) at the end, an objective/termination mismatch is especially damaging:
there is no intermediate reward to accidentally guide the agent toward the exit, so if the terminal
signal is misaligned there is literally nothing else to learn from. The result is stagnation — no
systematic improvement across training episodes. The fix is to make the objective precise: tell the
learner where an episode ends, which reward is terminal, and which discount convention is in use.

**Why it makes sense.** Imagine the extreme where the escape reward is delivered *after* the
episode is declared finished. From the agent's point of view the terminal successor state has value
zero and the \(+1\) never enters any backup, so every action looks equally worthless — exactly the
observed stagnation. The direction is clear: a misplaced terminal signal deletes the only
informative reward, whereas a correctly placed one propagates backward one step at a time through
the value function.

**Check your understanding.** Prediction: add a small cost of \(-0.01\) per step while keeping the
\(+1\) at escape. Once termination is signalled correctly, the agent now has a reason to escape
quickly, so improvement should appear; if termination is still mis-specified, the per-step cost
alone will not guide it toward the exit.

### Exercise 3.9 — Return computation from reward sequences

- **Book location:** Section 3.3 (equations 3.8–3.10), PDF page 63.
- **Type:** Computation.
- **Problem:** With \(\gamma = 0.5\), rewards `R1 = 1, R2 = 2, R3 = 6, R4 = 3, R5 = 2`, `T = 5`, find
  `G0, G1, …, G5`. Hint: work backwards.
**Hook.** A return is "reward now plus a discounted copy of the future". Starting at the end of the
episode, where the future is worth zero, makes every return a one-line calculation.

**Formal bridge.** The backward recursion is

\[
G_t = R_{t+1} + \gamma G_{t+1}, \qquad G_T = 0.
\tag{3.9.1}
\]

where \(G_t\) is the return from time \(t\), \(R_{t+1}\) is the reward received after \(t\),
\(\gamma=0.5\) is the discount factor, and \(T=5\) is the terminal time.

**Reasoning chain.** Start with \(G_5=0\), since no rewards follow termination. Then walk backwards:
\(G_4=R_5+0.5G_5=2\); \(G_3=R_4+0.5G_4=3+1=4\); \(G_2=R_3+0.5G_3=6+2=8\);
\(G_1=R_2+0.5G_2=2+4=6\); \(G_0=R_1+0.5G_1=1+3=4\). So
\(G=[4,6,8,4,2,0]\) for \(t=0,\ldots,5\). Each step uses only the reward immediately ahead and
the already computed next return, which is exactly the trick the hint points to.

**Why it makes sense.** Check the extreme \(\gamma=0\). Then \(G_t=R_{t+1}\) and the answer would
be \([1,2,6,3,2,0]\), with no influence from later rewards. At \(\gamma=0.5\), the large reward
\(R_3=6\) contributes \(6\) to \(G_2\), \(3\) to \(G_1\), and only \(1.5\) to \(G_0\): the same reward
is progressively diluted the further away it is. The direction of (3.9.1) is therefore the only
sensible one — a reward's influence must shrink, not grow, with distance.

**Check your understanding.** Prediction: raise \(\gamma\) toward \(1\). Later rewards carry more
weight, so \(G_0\) rises toward the undiscounted total \(1+2+6+3+2=14\); at \(\gamma=0\) it falls
to the immediate reward \(1\).

**Where this shows up in the code.** `compute_return` and `compute_returns_to_go` in
`chapter_03/solution.py`; tests `TestComputeReturn`.

### Exercise 3.10 — Infinite constant-reward return

- **Book location:** Section 3.3 (equation 3.10), PDF page 63.
- **Type:** Computation.
- **Problem:** With \(\gamma = 0.9\) and reward sequence `R1 = 2` followed by an infinite sequence of `7`s,
  find `G1` and `G0`.
**Hook.** An infinite sum sounds intimidating, but a constant tail collapses to one fraction.
Compute the tail first, then step back one reward to get the earlier return.

**Formal bridge.** A constant reward \(R\) forever has return

\[
G_{\text{tail}} = \sum_{k=0}^{\infty}\gamma^k R = \frac{R}{1-\gamma},
\tag{3.10.1}
\]

and the one special reward is attached by \(G_0=R_1+\gamma G_1\).

where \(R=7\) is the constant tail reward, \(\gamma=0.9\), \(R_1=2\) is the unusual first reward, and
\(G_1\) is the return from step 1 onward.

**Reasoning chain.** From time 1 onward every reward is \(7\), so (3.10.1) gives
\(G_1=7/(1-0.9)=7/0.1=70\). The reward at time 1 is not part of \(G_0\)'s tail directly; it enters
as the single immediate term, discounted only once in \(G_0\): \(G_0=R_1+\gamma G_1=2+0.9\times 70=
2+63=64\). Notice that \(G_0\) is smaller than \(G_1\) here — the first reward, \(2\), is below the
steady rate \(7\), so it drags the earlier return down.

The ordering of the computation matters: summing from \(t=0\) directly would require keeping track of
the special reward separately; peeling it off with the recursion reduces everything to the
geometric-series formula.

**Why it makes sense.** At \(\gamma=0\), (3.10.1) gives \(G_1=7\) and \(G_0=2\), the immediate
rewards only. As \(\gamma\to1\), every future \(7\) counts almost fully, so \(G_1\) grows without
bound; the first reward's contribution \(\gamma\times2\) stays at \(2\), which is why \(G_1-G_0\)
tends to a constant offset. The direction is right: a poor first reward cannot change the infinite
tail, it only shifts the earlier return by a discounted amount.

**Check your understanding.** Prediction: change the constant tail from \(7\) to \(8\). Then
\(G_1=80\) and \(G_0=2+0.9\times80=74\), so both returns rise by ten while the gap between them
stays at six.

### Exercise 3.11 — Proving the constant-reward return

- **Book location:** Section 3.3 (equation 3.10), PDF page 63.
- **Type:** Proof.
- **Problem:** Prove equation (3.10): \(G = R / (1 - \gamma )\).
**Hook.** Let's prove the formula behind the previous exercise, so we know exactly when it is valid.
It is the geometric series in disguise.

**Formal bridge.** For a constant reward \(R\), the return is

\[
G = \sum_{k=0}^{\infty}\gamma^k R = R\sum_{k=0}^{\infty}\gamma^k = \frac{R}{1-\gamma},
\qquad 0\le\gamma<1.
\tag{3.11.1}
\]

where \(R\) is the reward received at every step, \(\gamma\) is the discount factor, and the
condition \(\gamma<1\) makes the geometric series converge.

**Reasoning chain.** Factor the constant \(R\) out of the sum. The remaining series
\(1+\gamma+\gamma^2+\cdots\) has a standard closed form: multiply it by \((1-\gamma)\), and every
term cancels with the next, leaving \(1\). Hence the series equals \(1/(1-\gamma)\) and (3.11.1)
follows immediately. Equivalently, write \(G=R+\gamma G\) because the tail after the first reward
is a discounted copy of the whole return; solving gives \(G(1-\gamma)=R\).

The condition \(0\le\gamma<1\) is not cosmetic. If \(\gamma=1\), the partial sums grow as \(nR\)
and never settle; if \(\gamma>1\), they diverge faster. Only when each later copy is strictly
smaller does the infinite stack have a finite total.

**Why it makes sense.** The formula says "the same reward forever, with each later copy worth
\(\gamma\) times less" — a picture of endlessly shrinking contributions. At \(\gamma=0.5\) the
total is \(2R\): the first reward plus an equal amount from all later ones combined. As \(\gamma\)
approaches \(1\), the shrinking slows and the total blows up, which is exactly why the formula is
meaningful only below \(1\).

**Check your understanding.** Prediction: set \(R=1\) and \(\gamma=0.5\). Equation (3.11.1) gives
\(G=2\); raising \(\gamma\) to \(0.9\) gives \(G=10\), showing how sensitive a long future is to the
discount factor.

### Exercise 3.12 — Bellman equation check on the gridworld

- **Book location:** Section 3.5 (value functions and Bellman equations), PDF page 66.
- **Type:** Computation.
- **Problem:** Show numerically that the Bellman equation holds for the center state of the
  gridworld (Example 3.6), valued at `+0.7`, with respect to its four neighbouring states (values
  `+2.3, +0.4, −0.4, +0.7).
**Hook.** The printed gridworld values should satisfy a consistency check: a state's value must
equal the average of "immediate reward plus discounted value of where you land". Let's verify that
at the center.

**Formal bridge.** Under the equiprobable policy, the Bellman expectation backup is

\[
v'(s_0) = \frac{1}{4}\sum_a \left[r_a + \gamma v(s'_a)\right]
        = \frac{1}{4}\sum_a r_a + \gamma\cdot\frac{1}{4}\sum_a v(s'_a).
\tag{3.12.1}
\]

where \(s_0\) is the center state, \(a\) ranges over the four moves, \(s'_a\) and \(r_a\) are the
successor state and reward for move \(a\), \(\gamma=0.9\), and the four neighbour values are
\(2.3,\,0.4,\,-0.4,\,0.7\).

**Reasoning chain.** Split the backup into two averages, as in (3.12.1). The neighbour values sum
to \(2.3+0.4-0.4+0.7=3.0\), so their average is \(3.0/4=0.75\), and the discounted future term is
\(0.9\times0.75=0.675\). The reward term is \(\tfrac14\sum_a r_a\), determined by the gridworld's
reward structure. With that reward contribution the one-step backup equals the listed value
\(v(s_0)=0.7\): the equality is exactly the fixed-point condition the Bellman equation asserts.

This is a numerical consistency check, not a new computation of the whole grid: if the printed
value at the center differed from its backup, either the values or the dynamics in the figure would
be inconsistent. The same condition can be checked everywhere by computing a Bellman residual,
which should vanish at the solution.

**Why it makes sense.** Think of the extreme \(\gamma=0\): the backup would reduce to the average
immediate reward and the neighbours would stop mattering at all. With \(\gamma=0.9\), the
neighbours dominate, so a wrong neighbour value propagates into the center with weight \(0.9/4\) per
state. The positive \(2.3\) and the negative \(-0.4\) partly cancel, which is why the average is a
modest \(0.75\) — a picture of four directions whose good and bad outcomes largely offset.

**Check your understanding.** Prediction: raise one neighbour's value by \(0.4\) while leaving
rewards fixed. Equation (3.12.1) predicts the backup rises by \(0.9\times0.4/4=0.09\), so the
center's value must rise by the same amount to remain a Bellman fixed point.

**Where this shows up in the code.** `bellman_expectation_residual` and `verify_bellman` in
`chapter_03/solution.py`; test `TestBellman.test_expectation_residual_zero_on_solution`.

### Exercise 3.13 — Bellman equation for action values

- **Book location:** Section 3.5 (equations 3.14–3.17), PDF page 66.
- **Type:** Math.
- **Problem:** What is the Bellman equation for action values \(q\pi (s, a)\)? Give the sequence of
  equations analogous to (3.14) but for action values.
**Hook.** Action values fix the first action, so one layer of averaging disappears. Let's write the
backup that remains and see where the policy re-enters.

**Formal bridge.** The action-value Bellman equation is

\[
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_\pi(s')\right],
\tag{3.13.1}
\]

and substituting \(v_\pi(s')=\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\) gives

\[
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].
\tag{3.13.2}
\]

where \(\pi(a'\mid s')\) is the policy's probability of the next action \(a'\) in state \(s'\).

**Reasoning chain.** Compare with the state-value backup: at a state, the policy chooses the first
action, so there is an outer average over \(a\). Here \(a\) is fixed, so the outer average is over
what the world does — the pair \((s',r)\) — and there is no \(\pi(a\mid s)\) factor.

Once the world has produced \(s'\), the policy resumes control, and that is where \(\pi\) reappears
inside (3.13.2): the continuation value \(v_\pi(s')\) is itself an average over the next actions
\(a'\), weighted by \(\pi(a'\mid s')\) and expressed in action-value form as
\(q_\pi(s',a')\). The two equations are the same backup seen from different roots of the backup
diagram: state-rooted versus action-rooted.

**Why it makes sense.** Take the extreme of a deterministic policy \(\pi(a'\mid s')=1\) for one
action: the inner sum in (3.13.2) collapses to a single \(q_\pi\) value, and the action value becomes
a plain expectation over outcomes. Take \(\gamma=0\): the continuation disappears entirely and
\(q_\pi(s,a)\) is just the expected immediate reward. Both limits are exactly what "fix the first
action, then act under \(\pi\)" should give.

**Check your understanding.** Prediction: replace \(\pi\) by the greedy policy with respect to
\(q_\pi\). The inner average becomes a maximum over \(a'\), turning (3.13.2) into the optimality
equation for \(q_*\).

### Exercise 3.14 — Value shift from a constant reward addition

- **Book location:** Section 3.5, PDF page 66.
- **Type:** Math.
- **Problem:** Adding a constant `c` to all rewards adds a constant `v_c` to all state values; what
  is `v_c` in terms of `c` and \(\gamma\)?
**Hook.** Suppose we pay the agent \(c\) extra at every single step. What happens to the numbers on
the value grid, and what happens to the optimal policy?

**Formal bridge.** For continuing tasks the shift in all state values is

\[
v_c = \frac{c}{1-\gamma},
\tag{3.14.1}
\]

because adding \(c\) to every reward adds \(c\) at each of the infinitely many discounted future
steps.

where \(c\) is the constant added to every reward, \(\gamma\) is the discount factor, and \(v_c\) is
the uniform amount added to every state value.

**Reasoning chain.** Substitute the shifted reward \(r+c\) and shifted value \(v+c/(1-\gamma)\) into
the Bellman equation. The extra terms contribute \(c\) now plus \(\gamma\cdot c/(1-\gamma)\)
later, which sums to \(c(1-\gamma+\gamma)/(1-\gamma)=c/(1-\gamma)\) — the same shift we assumed,
so the equation is satisfied by the shifted values. That proves (3.14.1) is the right constant.

The crucial point is that the shift is the *same* for every state and every action. Since all
action values move by an identical amount, their ranking is untouched, so optimal relative values
and the optimal policy are unchanged. Adding a constant to all rewards changes the reported level
of return but not the decision problem.

**Why it makes sense.** At \(\gamma=0\), only the immediate reward counts, so the shift is just
\(c\). As \(\gamma\to1\), future steps matter almost as much as the present, so the accumulated
extra payments grow without bound and the shift diverges. The direction is necessarily positive
for \(c>0\): paying the agent more at every step cannot make a state look worse.

**Check your understanding.** Prediction: pay \(c\) only for one specific action instead of every
reward. That action's values shift but the others do not, so the relative ranking changes and the
optimal policy can change — unlike the uniform-shift case in (3.14.1).

### Exercise 3.15 — Constant reward shift in an episodic task

- **Book location:** Section 3.5, PDF page 66.
- **Type:** Math.
- **Problem:** In an episodic task (e.g., maze running), would adding a constant `c` to all rewards
  have any effect, or leave the task unchanged?
**Hook.** The infinite-horizon shift relied on infinitely many extra payments. In an episode only
\(T\) payments fit — so let's recount.

**Formal bridge.** Over an episode of length \(T\), the extra \(c\) at each step contributes

\[
\Delta G = c\sum_{k=0}^{T-1}\gamma^k = c\,\frac{1-\gamma^T}{1-\gamma}.
\tag{3.15.1}
\]

where \(T\) is the episode length, \(c\) is the constant added to every reward, and \(\gamma\) is the
discount factor.

**Reasoning chain.** The geometric series now stops after \(T\) terms, so the infinite-horizon
result \(c/(1-\gamma)\) is multiplied by \(1-\gamma^T\). For a fixed episode length, every complete
trajectory receives exactly the same shift \(\Delta G\), so the ranking of policies is unchanged:
only the reported return level moves. The optimal policy is therefore unaffected, just as in the
continuing case.

The fixed-length qualifier matters. If different policies induce different episode lengths, the
shift depends on \(T\), so a policy that lingers collects more of the extra payments. In that case
(3.15.1) no longer gives a uniform shift and the policy ranking can change — a useful caveat when
applying the idea to a maze where the agent controls when it exits.

**Why it makes sense.** At \(T=1\) the shift is just \(c\), since only one reward is paid. As
\(T\to\infty\), \(\gamma^T\to0\) and (3.15.1) approaches the continuing answer \(c/(1-\gamma)\),
showing that the episodic case is the finite-horizon version of the same phenomenon. The shift is
positive for \(c>0\) and grows with \(T\): more steps mean more extra payments, exactly as intuition
suggests.

**Check your understanding.** Prediction: let episodes have variable length and add the same \(c\)
everywhere. Longer episodes now get a larger shift from (3.15.1), so the agent may prefer to dawdle;
the policy-neutrality argument only survives when all trajectories share the same horizon.

### Exercise 3.16 — State backup via the stochastic policy

- **Book location:** Section 3.5 (backup diagrams), PDF page 66.
- **Type:** Math.
- **Problem:** Give the Bellman equation for \(v\pi (s)\) in terms of the actions available in `s` and
  the probabilities \(\pi (a|s)\).
**Hook.** A state-rooted backup has two layers of chance: the policy picks an action, then the world
picks an outcome. The Bellman equation is just the average over both.

**Formal bridge.**

\[
v_\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_\pi(s')\right].
\tag{3.16.1}
\]

where \(\pi(a\mid s)\) is the probability of action \(a\) in state \(s\), \(p(s',r\mid s,a)\) is the
dynamics, \(r\) is the immediate reward, \(v_\pi(s')\) is the successor state's value, and \(\gamma\)
discounts only the future, not the immediate reward.

**Reasoning chain.** Read (3.16.1) from left to right as the backup diagram: from \(s\), the policy
branches over actions with probability \(\pi(a\mid s)\); for each chosen action, the world branches
over \((s',r)\) with probability \(p(s',r\mid s,a)\). Each leaf contributes its reward plus the
discounted value of its next state. Summing all leaves with their probabilities gives the expected
return from \(s\) under \(\pi\).

There is no maximisation here because \(\pi\) is fixed: we average over its actions rather than
choosing the best one. Discounting applies to \(v_\pi(s')\) only — the immediate reward is received
now, so it is not scaled. This is the equation whose fixed point is the state-value function for
any given policy.

**Why it makes sense.** With a deterministic policy, the outer sum collapses to one action and the
equation becomes an average over the world's outcomes alone. With \(\gamma=0\), the continuation term
vanishes and the value is just the expected next reward. Increasing \(\gamma\) gives later states
more influence; if the future term were given more weight than the reward, the backup would misprice
short-term consequences.

**Check your understanding.** Prediction: move from the fixed policy \(\pi\) to the greedy policy
with respect to \(v\). The outer average over \(\pi(a\mid s)\) becomes a maximum over \(a\), giving the
optimality equation instead of (3.16.1).

### Exercise 3.17 — Action backup for \(q\pi (s, a)\)

- **Book location:** Section 3.5 (backup diagrams), PDF page 66.
- **Type:** Math.
- **Problem:** Give the equation for the action value \(q\pi (s, a)\) in terms of the expected next reward
  and expected next state value, with an explicit (non-expected-value) formulation.
**Hook.** The action backup can be written as one expectation or split into two familiar pieces. Let's
do both so we can see they are the same statement.

**Formal bridge.** Combining reward and successor in one expectation,

\[
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_\pi(s')\right],
\tag{3.17.1}
\]

and separating them using \(p(s'\mid s,a)=\sum_r p(s',r\mid s,a)\),

\[
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\,r
           + \gamma\sum_{s'} p(s'\mid s,a)\,v_\pi(s').
\tag{3.17.2}
\]

where \(p(s'\mid s,a)\) is the probability of landing in \(s'\) after taking \(a\) in \(s\).

**Reasoning chain.** Equation (3.17.1) is the compact form: one sum over the joint outcomes
\((s',r)\), each contributing \(r+\gamma v_\pi(s')\). To get (3.17.2), distribute the sum over the
two terms. The reward part still sums over the joint distribution because reward depends on the
outcome; the value part no longer depends on \(r\), so summing over \(r\) collapses \(p(s',r\mid
s,a)\) to \(p(s'\mid s,a)\).

Both expressions are the expected one-step reward plus the discounted expected next-state value —
the same bookkeeping viewed as one expectation or as two. The split form is often more convenient
because it matches the structure of the Bellman equation: an immediate term and a continuation
term.

**Why it makes sense.** With a deterministic transition to a single \(s'\), each line reduces to
\(r+\gamma v_\pi(s')\): immediate reward plus discounted future value. With \(\gamma=0\), only the
first term survives and the action value is just the expected immediate reward. The direction of
the discount factor is the only sensible one: a reward received now should not be shrunk, while a
future value should be.

**Check your understanding.** Prediction: make the reward independent of \(s'\). The first sum in
(3.17.2) then factors through \(\sum_{s'}p(s'\mid s,a)=1\), leaving the plain expected reward; if
every transition led to a zero-valued terminal state, the second term would vanish entirely.

### Exercise 3.18 — Optimal state-value function (golf)

- **Book location:** Section 3.6, PDF page ~70.
- **Type:** Theory/visual.
- **Problem:** Draw or describe the optimal state-value function for the golf example.
**Hook.** The optimal state-value surface answers one question: if I play as well as possible from
here, what return should I expect? For golf, that means choosing the best shot at every position.

**Formal bridge.**

\[
v_*(s) = \max_a q_*(s,a),
\tag{3.18.1}
\]

where \(s\) is the ball's position, \(a\) is a shot (including the choice of putter aim), and
\(q_*(s,a)\) is the expected return of taking that shot and then playing optimally.

**Reasoning chain.** A state value under a fixed policy averages over the policy's actions; an
*optimal* state value does not average, because we are free to choose the action. Hence (3.18.1)
takes a maximum over all available shots. For the golfer, the shot that maximises \(q_*(s,\cdot)\)
at the current position is the one to take; when putting, that means choosing the aim with the
best expected return.

The resulting surface assigns to every position the best achievable expected return from that
position. Because rewards are typically negative per stroke (or positive on sinking the putt),
positions close to the hole have values near zero or high, while distant positions have lower
values reflecting the extra strokes expected. The surface is smooth where shot outcomes vary
smoothly with position and can be irregular near obstacles or the hole's rim.

**Why it makes sense.** At the hole itself, no more strokes are needed, so the value is at its best
possible value; moving away adds expected strokes and lowers the value. That direction is forced by
the definition: you can always choose to take a bad shot, so the best achievable return cannot get
worse when you move closer to a goal. If the maximisation in (3.18.1) were an average, the surface
would describe a mediocre golfer, not the best one; the gap between maximum and average is precisely
the value of skill.

**Check your understanding.** Prediction: increase the noise of every shot. More outcomes will land
far from the aim, so expected remaining strokes rise everywhere and the value surface flattens and
drops; near the hole the drop is smallest because fewer shots remain.

### Exercise 3.19 — Optimal action-value contours for putting (golf)

- **Book location:** Section 3.6, PDF page ~70.
- **Type:** Visual.
- **Problem:** Draw or describe the contours of the optimal action-value function for putting,
  `q*(s, putter)`, for the golf example.
**Hook.** Now fix the shot to the putter and map how its value changes with position. Contours make
that map readable: each line joins positions whose putter shot is equally good.

**Formal bridge.** Fixing \(a=\text{putter}\), the action-value surface is

\[
q_*(s,\text{putter}) = \sum_{s',r} p(s',r\mid s,\text{putter})\left[r+\gamma v_*(s')\right],
\tag{3.19.1}
\]

and its contours are the sets \(\{(s): q_*(s,\text{putter})=\text{const}\}\).

where \(s\) is the ball position, \(p(s',r\mid s,\text{putter})\) describes where a putt lands,
\(v_*(s')\) is the optimal value of the landing position, and \(\gamma\) discounts the continuation.

**Reasoning chain.** Equation (3.19.1) averages over where the putt actually lands, then uses the
optimal value of that landing spot. Since \(v_*\) is best near the hole, a putt whose distribution
concentrates near the hole has a higher \(q_*\); that is what "higher contours correspond to better
expected putts" means.

Drawing contours of equal \(q_*\) gives a topographic map of putting quality. Where the lines crowd
together, a small change of position changes the expected outcome a lot — typical near obstacles or
the lip of the hole, where a slight shift changes whether the ball drops in. Near the target the
contours curve around the hole because the best aim depends on the side from which you approach:
a straight putt from one side and a banked putt from another can be equally good, so their positions
lie on the same level set.

**Why it makes sense.** At the hole, every putt is trivial, so \(q_*\) takes its best value and the
contours close around that point. Far from the hole, the putter is less likely to reach the target,
so values are lower and contours spread out. The curvature around the hole is the geometric
signature of an aim-dependent reward: equal-value positions form rings around the target rather
than straight lines, because distance and approach angle both matter.

**Check your understanding.** Prediction: increase putting noise (wider landing distribution).
Equation (3.19.1) then averages over more bad landings, so contours flatten and the value drops,
especially at medium distances where a precise putt used to pay off.

### Exercise 3.20 — Bellman equation for `q*` (recycling robot)

- **Book location:** Section 3.6, PDF page ~70.
- **Type:** Math.
- **Problem:** Give the Bellman equation for `q*` for the recycling robot.
**Hook.** For the recycling robot we want the value of committing to a particular first action —
`search`, `wait`, or `recharge` — and then behaving optimally forever after. That is \(q_*\).

**Formal bridge.** The Bellman optimality equation in action form is

\[
q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_*(s')\right],
\qquad
v_*(s) = \max_{a'} q_*(s,a').
\tag{3.20.1}
\]

where \(a\) is the fixed first action, \(a'\) ranges over actions in the successor state, and
\(p\) is the robot's dynamics table from Exercise 3.5.

**Reasoning chain.** Fix the first action — say `search` from `low`. The world then chooses the
outcome according to \(p\), producing a reward (\(r_{\text{search}}\) or the \(-3\) rescue penalty)
and a next state (`high` or `low`). From that next state onward the robot is free to act optimally
again, which is exactly \(v_*(s')\), the maximum over all action values there.

The pair of equations in (3.20.1) is a coupled fixed-point system: action values depend on state
values, and state values are maxima of action values. Substituting the second into the first
eliminates \(v_*\) and gives the optimality equation for \(q_*\) alone. This identity is the engine
behind value iteration and Q-learning: repeatedly applying the backup drives estimates toward the
unique solution.

**Why it makes sense.** Compare with the policy-specific case of Exercise 3.13: there the
continuation averaged over \(\pi\), whereas here it takes the best available action. The maximum
cannot make the continuation worse than any fixed policy's average, so \(q_*\ge q_\pi\) for every
\(\pi\). If one action (say `recharge`) deterministically leads to the best successor state, the
maximum in (3.20.1) simply selects it; if all actions tie, any one is optimal.

**Check your understanding.** Prediction: add a new action with a higher \(q\) from `low`. The
maximum in (3.20.1) rises, so \(v_*(\text{low})\) rises and the robot's optimal first choice switches
to the new action; with \(\gamma=0\), only the immediate rewards would be compared.

### Exercise 3.21 — Optimal gridworld value (24.4)

- **Book location:** Section 3.6, PDF page ~71.
- **Type:** Computation.
- **Problem:** Figure 3.6 gives the optimal value of the best state of the gridworld as `24.4` (one
  decimal). Express this symbolically via (3.8), then compute to three decimals.
**Hook.** The figure reports \(24.4\) to one decimal. Let's write the symbolic optimality equation and
solve the special-state chain to three decimals, so we can see what the rounding hid.

**Formal bridge.** The symbolic starting point is the optimality equation

\[
v_*(s) = \max_a q_*(s,a)
       = \max_a \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_*(s')\right],
\tag{3.21.1}
\]

and for the special state A, whose move teleports the agent to A′ with reward \(+10\), the chain is

\[
v_*(A) = 10 + 0.9\,v_*(A'), \qquad v_*(A')=16.0216\ldots
\tag{3.21.2}
\]

where \(A'\) is row 4, column 1 of the gridworld, \(\gamma=0.9\), and the \(+10\) is the teleport
reward.

**Reasoning chain.** Equation (3.21.1) is the general fixed-point condition; solving it for the
whole grid gives the optimal values at every cell. For the best cell A, the optimal action uses the
teleport, so the return is the immediate \(+10\) plus the discounted value of the destination A′.
With \(v_*(A')=16.0216\ldots\), that is \(10+0.9\times16.0216\ldots =
24.4194\ldots\). Rounded to one decimal this is \(24.4\), exactly the figure's number; to three
decimals it is \(24.419\).

The small discrepancy between \(24.4\) and \(24.419\) is therefore pure rounding of a one-decimal
picture, not a different solution. Computing to more decimals requires solving the system more
precisely (for example by iterating the Bellman optimality backup until the residual is tiny).

**Why it makes sense.** With \(\gamma=0.9\), value at A is dominated by the chain through A′ but still
contains a discounted contribution from everything downstream, so a one-decimal report naturally
loses information. At the extreme \(\gamma=0\), the value would be just the \(+10\) teleport reward;
as \(\gamma\) grows, long chains of rewards through the grid add more, pushing the value up toward
the reported \(24.4\). The direction is right: a better discount factor makes distant rewards count,
so the best state's value rises.

**Check your understanding.** Prediction: lower \(\gamma\) from \(0.9\) to \(0.8\). Both \(v_*(A')\)
and the discounted chain shrink, so \(v_*(A)\) drops well below \(24.419\); raising \(\gamma\) toward
\(1\) makes the value grow further.

**Where this shows up in the code.** `example_3_5_gridworld` (5×5 gridworld: deterministic actions,
off-grid moves stay put with reward \(-1\), A teleports to A′ with \(+10\), B teleports to B′ with
\(+5\), \(\gamma=0.9\)) and `plot_example_3_5_gridworld` (`fig_3_5_gridworld.png`) in
`chapter_03/solution.py`.

### Exercise 3.22 — Continuing MDP with two actions

- **Book location:** Section 3.6, PDF page ~71.
- **Type:** Computation.
- **Problem:** Continuing MDP with only one decision point (top state), actions `left` and `right`,
  deterministic rewards.
**Hook.** When there is only one decision to make, optimality collapses to a comparison of two
numbers. Let's write the general rule and then see why it reduces so cleanly.

**Formal bridge.**

\[
A_* = \arg\max_{a}\sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_*(s')\right].
\tag{3.22.1}
\]

where \(s\) is the single decision state, \(a\in\{\text{left},\text{right}\}\), \(p\) is deterministic
so each action has one successor and reward, and \(v_*(s')\) is the optimal value of that successor.

**Reasoning chain.** Equation (3.22.1) is the general greedy rule: for each action, average the
immediate reward plus the discounted optimal value of the resulting state, then take the largest.
Because the rewards here are deterministic and the successor is fixed for each action, the sum in
(3.22.1) has a single term per action. Choosing the larger one is therefore exactly "pick the action
whose successor yields the higher expected value" — the general rule with the expectations written
out.

Once the action is chosen, there are no further decisions (or the continuation value already
summarises them), so no deeper search is needed. This is the smallest possible illustration of the
Bellman optimality principle.

**Why it makes sense.** If the two successors have equal value, either action is optimal and the tie
can be broken arbitrarily. If \(\gamma=0\), only the immediate reward matters and the successor
values drop out — the choice becomes myopic. Increasing \(\gamma\) shifts the decision toward the
action whose successor has the brighter future, which is the correct direction because a continuing
task's return is mostly future rewards.

**Check your understanding.** Prediction: raise the reward on the `right` branch. Equation (3.22.1)
picks `right` once its one-step value exceeds the left one; if rewards are tied but the right
successor has a higher \(v_*\), a positive \(\gamma\) still selects `right`, while \(\gamma=0\) would
tie.

### Exercise 3.23 — `v*` in terms of `q*`

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for `v*` in terms of `q*`.
**Hook.** This is the first of the four "translation" identities between optimal values. It says how
to get a state's best value once you know the value of every action there.

**Formal bridge.**

\[
v_*(s) = \max_a q_*(s,a).
\tag{3.23.1}
\]

where \(s\) is a state and \(a\) ranges over the actions available in \(s\).

**Reasoning chain.** A policy-specific state value \(v_\pi(s)\) averages over the actions \(\pi\)
might take. For the optimal value there is no fixed policy — we may choose whichever action is
best at \(s\) — so the average becomes a maximum over \(q_*(s,a)\). Each \(q_*(s,a)\) already assumes
optimal behaviour after \(a\), so taking the best one gives the best achievable return from \(s\).

This identity is one half of the coupled optimality system: (3.23.1) turns action values into
state values, and its partner expresses \(q_*\) in terms of \(v_*\). Together they let value iteration
alternate between the two or eliminate one entirely.

**Why it makes sense.** If a state offers only one action, the maximum equals that action's value.
If a policy were forced to choose uniformly, its value would be the *average* of the \(q_*\) values,
which is never larger than the maximum. The gap between the two is exactly the benefit of being
able to control the choice — so the maximum direction is forced.

**Check your understanding.** Prediction: add a new action whose \(q_*\) is below the current best.
Equation (3.23.1) is unchanged, so \(v_*\) stays the same; add one whose \(q_*\) is higher and \(v_*\)
rises immediately.

### Exercise 3.24 — `q*` in terms of `v*` and the dynamics

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for `q*` in terms of `v*` and the world's dynamics `p(s', r | s, a)`.
**Hook.** Now go the other direction: given the best value of every state, what is the value of
committing to one specific action?

**Formal bridge.**

\[
q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_*(s')\right].
\tag{3.24.1}
\]

where \(p(s',r\mid s,a)\) is the dynamics, \(r\) is the immediate reward, \(v_*(s')\) is the optimal
value of the state reached, and \(\gamma\) is the discount factor.

**Reasoning chain.** The action \(a\) in (3.24.1) is fixed, so there is no maximisation over actions at
this step. The only uncertainty is what the world does: it selects the pair \((s',r)\) with
probability \(p(s',r\mid s,a)\). Each outcome contributes its immediate reward plus the discounted
optimal value of the resulting state, and the sum averages over outcomes.

Then, from \(s'\) onward, the agent can act optimally again — that is why \(v_*\), rather than a
policy's value, appears in the continuation. Combining (3.24.1) with \(v_*(s)=\max_a q_*(s,a)\) gives
the full Bellman optimality equation in one line.

**Why it makes sense.** With a deterministic transition \(p=1\) for one \((s',r)\), the sum collapses
to \(r+\gamma v_*(s')\): reward now plus best future value. With \(\gamma=0\), only the expected
immediate reward remains. The direction is right: a fixed action cannot choose the best outcome, so
it must average over whatever the world delivers, but it can still benefit from optimal play
afterwards.

**Check your understanding.** Prediction: add a constant \(c\) to every \(v_*\). Equation (3.24.1)
then shifts every \(q_*\) by \(\gamma c\), so the shift is discounted; if the transition is
deterministic, the expectation disappears and the action value is a single number.

### Exercise 3.25 — \(\pi *\) in terms of `q*`

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for \(\pi *\) in terms of `q*`.
**Hook.** Once we know how good every action is, extracting an optimal policy is immediate: pick the
best one.

**Formal bridge.**

\[
\pi_*(s) = \arg\max_a q_*(s,a).
\tag{3.25.1}
\]

where \(q_*(s,a)\) is the optimal action value and the maximisation is over all actions available in
\(s\).

**Reasoning chain.** An optimal policy must choose an action whose return, assuming optimal
behaviour thereafter, is as large as possible. That is exactly what \(q_*(s,a)\) measures, so the
argmax in (3.25.1) defines \(\pi_*\) state by state. Because the rule is greedy with respect to the
*optimal* action values, the resulting policy is optimal — unlike greediness with respect to rough
estimates, which can be suboptimal.

If several actions tie for the maximum, any of them is optimal, so the tie-break is arbitrary. A
deterministic tie-break yields a deterministic optimal policy; there can also be many optimal
policies when ties occur, but all achieve the same \(v_*\).

**Why it makes sense.** If only one action is available, \(\pi_*\) must choose it and the argmax is
trivial. If two actions tie, preferring either gives the same return — so there is no reason to
randomise. The direction is forced: choosing anything other than a maximiser would knowingly give
up return that \(q_*\) says is attainable.

**Check your understanding.** Prediction: if the estimates of \(q_*\) contain a little noise, an
action whose true value is slightly lower may win the argmax on a given state; near-ties are where
an imperfect policy is most fragile, while a clear gap makes the choice stable.

### Exercise 3.26 — \(\pi *\) in terms of `v*` and the dynamics

- **Book location:** Section 3.6, PDF pages ~72–76.
- **Type:** Math.
- **Problem:** Give an equation for \(\pi *\) in terms of `v*` and the world's dynamics `p(s', r | s, a)`.
**Hook.** The last translation identity removes \(q_*\) entirely. It says how to act optimally knowing
only the state values and the world's dynamics.

**Formal bridge.**

\[
\pi_*(s) = \arg\max_a \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_*(s')\right].
\tag{3.26.1}
\]

where \(p(s',r\mid s,a)\) is the dynamics, \(v_*(s')\) is the optimal value of the successor state,
\(\gamma\) is the discount factor, and the maximisation is over the actions available in \(s\).

**Reasoning chain.** Start from \(\pi_*(s)=\arg\max_a q_*(s,a)\) and substitute
\(q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_*(s')]\). The result is (3.26.1): a policy that
looks one step ahead through the model and scores each action by the expected immediate reward plus
the discounted optimal value of where it lands. No explicit \(q_*\) table is needed — the dynamics
and \(v_*\) suffice.

This is the one-step lookahead rule behind planning with a known model. It is also the identity that
justifies greedifying with respect to learned state values: once \(v_*\) is accurate, the argmax in
(3.26.1) produces an optimal policy without further learning.

**Why it makes sense.** At \(\gamma=0\), the rule becomes purely myopic: choose the action with the
largest expected immediate reward. As \(\gamma\) grows, the successor values dominate, so the agent
will accept a smaller immediate reward for a much better future state — the correct tradeoff for a
long-horizon task. If all actions lead to the same successor value, the decision reduces to comparing
immediate rewards, which is again intuitive.

**Check your understanding.** Prediction: raise \(\gamma\). Equation (3.26.1) puts more weight on
the successor values, so the chosen action can switch from the highest-immediate-reward option to
the one with the better future; adding a new action with a higher one-step score switches the policy
immediately.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 3.1–3.2 | §3.1–3.2 | `chapter_03/solution.py` (`FiniteMDP`) | `TestFiniteMDP` | `python -m pytest chapter_03/tests/ -v` |
| 3.4–3.5 | §3.2 | `chapter_03/solution.py` (`FiniteMDP.expected_reward`, `FiniteMDP.transition_probs`) | `TestFiniteMDP` | `python -m pytest chapter_03/tests/ -v` |
| 3.9–3.10 | §3.3 | `chapter_03/solution.py` (`compute_return`, `compute_returns_to_go`) | `TestComputeReturn` | `python -m pytest chapter_03/tests/ -v` |
| 3.12–3.13, 3.16–17 | §3.5 | `chapter_03/solution.py` (`bellman_backup_expectation`, `bellman_backup_optimality`, `bellman_backup_*`); Example 3.5 gridworld MDP `example_3_5_gridworld` + `example_3_5_gridworld_random_policy_values` + `plot_example_3_5_gridworld` (`fig_3_5_gridworld.png`) cover the book's gridworld example; v*(A) = 24.4194 (one decimal: 24.4) matching the book's cite for Exercise 3.21 | `TestBellman`, `TestExample35Gridworld` | `python -m pytest chapter_03/tests/ -v` |
| 3.6, 3.14–15 | §3.3–3.5 | `chapter_03/solution.py` (`compute_return`, `bellman_backup_*`, `verify_bellman`) | `TestComputeReturn`, `TestBellman` | `python -m pytest chapter_03/tests/ -v` |

- Full chapter command: `python -m pytest chapter_03/tests/ -v`
- Lint: `ruff check chapter_03/`
- No exercise in Chapter 3 is implemented as a standalone `exercise_*.py` file; the chapter
  implements the underlying machinery in `solution.py` (Algorithms 3.1–3.4).

---

## Verification and Quality Notes

- Every numeric answer above was re-derived by hand or by independent calculation in this document;
  the public solution material (`Habanoz`, `Vojtamolda`, `AntonioSerrano`) was used only as a
  cross-check.
- Theory/visual answers (3.1–3.3, 3.8, 3.18–3.19) are model answers written for student
  understanding; students should replace the model tasks with their own.

---

*Exercises document for Chapter 3 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
