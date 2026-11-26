# Chapter 4 — Dynamic Programming: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_04`

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_04` — Dynamic Programming (Algorithms 4.1–4.4: iterative policy
  evaluation `policy_evaluation`, policy improvement `policy_improvement`, policy iteration
  `policy_iteration`, value iteration `value_iteration`; gambler's problem `gamblers_problem`).
- **Book chapter:** Chapter 4 — Dynamic Programming.
- **Book sections covered by the chapter implementation:**
  - Section 4.1 — Policy evaluation (prediction), Example 4.1 gridworld.
  - Section 4.2 — Policy improvement.
  - Section 4.3 — Policy iteration (Jack's car rental is Example 4.2).
  - Section 4.4 — Value iteration (boxed algorithm, Eq. 4.10; Example 4.3 gambler's problem).
  - Section 4.5 — Asynchronous dynamic programming.
- **Key book equations referenced by the exercises:**
  - Eq. (4.3) — Bellman expectation equation for \(v_\pi\).
  - Eq. (4.4)/(4.5) — iterative policy evaluation: \(v_{k+1}(s)=\sum _a \pi (a|s) \sum _{s',r} p(s',r|s,a)[r + \gamma v_k(s')]\).
  - Eq. (4.9) — action-value Bellman expectation: \(q_\pi (s,a)=\sum _{s',r} p(s',r|s,a)[r + \gamma \sum _{a'} \pi (a'|s') q_\pi (s',a')]\).
  - Eq. (4.10) — value iteration: \(v_{k+1}(s)=max_a \sum _{s',r} p(s',r|s,a)[r + \gamma v_k(s')]\).
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction, approximate):** Chapter 4 spans
  roughly PDF pages 60–87; the exercises were extracted from approximately PDF pages 79, 80, 83, 84, 86, and 87.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 4. Exercise statements below are **paraphrased** in student-friendly language; short
  verbatim quotes are used only where unavoidable, always with section/equation citations.
- **Source and scope:** Sutton & Barto (2nd ed.) Chapter 4; exercise statements are paraphrased and
  derivations are worked from the book's equations, with section/equation/page citations for
  traceability.

## How to Use This Document

- Each exercise lists: **Book location**, **Type** (`Theory`, `Computation`, `Math`, `Proof`,
  `Programming`), **Problem**, and a five-part answer: a conversational **Hook**, a numbered formal
  equation with a symbol glossary (**Formal bridge**), a step-by-step **Reasoning chain**, a
  **Why it makes sense** intuition check at extreme cases, and a prediction-style **Check your
  understanding** self-test; a short code pointer into `chapter_04/` follows where a companion
  implementation exists.
- Computation exercises show worked arithmetic; Math/Proof exercises show derivations; Programming
  exercises describe an experiment to run.

---

## Exercises

### Exercise 4.1 — Action values in the 4×4 gridworld

- **Book location:** Section 4.1 / Example 4.1, PDF p. ~79.
- **Type:** Computation.
- **Problem (paraphrased):** In Example 4.1, under the equiprobable random policy \(\pi\), what are
  \(q_\pi (11, down)\) and \(q_\pi (7, down)\)?
**Hook.** An action value is one-step reward plus the discounted value of where you land. Two gridworld
actions show how that works when the landing state is terminal versus ordinary.

**Formal bridge.** For a fixed first action \(a\),

\[
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r + \gamma v_\pi(s')\right].
\tag{4.1.1}
\]

where \(s\) is the current state, \(a\) is the action taken, \(s'\) and \(r\) are the successor state
and reward, \(\gamma=1\) because the gridworld is undiscounted, and \(v_\pi(T)=0\) at termination.

**Reasoning chain.** Every non-terminal transition in Example 4.1 gives reward \(r=-1\), so
(4.1.1) becomes \(-1+v_\pi(s')\) for a deterministic move. From state 11, moving `down` lands
directly in the terminal state \(T\), whose value is zero, so \(q_\pi(11,\text{down})=-1+0=-1\).
There is no future to count once the episode is over.

From state 7, moving `down` lands in state 11. Figure 4.1 gives the converged random-policy value
\(v_\pi(11)\approx-14\), so \(q_\pi(7,\text{down})\approx-1-14=-15\). The \(-1\) is the cost of the
step just taken; the \(-14\) is the expected cost of all the random-policy steps still to come
from state 11.

**Why it makes sense.** Check the boundary case: an action that ends the episode has value exactly
the terminal reward, with no discounted continuation. An action one step earlier must be worse by
at least one step's \(-1\), which is why \(-15\) is below \(-14\). The direction is forced — every
extra step in this gridworld costs a reward of \(-1\), so value must decrease as we move away from
the terminal states.

**Check your understanding.** Prediction: if the step reward were changed from \(-1\) to \(0\), both
action values would rise toward zero, and the terminal action would be exactly \(0\); if the grid were
enlarged, \(v_\pi(11)\) would become more negative and \(q_\pi(7,\text{down})\) would fall with it.

### Exercise 4.2 — Value of a newly added state

- **Book location:** Section 4.1 / Example 4.1, PDF pp. ~79–80.
- **Type:** Computation.
- **Problem (paraphrased):** A new state 15 is added below state 13. In case (a), transitions from
  original states are unchanged and actions from state 15 go left/up/right/down to states
  12/13/14/15 respectively. In case (b), the dynamics of state 13 also change so that action down
  from state 13 leads to state 15. What is \(v_\pi (15)\) for the equiprobable random policy in each case?
**Hook.** Adding a state creates a new Bellman equation. In case (b) it also changes an old one, so we
have to solve the coupled system rather than just plug in numbers.

**Formal bridge.** With four equiprobable actions, \(r=-1\) and \(\gamma=1\), the evaluation
equation for state 15 is

\[
v_\pi(15) = -1 + \frac{v_\pi(12)+v_\pi(13)+v_\pi(14)+v_\pi(15)}{4},
\tag{4.2.1}
\]

because the `down` action from 15 leaves the agent in state 15 itself.

where \(v_\pi(s)\) is the random-policy value of state \(s\), the probabilities are \(1/4\) each, and
the \(-1\) is the reward on every step.

**Reasoning chain.** Case (a): the original states' transitions are unchanged, so Figure 4.1 still
gives \(v_\pi(12)=-22\), \(v_\pi(13)=-20\) and \(v_\pi(14)=-14\). Substituting into (4.2.1) gives
\((3/4)v_\pi(15)=-1+(-22-20-14)/4=-15\), hence \(v_\pi(15)=-20\). The state's own `down` action
creates the \(3/4\) coefficient on the left.

Case (b): state 13 now leads to 15 on `down`, so its value changes. Its unchanged neighbours are
up→9 (with \(v_\pi(9)=-20\)), left→12 and right→14, giving
\(v_\pi(13)=-1+(v_\pi(12)+v_\pi(9)+v_\pi(14)+v_\pi(15))/4=-15+v_\pi(15)/4\). Meanwhile state 15's
equation can be rearranged to \(3v_\pi(15)=-40+v_\pi(13)\). Substituting the expression for
\(v_\pi(13)\) yields \(3v_\pi(15)=-55+v_\pi(15)/4\), i.e. \(11v_\pi(15)/4=-55\), so
\(v_\pi(15)=-20\) and consequently \(v_\pi(13)=-20\).

**Why it makes sense.** The same answer in both cases is no accident: in case (b), state 13's new
destination (15) has exactly the value it had before (-20), so its backup is unchanged and nothing
propagates back to the original grid. Policy evaluation moves value changes only along edges whose
successors changed value; if the new successor's value coincides with the old one, the whole
perturbation vanishes. A self-loop action makes the coefficient \(3/4\) rather than \(1\), which is
the algebraic trace of "one of four actions does not get you anywhere".

**Check your understanding.** Prediction: change the added state's `down` action to go to the
terminal state instead of back to itself. Equation (4.2.1) loses the \(v_\pi(15)/4\) term, so the
left-hand coefficient becomes \(1\) and the value rises to \(-1+(-22-20-14+0)/4=-15.25\).

### Exercise 4.3 — Action-value analogs of (4.3), (4.4), (4.5)

- **Book location:** Section 4.3, PDF p. ~83.
- **Type:** Math.
- **Problem (paraphrased):** Give equations analogous to (4.3), (4.4), and (4.5) for the action-value
  function \(q_\pi\) and its successive approximations `q_0, q_1, q_2, ...`
**Hook.** Dynamic programming backups have the same shape whether they store state values or action
values. Let's write the action-value versions of the definition, the iteration and the improvement
step side by side.

**Formal bridge.** The fixed point and its iterative approximation are

\[
q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right],
\tag{4.3.1}
\]

\[
q_{k+1}(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_k(s',a')\right].
\tag{4.3.2}
\]

where \(q_k(s,a)\) is the \(k\)-th approximation to the action value, \(q_\pi\) is its fixed point,
\(\pi(a'\mid s')\) is the policy's next-action probability, and \(p(s',r\mid s,a)\) is the dynamics.

**Reasoning chain.** Compare (4.3.1) with the state-value Bellman expectation equation. Holding the
first action \(a\) fixed removes the outer average over actions; the world then chooses \((s',r)\),
and the continuation is \(v_\pi(s')\), expanded as an average over the next action \(a'\) weighted by
\(\pi(a'\mid s')\), now written directly as \(q_\pi(s',a')\).

Equation (4.3.2) is the same backup applied to the current estimate \(q_k\) instead of the true
\(q_\pi\): starting from any initial \(q_0\), repeated sweeps converge to \(q_\pi\) under the usual
policy-evaluation conditions (a finite MDP and a fixed policy). Once evaluation has converged,
policy improvement is simply \(\pi'(s)=\arg\max_a q_\pi(s,a)\), with the usual tie-breaking or
ε-soft qualifications. So the whole policy-iteration loop can be run in action-value form.

**Why it makes sense.** The gap between state and action values is just which layer of randomness
is averaged first. If \(\pi\) is deterministic, the inner sum in (4.3.1) collapses to one term; if
\(\gamma=0\), the continuation vanishes and only the expected immediate reward remains. In both
limits the action-value backup reduces to the obvious quantity — the structure survives because it
is the same Bellman expectation written from a different root.

**Check your understanding.** Prediction: replace \(\pi\) by the greedy policy in (4.3.2). The inner
average becomes a maximum over \(a'\), turning evaluation into the optimality backup for \(q_*\);
with a uniform \(\pi\), the same equation averages over all next actions instead.

### Exercise 4.4 — Guaranteeing termination of policy iteration

- **Book location:** Section 4.3, PDF p. ~83.
- **Type:** Theory.
- **Problem (paraphrased):** Policy iteration may never terminate if the policy keeps switching among
  equally good policies. How should the algorithm be modified to guarantee convergence?
**Hook.** Policy iteration is guaranteed to improve — unless it keeps "improving" by swapping between
policies of equal value. Let's close that loophole.

**Formal bridge.** The improvement step should change the policy only when the gain is strict:

\[
\pi'(s) = \arg\max_a q_\pi(s,a) \quad\text{only if}\quad
q_\pi(s,\arg\max_a q_\pi(s,a)) > q_\pi(s,\pi(s));
\tag{4.4.1}
\]

otherwise \(\pi'(s)=\pi(s)\).

where \(q_\pi(s,a)\) is the action value under the current policy and \(\pi(s)\) is its current
action in state \(s\).

**Reasoning chain.** The policy improvement theorem promises that acting greedily with respect to
\(q_\pi\) gives a policy at least as good as \(\pi\). "At least as good" includes equality, so if two
actions tie, an implementation that breaks ties differently on successive passes can flip forever
between equally good policies without ever making progress. Equation (4.4.1) forbids that: when the
best action's value equals the current action's value, keep the current action.

With this rule, every actual change in the policy is a strict improvement at some state and no
degradation elsewhere, so the new policy is strictly better overall. A finite MDP has finitely many
deterministic policies, so an infinite sequence of strict improvements is impossible; the loop must
stop, and it can only stop at a policy that is greedy with respect to its own action values — an
optimal policy. An equivalent fix is to fix a deterministic tie-breaking order once and for all and
never replace an action by an equal-valued alternative.

**Why it makes sense.** Imagine two actions both worth \(-5\) at a state. Swapping between them
changes the policy table but changes no value anywhere, so no progress has been made — the algorithm
is spinning its wheels. Requiring strict improvement makes each iteration pay for itself with a
measurable gain. The direction cannot reverse because improvement never lowers any state's value,
so there is no path back to a previously visited policy.

**Check your understanding.** Prediction: drop the strict-improvement rule and use a random tie-break
instead. On a task with many equal-valued actions, policy iteration can cycle indefinitely among
equally good policies; with the rule in (4.4.1), the number of iterations is bounded by the finite
number of policies.

### Exercise 4.5 (Programming) — Jack's car rental with modifications

- **Book location:** Section 4.3 / Example 4.2, PDF p. ~84.
- **Type:** Programming.
- **Problem (paraphrased):** Implement policy iteration for Jack's car rental and re-solve it with the
  stated modifications (e.g. one car can be shuttled free from the first location to the second, and
  parking incurs extra cost when more than 10 cars are kept overnight at a location). Verify by
  replicating the original problem first.
**Hook.** The point of this exercise is not a new algorithm: it is to check that dynamic programming
handles changed rewards and dynamics once you encode them correctly. Validate first, modify second.

**Formal bridge.** Policy iteration alternates evaluation and greedy improvement,

\[
v_{\pi_k}(s) = \sum_a \pi_k(a\mid s)\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma v_{\pi_k}(s')\right],
\qquad
\pi_{k+1}(s)=\arg\max_a\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma v_{\pi_k}(s')\right].
\tag{4.5.1}
\]

where \(s=(c_1,c_2)\) is the number of cars at the two locations, \(a\) is the number moved overnight
(\(-5\) to \(+5\)), \(p\) comes from Poisson rental/return counts, and \(r\) includes rental revenue,
move costs, and any parking fee.

**Reasoning chain.** First reproduce Example 4.2: Poisson rentals with rates \(\lambda_1=3,\lambda_2=4\),
Poisson returns with \(\lambda_1=3,\lambda_2=2\), rental revenue $10 per car, move cost $2 per car,
at most 20 cars per location and at most 5 cars moved per night. Encode this as \(p(s',r\mid s,a)\)
and run policy iteration from the zero-move policy; the expected result is the known optimal policy
with \(v_*(0,0)\approx421.4\), reached in about five improvement rounds. Matching that baseline tells
you the implementation is sound before any modification.

Then change the reward model in two ways. Make the first car shuttled from location 1 to location 2
free, i.e. reduce the move cost by $2 when that move applies; and charge a $4 overnight parking cost
at any location holding more than 10 cars after the move, independent of how many extra cars there
are. Nothing in (4.5.1) changes structurally — only \(r\) changes — so re-solve with the same policy
iteration and compare the resulting policy and value surface with the baseline. Expect fewer shuttles
into location 2 (the first is now free but further ones are not) and pressure to keep inventories at
or below 10.

If the computation is slow, the book's own suggestion applies: shrink the ranges, for example halve
the maximum number of cars, and re-run. Transitions should be exact, folding Poisson tails into the
last state so probabilities sum to one; the evaluation step can solve the 441×441 Bellman system
directly rather than sweeping.

**Why it makes sense.** Dynamic programming needs a model, not a particular story. Shifting a $2 cost
or adding a $4 fee changes the one-step rewards, so the optimal policy must react in the direction
that avoids those costs: less paid moving and fewer cars parked above the threshold. If a change in
the story did not move the policy at all, that would mean the modified cost was never incurred on
the optimal trajectory, or that the modification was encoded incorrectly.

**Check your understanding.** Prediction: raise the parking fee from $4 to $20. The optimal policy
should move cars more aggressively to keep both locations at or below 10 cars overnight; if the fee
is set to zero, the policy should revert toward the original baseline (apart from the free first
shuttle).

**Where this shows up in the code.** `JacksCarRentalMDP` and `policy_iteration` in
`chapter_04/solution.py`, including `JacksCarRentalMDP(free_first_shuttle=True,
second_parking_fee=4.0)` with `parking_threshold=10`; figure `figures/fig_4_2_jacks_car_rental.png`;
tests `TestJacksCarRental`.

### Exercise 4.6 — Policy iteration for `q_*`

- **Book location:** Section 4.3, PDF p. ~84.
- **Type:** Math / Algorithm.
- **Problem (paraphrased):** Give a complete policy-iteration algorithm for computing `q_*`, analogous
  to the algorithm for `v_*`.
**Hook.** Policy iteration works with action values just as well as with state values. The only change
is which table the evaluation backup updates and which table the greedy step consults.

**Formal bridge.** The evaluation backup and improvement step are

\[
q(s,a) \leftarrow \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q(s',a')\right],
\qquad
\pi(s)\leftarrow\arg\max_a q(s,a).
\tag{4.6.1}
\]

where \(q\) is the current action-value estimate, \(\pi\) is the current policy, and the loop stops
when the largest update \(\Delta\) falls below the threshold \(\theta\).

**Reasoning chain.** Initialize \(q(s,a)\) arbitrarily for all state-action pairs and \(\pi(s)\)
arbitrarily in \(A(s)\). In the evaluation phase, sweep over every pair \((s,a)\), replacing
\(q(s,a)\) by the backed-up value in (4.6.1) and tracking the largest change \(\Delta\). When
\(\Delta<\theta\), the table is an approximate \(q_\pi\) for the current policy.

In the improvement phase, set `policy-stable` true, then for each state save the old action and set
\(\pi(s)=\arg\max_a q(s,a)\). If the action changed, mark the policy unstable — applying the
strict/tie-safe rule from Exercise 4.4 so equal-valued swaps cannot cause a cycle. If no state
changed its action, the policy is greedy with respect to its own \(q_\pi\), hence optimal, and the
algorithm returns \(q\approx q_*\) and \(\pi\approx\pi_*\); otherwise return to evaluation.

**Why it makes sense.** Greedifying with respect to \(q_\pi\) is guaranteed to improve the policy, and
a finite MDP has only finitely many policies, so the alternation must eventually settle. The action
values hold strictly more information than state values alone — they already score each candidate
action — so no extra model lookup is needed in the improvement step. If the tie rule from Exercise
4.4 were omitted, this "must terminate" guarantee would be the first thing to fail.

**Check your understanding.** Prediction: make \(\theta\) smaller. The evaluation sweeps become more
numerically exact before each improvement, so the final \(q\) is more accurate but each round costs
more computation; with \(\theta\) too large, improvement may act on noisy values and extra rounds
may be needed.

### Exercise 4.7 — Policy iteration restricted to ε-soft policies

- **Book location:** Section 4.2/4.3, PDF p. ~84.
- **Type:** Theory.
- **Problem (paraphrased):** If only ε-soft policies are allowed, what changes are required in steps 3,
  2, and 1 (in that order) of the policy iteration algorithm for \(v_\pi\)?
**Hook.** Now the policy is not free to be deterministic: every action must keep at least an \(\epsilon\)
share of probability. We take the steps in the order the exercise asks — improvement, evaluation,
initialization.

**Formal bridge.** The ε-soft greedy improvement rule is

\[
\pi'(a\mid s) =
\begin{cases}
1-\epsilon + \epsilon/|A(s)|, & a\in A_*(s),\\
\epsilon/|A(s)|, & \text{otherwise},
\end{cases}
\qquad
A_*(s)=\arg\max_a q_\pi(s,a),
\tag{4.7.1}
\]

where \(q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_\pi(s')]\) and \(|A(s)|\) is the number of
actions in state \(s\).

**Reasoning chain.** Step 3: replace the fully greedy deterministic improvement by (4.7.1). Compute
\(q_\pi\) from the current \(v_\pi\), find the maximising actions \(A_*(s)\), give them the large
probability \(1-\epsilon+\epsilon/|A(s)|\), and give every other action the small probability
\(\epsilon/|A(s)|\); if several actions tie, split the greedy mass among them. This is the greediest
policy *within* the ε-soft class, so it improves the policy without violating the constraint.

Step 2: evaluation is unchanged in form. Iterative policy evaluation (Eq. 4.5) uses
\(\pi(a\mid s)\) as given, so we simply plug in the ε-soft probabilities from the current policy;
\(v_\pi\) then describes exactly the policy that will actually be executed.

Step 1: initialization must also respect the constraint — start from any ε-soft policy, the simplest
being the uniform policy \(\pi(a\mid s)=1/|A(s)|\). With all three steps in place, the algorithm
converges to an ε-soft optimal policy: best among the policies the constraint allows.

**Why it makes sense.** At \(\epsilon=0\), (4.7.1) collapses to a deterministic greedy policy, so the
restricted algorithm recovers ordinary policy iteration. At the opposite extreme \(\epsilon=1\), all
actions have equal probability and improvement has no room to act, so learning can only evaluate
the fixed uniform policy. The small probability reserved for non-greedy actions is the price of
continued exploration; the best achievable value is correspondingly below the unconstrained
optimum, but no lower, because the greedy mass still goes to the best actions.

**Check your understanding.** Prediction: lower \(\epsilon\) toward zero. The ε-soft optimal policy
approaches the unconstrained optimal policy and its value rises; raising \(\epsilon\) flattens the
distribution, so the value falls toward that of uniform random play.

### Exercise 4.8 — Why the gambler's optimal policy looks "curious"

- **Book location:** Section 4.4 / Example 4.3, PDF p. ~86.
- **Type:** Theory.
- **Problem (paraphrased):** Explain why the optimal policy for the gambler’s problem has its peculiar
  shape—in particular, why it bets everything at capital 50 but not at capital 51 (for `p_h < 0.5`,
  e.g. `p_h=0.4`).
**Hook.** The gambler's optimal stakes look erratic: all-in at 50, cautious at 51. The explanation is
that the objective is a probability of success, not a pile of money.

**Formal bridge.** The gambler maximizes the probability of reaching the goal,

\[
v(s) = \max_{a\in\{0,\ldots,\min(s,100-s)\}}
\left[p_h\,v(s+a) + (1-p_h)\,v(s-a)\right],
\tag{4.8.1}
\]

with \(v(100)=1\), \(v(0)=0\), and reward \(1\) only on reaching 100 (0 otherwise).

where \(s\) is current capital, \(a\) is the stake, \(p_h\) is the probability of winning a bet
(\(p_h=0.4\) in the question), and \(v(s\pm a)\) is the value of the capital after a win or loss.

**Reasoning chain.** The objective is to maximise the probability of reaching capital 100, not the
expected capital. With \(p_h<0.5\), the wager is unfavorable: repeated small bets tend to grind capital
down, so a cautious strategy is more likely to fail eventually. The optimal response is bold,
high-variance play — fewer, larger bets that either reach the goal quickly or end the game.

At capital 50, staking everything reaches 100 on a single win, with probability \(p_h\); any smaller
stake requires several unfavorable bets in a row, which is worse. At capital 51, the analogous
aggressive stake of 49 would reach 100 on a win but drop the gambler to capital 2 on a loss — a state
from which success is very unlikely. A smaller stake preserves a better chance of recovering, so
(4.8.1) favours it. The optimal stake is therefore a tradeoff between the chance of hitting the goal
immediately and the value of the losing successor state.

This tradeoff changes with \(s\), producing the non-monotone, triangular-looking optimal policy in
Figure 4.3, reproduced by `gamblers_problem(p_h=0.4)` with smallest-stake tie-breaking. As Exercise
4.9 notes, the optimal policy is not unique, so tie-breaking affects the exact staircase shape.

**Why it makes sense.** Consider the extreme \(p_h=0.5\): the game is fair, a martingale, and every
stake has the same success probability, so boldness brings no advantage. As \(p_h\) falls below \(0.5\),
the odds turn against the gambler and variance becomes valuable — an unfavorable game is more likely
to be beaten by a short lucky run than by a long grind. At the other extreme \(p_h=1\), a tiny stake
always wins, so boldness is unnecessary. The sign of the effect (bold when \(p_h<0.5\)) is forced by
this logic.

**Check your understanding.** Prediction: raise \(p_h\) from \(0.4\) to \(0.55\). The game becomes
favorable, so small, repeated stakes are now advantaged and the optimal policy becomes cautious
rather than bold; at \(p_h=0.5\) the choice of stake is nearly indifferent.

### Exercise 4.9 (Programming) — Value iteration for the gambler's problem

- **Book location:** Section 4.4/4.5, Example 4.3, PDF p. ~87.
- **Type:** Programming.
- **Problem (paraphrased):** Implement value iteration for the gambler’s problem and solve it for
  `p_h = 0.25` and `p_h = 0.55` (introducing dummy terminal states for capital 0 and 100 with values 0
  and 1). Show results graphically as in Figure 4.3. Are the results stable as \(\theta \to 0\)?
**Hook.** Here the dynamics are fully known, so value iteration can apply the Bellman optimality
backup directly. We also check whether tightening the stopping threshold changes the answer.

**Formal bridge.** With capital \(s\in\{1,\ldots,99\}\) and stake
\(a\in\{0,\ldots,\min(s,100-s)\}\), the update is

\[
V_{k+1}(s)=\max_a\left[p_h\left(r_{\text{win}}+V_k(s+a)\right)
+(1-p_h)\left(0+V_k(s-a)\right)\right],
\tag{4.9.1}
\]

run until the largest change is below \(\theta\), with dummy terminal states \(V(0)=0\) and
\(V(100)=1\).

where \(p_h\) is the win probability (run at \(0.25\) and \(0.55\), plus \(0.4\) for comparison),
\(r_{\text{win}}=1\) only when the win reaches 100, and \(\theta\) is the stopping threshold.

**Reasoning chain.** Set up the states \(s=1,\ldots,99\) and the allowed stakes
\(a=0,\ldots,\min(s,100-s)\). A win moves capital to \(s+a\) with the reward \(1\) only if that is
100; a loss moves it to \(s-a\) with reward \(0\), and reaching 0 is terminal with value 0. The
value of the goal state itself is 1.

Apply (4.9.1) repeatedly from an initial value table until the largest change across states falls
below \(\theta\). Then plot the converged value function and the argmax stake against capital for
\(p_h=0.25\) and \(p_h=0.55\), adding \(p_h=0.4\) for comparison with Figure 4.3. The low-probability
case should show the bold, triangular stakes of Exercise 4.8; the high-probability case should be
much more cautious.

For the stability question: as \(\theta\to0\), the sweeps simply continue until updates become
negligible, so both the value estimates and the induced policy settle at the fixed point. The only
residual variation comes from floating-point precision and tie-breaking among equally good stakes —
recall that the optimal policy is non-unique, so the plotted staircase can differ in its ties even
when the values agree.

**Why it makes sense.** Value iteration is a contraction toward the unique optimal value function,
so a smaller \(\theta\) can only make \(V\) closer to that fixed point; it cannot move it to a
different one. The policy is less stable than the values because argmax is discontinuous at ties —
a tiny numerical change can flip which equal-valued stake is displayed. That is why values converge
cleanly while the policy plot may flicker near ties.

**Check your understanding.** Prediction: tighten \(\theta\) by a factor of ten. The value curves barely
move, but more sweeps are required; the policy's tie-breaking may shift on a few capitals even though
the values are essentially identical.

**Where this shows up in the code.** `gamblers_problem(p_h, goal=100, theta=...)` and
`value_iteration` in `chapter_04/solution.py`; tests `TestGamblersProblem` and `TestValueIteration`.

### Exercise 4.10 — Value iteration update for action values

- **Book location:** Section 4.5, PDF p. ~87.
- **Type:** Math.
- **Problem (paraphrased):** What is the analog of the value iteration update (4.10) for action values,
  i.e. for `q_{k+1}(s,a)`?
**Hook.** The last exercise swaps the table used by value iteration. The backup structure is unchanged;
only the continuation term is written in action-value form.

**Formal bridge.**

\[
q_{k+1}(s,a) = \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma\max_{a'} q_k(s',a')\right],
\tag{4.10.1}
\]

with the terminal convention \(q(\text{terminal},\cdot)=0\).

where \(q_k(s,a)\) is the current action-value estimate, \(a'\) ranges over actions in the successor
state \(s'\), and \(\gamma\) discounts the optimal continuation.

**Reasoning chain.** Start from the state-value update (4.10): \(v_{k+1}(s)=\max_a\sum_{s',r}
p(s',r\mid s,a)[r+\gamma v_k(s')]\). To get an action-value version, fix the first action \(a\), so
the maximisation over \(a\) and the outer sum disappear; the world still chooses \((s',r)\), and the
continuation must be the best value of \(s'\). Since \(v_*(s')=\max_{a'}q_*(s',a')\), replace
\(v_k(s')\) by \(\max_{a'}q_k(s',a')\), giving (4.10.1).

Apply this backup to every pair \((s,a)\) on each sweep. Terminal state-action pairs are set to zero,
matching the convention that no rewards follow termination. Iterating the backup converges to
\(q_*\), the unique fixed point of the Bellman optimality equation in action form.

**Why it makes sense.** At \(\gamma=0\), (4.10.1) reduces to the expected immediate reward — there
is no future to maximise. If the maximisation were omitted, the update would evaluate whatever
policy the old \(q_k\) happened to encode rather than improving it; the greedy step is what turns
evaluation into optimality. Once \(q_*\) is known, an optimal policy is read off by \(\arg\max_a
q_*(s,a)\) with no model lookup at action time.

**Check your understanding.** Prediction: set \(\gamma\) close to 1. The maximised continuation
dominates the backup, so values grow with the length of the best future chain; at \(\gamma=0\) all
action values collapse to one-step expected rewards and the policy becomes myopic.

---

## Coding Exercises Mapped to Repository Code

| Book exercise(s) | Type / mapping | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| 4.1 | Computation (gridworld q-values) — no dedicated repo test | — | — | — |
| 4.2 | Computation (added-state values) — no dedicated repo test | — | — | — |
| 4.3 | Math (q-evaluation equations) — no dedicated repo test | `policy_evaluation` (state-value analog) | `TestPolicyEvaluation` (related) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyEvaluation -v` |
| 4.4 | Theory (termination/tie handling) — no dedicated repo test | `policy_iteration` (implements improvement loop) | `TestPolicyIteration` (related) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyIteration -v` |
| 4.5 | Programming (Jack's car rental + modifications) — implemented | `JacksCarRentalMDP` (+ `policy_iteration`) | `TestJacksCarRental` | `python -m pytest chapter_04/tests/test_solution.py::TestJacksCarRental -v` |
| 4.6 | Math/Algorithm (policy iteration for `q_*`) — no dedicated repo test | — | — | — |
| 4.7 | Theory (ε-soft improvement) — no dedicated repo test | `policy_improvement` (greedy improvement, not ε-soft) | `TestPolicyImprovement` (related, not ε-soft) | `python -m pytest chapter_04/tests/test_solution.py::TestPolicyImprovement -v` |
| 4.8 | Theory (gambler's policy shape) | `gamblers_problem`, `value_iteration` | `TestGamblersProblem`, `TestValueIteration` | `python -m pytest chapter_04/tests/test_solution.py::TestGamblersProblem -v` |
| 4.9 | Programming (gambler’s problem for `p_h=0.25,0.55`, stability as \(\theta \to 0\)) | `gamblers_problem`, `value_iteration` | `TestGamblersProblem`, `TestValueIteration` | `python -m pytest chapter_04/tests/test_solution.py::TestGamblersProblem -v` |
| 4.10 | Math (action-value value iteration) — no dedicated repo test | `value_iteration` (state-value analog) | `TestValueIteration` (related) | `python -m pytest chapter_04/tests/test_solution.py::TestValueIteration -v` |

- Full chapter command: `python -m pytest chapter_04/tests/ -v`
- Lint: `ruff check chapter_04/`
- No standalone `exercise_*.py` file exists for this chapter. Theory/derivation exercises are
  answered in this document; 4.5 and 4.8–4.9 are directly exercised by the repo tests.

---

## What to take away

- Policy iteration and value iteration are the same contraction viewed from two sides. Policy
  iteration alternates full evaluation of the current policy with greedy improvement; value iteration
  folds one evaluation sweep into each improvement step. Both reach the unique optimal value, but the
  greedy policy can flicker between equally good actions even after the values have settled.
- The gambler's problem teaches that value is not money. When the bet is unfavourable, variance is the
  asset: bold stakes finish the race quickly or fail quickly, while a long grind almost surely loses.
  At a fair coin every stake has the same success probability, which is why the optimal policy is
  non-unique and its shape depends on tie-breaking.
- Termination is a property to argue for, not to assume. Greedy improvement needs a finite-policy or
  stopping argument to guarantee policy iteration halts, and restricting the search to ε-soft policies
  converges to the best policy *within that class*, not necessarily to the unrestricted optimum.
- A common trap is to treat a converged gridworld number as a standalone fact rather than the fixed
  point of the Bellman expectation equation for a particular policy. States whose successor is a
  terminal transition are exactly determined by that one-step reward, which is why some action values
  are exact while their neighbours are only converged approximations.
- Chapter 4 solves the model it is handed; Chapter 5 replaces the expectation with sampled returns and
  Chapter 6 replaces the full return with a bootstrapped one-step target. The Bellman optimality
  equation is the shared destination, and DP is simply the version that gets to compute the model.

**Where the theory meets the code.** Exercises 4.1–4.2 take their numbers from the Bellman expectation
equations and Figure 4.1's converged gridworld values, with \(q_\pi (11,down)\) exact by the terminal
transition; 4.4, 4.6, 4.7 and 4.10 are DP derivations from the book's definitions. Programming items
4.5 and 4.9 are backed by `JacksCarRentalMDP` (`free_first_shuttle`, `second_parking_fee`) and
`gamblers_problem`. In the 2nd (2018) edition Figure 4.2 is Jack's car rental and Figure 4.3 is the
gambler's problem; the local draft and older editions number them differently, and the repository
figures (`fig_4_2_jacks_car_rental.png`, `fig_4_3_gamblers_problem.png`) are illustrative
demonstrations, not pixel-identical redraws. The gambler's optimal policy is non-unique, with ties
broken toward the smallest stake to reproduce the book's staircase/pyramid shape.

---

*Exercises document for Chapter 4 — part of the per-chapter exercises-and-solutions series
(`chapter_XX/exercises.md`).*
