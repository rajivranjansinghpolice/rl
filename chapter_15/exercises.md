# Chapter 15 — Neuroscience: Exercises and Solutions

> Companion document for Sutton & Barto, _Reinforcement Learning: An Introduction_
> (2nd Edition, complete draft, November 5, 2017).
> Repo chapter: `chapter_15`
> Book chapter: **Chapter 15** — Neuroscience

*Book contains no formal numbered exercises for this chapter.* The prompts below are derived from
the sections themselves and are labeled **[derived]**.

---

## Scope and Book Mapping

- **Repo chapter:** `chapter_15` — Neuroscience (`PhasicDopamineModel`, conditioning-trace
  simulation, `cs_delta`/`us_delta` analyses).
- **Book chapter:** Chapter 15 — Neuroscience (the neural substrate of reinforcement learning —
  dopamine, actor–critic circuits).
- **Book sections covered:**
  - §15.1 Neuroscience Basics; §15.2 Reward Signals, Values, and Prediction;
    §15.3 The Reward Prediction Error Hypothesis; §15.4 Dopamine;
    §15.5 Experimental Support for the Reward Prediction Error Hypothesis;
    §15.6 TD Error/Dopamine Correspondence; §15.7 Neural Actor–Critic;
    §15.8 Actor and Critic Learning Rules; §15.9 Hedonistic Neurons;
    §15.10 Collective Reinforcement Learning; §15.11 Model-based Methods in the Brain;
    §15.12 Addiction; §15.13 Summary.
- **PDF page references (Barto_Sutton_Book.pdf, pypdf extraction):** §15.3 starts on PDF page 334;
  §15.4 on PDF page 336; §15.5 on PDF page 339; §15.6 on PDF page 341; §15.7 on PDF page 346;
  §15.8 on PDF page 348; §15.9 on PDF page 352; §15.10 on PDF page 353; §15.11 on PDF page 355;
  §15.12 on PDF page 357; §15.13 on PDF page 358.

## Sources and Ground Rules

- **Primary source:** Sutton & Barto (2017/2018), _Reinforcement Learning: An Introduction_,
  Chapter 15. Section statements paraphrased; short verbatim references only.
- **Source and scope:** Sutton & Barto (2nd ed.) Chapter 15; exercise statements are paraphrased
  and derivations are worked from the book's equations, with section/equation/page citations for
  traceability.

---

## Derived Exercises

### Derived 1 — The reward prediction error hypothesis [derived]

- **Book location:** Section 15.3, PDF page 334.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive the TD reward prediction error \(\delta _t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\)
  and explain, under the reward-prediction-error hypothesis, why phasic dopamine is better modeled as
  encoding prediction error than reward itself, and why responses shift from outcomes to predictive cues.

**Hook.** A dopamine neuron bursts for an unexpected reward but falls silent when that reward is
fully predicted. To explain that, we need a signal that measures surprise rather than reward.

**Formal bridge.** The one-step TD error compares what actually happened plus the predicted
future against the current prediction, and value moves in proportion to that difference:

\[
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t), \qquad V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t.
\tag{15.1.1}
\]

where \(R_{t+1}\) is the reward received, \(V(S_t)\) is the predicted value of the current state,
\(V(S_{t+1})\) the bootstrapped prediction for the next state, \(\gamma\) the discount, \(\alpha\)
the step size, and \(\delta_t\) the prediction error proposed as the dopamine signal.

**Reasoning chain.** Start from the Bellman prediction target: the value of the current state
should equal the reward received now plus the discounted value of where we end up. The TD error in
(15.1.1) is the gap between that target and the current prediction — received plus predicted future
value minus what was expected.

Before learning, \(V\approx 0\), so an unexpected reward gives \(\delta_t \approx R_{t+1}>0\): a
phasic burst at the US. That matches the dopamine response to surprise. After learning, the cue
predicts reward, so \(V(\text{CS})\) rises and the value just before the US approaches the expected
discounted reward. Then \(\delta_t\) becomes positive at the predictive **cue** and approximately
zero at a fully predicted US, because \(R_{t+1}+\gamma V(S_{t+1}) \approx V(S_t)\).

Thus phasic dopamine tracks *changes in prediction* — the reward prediction error — not reward
magnitude per se. The response shifts from the outcome to the cue because the cue is where the
prediction changes; by the time the US arrives, there is no surprise left.

**Why it makes sense.** A neuron that coded raw reward would fire at every US, expected or not, and
there would be nothing to explain the observed transfer of the burst to the cue. A difference
signal has exactly the needed property: once prediction equals outcome, the difference is zero.
The sign is also right — a larger-than-expected reward gives a positive burst, a smaller-than-expected
one gives a dip below baseline, so the signal says which way to revise the prediction.

**Check your understanding.** Prediction: make the US occasionally omitted. Delivered US trials
now give \(\delta>0\) and omitted ones give \(\delta<0\), so dopamine bursts and dips alternate.
If instead the cue is made perfectly predictive, \(\delta\) at the US shrinks toward zero while the
response at the cue grows.

### Derived 2 — Schultz et al.: cue→outcome shift and omission dip [derived]

- **Book location:** Section 15.5, PDF page 339; Section 15.6, PDF page 341.
- **Type:** Computation / Math.
- **Problem (paraphrased):** Show how \(\delta _t\) explains (a) burst to unexpected US, (b) burst to CS with
  ~0 response to predicted US, and (c) dip at omitted US.

**Hook.** These three neurophysiological signatures look like three different phenomena. The point
of this exercise is to show they are one equation evaluated in three situations.

**Formal bridge.** Let \(V^-\) be the predicted value just before the US after conditioning and
\(V^+\) the value just after cue onset. The TD error \(\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)\)
takes three characteristic values:

\[
\delta_{\text{US}}^{\text{unpred}} \approx R - 0 > 0, \qquad
\delta_{\text{CS}} = 0 + \gamma V^+ - V_{\text{CS}}^- > 0, \qquad
\delta_{\text{US}}^{\text{pred}} \approx R + 0 - V^- \approx 0, \qquad
\delta_{\text{US}}^{\text{omit}} \approx 0 + 0 - V^- < 0.
\tag{15.2.1}
\]

where \(R\) is the US reward magnitude, \(V^-\) is the expectation at the usual US time,
\(V^+\) is the boosted expectation at the cue, \(\gamma\) discounts the next state, and each line
is the same difference \(\delta_t\) at a different moment in the trial.

**Reasoning chain.** Take the first line of (15.2.1): for an unpredicted US, \(V\approx 0\) at the time of reward, so
\(\delta_{\text{US}}\approx R-0>0\): the classic phasic burst.

For a predicted US, conditioning has made \(V^-\approx E[R]\). At cue onset the prediction jumps,
so \(\delta_{\text{CS}}=0+\gamma V^+-V_{\text{CS}}^->0\) — the burst transfers to the CS. At the
US itself, \(\delta_{\text{US}}\approx R+0-V^-\approx 0\), because the reward was already expected;
the neuron barely responds.

For an omitted US, \(R=0\) at the expected time but \(V^->0\), so
\(\delta_{\text{US}}\approx 0+0-V^-<0\): a phasic dip below baseline. These three signatures —
burst, transfer with silence, and omission dip — follow directly from the same TD error signal.

**Why it makes sense.** The omission dip is the decisive test. A pure reward detector can only go
quiet when nothing arrives; it cannot go negative. A prediction-error detector must go negative,
because it holds a positive expectation \(V^-\) and the actual reward is zero — the difference has
to be negative. That is why the dip is strong evidence for RPE coding rather than reward coding.

**Check your understanding.** Prediction: partially condition the animal so \(V^-\approx R/2\).
The response at a delivered US becomes about \(+R/2\) instead of zero, and the omission dip becomes
about \(-R/2\); as conditioning strengthens, both US responses shrink while the cue response grows.

### Derived 3 — Neural actor–critic [derived]

- **Book location:** Sections 15.7–15.8, PDF pages 346–348.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive actor and critic learning rules in the TD actor–critic framework and
  map actor/critic roles to brain structures; explain the role of \(\delta\) for both.

**Hook.** One scalar prediction error has to do two jobs: tell the critic how to correct its value
estimate and tell the actor which action to reinforce. Let's derive both rules and see why the same
\(\delta\) can serve.

**Formal bridge.** The critic learns a value function by TD; the actor learns a policy by moving
its log-probability in the direction of the same TD error:

\[
w \leftarrow w + \alpha_c\,\delta_t\,\nabla_w V_w(S_t), \qquad
\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t).
\tag{15.3.1}
\]

\[
\theta \leftarrow \theta + \alpha_a\,\delta_t\,\nabla_\theta \ln \pi_\theta(A_t\mid S_t).
\tag{15.3.2}
\]

where \(w\) are critic weights, \(V_w(s)\) its value estimate, \(\theta\) are actor parameters,
\(\pi_\theta(a|s)\) the policy, \(A_t\) the action taken, \(R_{t+1}\) the reward, \(\gamma\) the
discount, \(\delta_t\) the shared TD error, and \(\alpha_c,\alpha_a\) the two step sizes.

**Reasoning chain.** In (15.3.1) the critic performs policy evaluation: it learns \(V_w(s)\) with TD error
\(\delta_t = R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t)\) and update
\(w \leftarrow w+\alpha_c\delta_t\nabla_w V_w(S_t)\).

The actor, in (15.3.2), performs policy improvement: it parameterizes \(\pi_\theta(a|s)\) and updates
\(\theta \leftarrow \theta+\alpha_a\delta_t\nabla_\theta\ln\pi_\theta(A_t|S_t)\), using the same
scalar \(\delta_t\) as a reinforcement signal. When the outcome is better than predicted,
\(\delta_t>0\) raises the probability of the action just taken; when it is worse, the action's
probability falls.

Neurally, the critic is often associated with VTA/SNc dopamine projections signaling \(\delta_t\),
while the actor is associated with dorsal/ventral striatum and corticostriatal plasticity selecting
actions. Both are modulated by dopamine/RPE.

**Why it makes sense.** "How much better than expected was this outcome?" is exactly the
information both learners need: the critic uses it to fix its prediction, and the actor uses it to
decide whether the preceding action deserves reinforcement. If we replaced \(\delta_t\) with raw
reward, the actor would keep reinforcing actions that produced fully expected rewards, which add
no new information; the update would no longer be a policy-gradient-with-baseline rule.

**Check your understanding.** Prediction: suppose the critic becomes perfect, so \(\delta_t\to 0\)
on every step. Both learning rules stop, even though rewards keep arriving: the actor gets no
signal because there is no surprise left to assign credit for. If the critic is poor, the actor's
updates become noisy and misdirected even with the same rewards.

### Derived 4 — Hedonistic neurons [derived]

- **Book location:** Section 15.9, PDF page 352.
- **Type:** Theory.
- **Problem (paraphrased):** Contrast "liking" and "wanting"; explain dissociation between hedonic value
  and incentive salience and its relevance to addiction.

**Hook.** It is possible to want something intensely without enjoying it. To make that dissociation
precise, separate the signal that drives choice from the signal that reports pleasure.

**Formal bridge.** Write choice as driven by incentive salience ("wanting"), while hedonic impact
("liking") is a separate readout that does not enter the decision rule:

\[
\pi(a|s) = \frac{\exp\!\bigl(\beta\,\text{incentive}(a,s)\bigr)}{\sum_{a'}\exp\!\bigl(\beta\,\text{incentive}(a',s)\bigr)}, \qquad \text{incentive is dopamine/RPE-driven, distinct from hedonic impact}.
\tag{15.4.1}
\]

where \(\pi(a|s)\) is the choice probability, \(\beta\) is an inverse-temperature (gain) parameter,
\(\text{incentive}(a,s)\) is the motivational value of action \(a\) in state \(s\), and hedonic
impact is the separate pleasure signal that can move independently of it.

**Reasoning chain.** **Liking** refers to hedonic pleasure or impact; **wanting** refers to incentive
salience or motivational drive to obtain a reward (Berridge & Robinson). In the decision rule
(15.4.1), it is the wanting term that gains weight when dopamine signaling is amplified.

Dopamine is more closely linked to wanting/RPE signaling than to liking: experimental
manipulations can increase wanting (cue-triggered pursuit, craving) without increasing — or while
decreasing — liking. In addiction, sensitization of wanting can make cues compulsively motivating
even when the drug is no longer pleasurable.

**Why it makes sense.** If wanting and liking were one variable, reducing the drug's pleasure would
necessarily reduce pursuit. The observed dissociation shows that the motivational signal can be
amplified independently of the hedonic one: raising \(\beta\,\text{incentive}\) in (15.4.1) makes a
cue more compelling even as the reported pleasure falls. The two are different readouts of one
experience — one drives approach, the other reports enjoyment — so a therapy that lowers liking
need not lower craving.

**Check your understanding.** Prediction: lower the drug's pleasantness while leaving cues and
dopamine/RPE signaling intact. Wanting-driven seeking persists or even increases while liking
falls. If instead dopamine transmission at the cue is blocked, cue-triggered wanting drops sharply
even though the hedonic experience is unchanged.

### Derived 5 — Model-based methods in the brain [derived]

- **Book location:** Section 15.11, PDF page 355.
- **Type:** Theory.
- **Problem (paraphrased):** Argue for hippocampal involvement in model-based RL, contrast with
  model-free habits, and describe empirical signatures.

**Hook.** A system that plans and a system that follows cached habits can produce the same action
today but respond very differently when a contingency changes. The experiment is to change it.

**Formal bridge.** Model-based control recomputes action value from a learned transition model;
model-free control stores value directly from experience:

\[
Q_{\text{mb}}(s,a)=\sum_{s'}p(s'\mid s,a)\bigl[r(s,a,s')+\gamma V(s')\bigr], \qquad
Q_{\text{mf}}(s,a)\leftarrow Q_{\text{mf}}(s,a)+\alpha\bigl[R+\gamma\max_{a'}Q_{\text{mf}}(S',a')-Q_{\text{mf}}(s,a)\bigr].
\tag{15.5.1}
\]

where \(p(s'\mid s,a)\) is the learned model of transitions, \(r\) the outcome reward whose value
can be devalued, \(\gamma\) the discount, \(V\) the planned value, and \(Q_{\text{mf}}\) the cached
action value with step size \(\alpha\).

**Reasoning chain.** **Model-based** control uses an internal model of transitions and outcomes to
evaluate actions by planning, so it is sensitive to devaluation and contingency changes — the
sum in (15.5.1) can be recomputed with a new \(r\) or \(p\) immediately. **Model-free** control
relies on cached values or policies, which must be relearned from new experience.

The hippocampus supports relational and spatial representations and prospective evaluation
consistent with cognitive maps and planning; the dorsolateral striatum is more associated with
habits. Empirically, behavior can shift from goal-directed (devaluation-sensitive) to habitual
with overtraining, as in Daw/Niv/Dayan tasks, indicating parallel systems.

**Why it makes sense.** A cache has compressed the action–outcome contingency into a number, so
changing the outcome's value gives it nothing to update until new experience arrives. A model keeps
the contingency explicit, so substituting a new \(r\) in (15.5.1) changes the choice at once.
Devaluation sensitivity is therefore the behavioral signature of planning; insensitivity says
control has shifted to the cache.

**Check your understanding.** Prediction: devalue the outcome after training. Model-based choices
shift immediately, while model-free habits persist until retrained. With heavy overtraining the
immediate shift becomes smaller, because more control has migrated to the habitual system.

### Derived 6 — Addiction as a persistent prediction error [derived]

- **Book location:** Section 15.12, PDF page 357.
- **Type:** Theory.
- **Problem (paraphrased):** Explain how drug cues hijack RPE/wanting, why motivation can persist after
  pleasure fades, and why "sensitization" maintains aberrant learning/signaling.

**Hook.** Normal prediction errors shrink as an outcome becomes expected. Addiction is interesting
because the motivational signal refuses to shrink — it grows instead.

**Formal bridge.** Cue–drug associations are stamped in by the dopaminergic prediction error, and
sensitization scales the incentive signal that drives seeking:

\[
\Delta\,\text{cue–drug strength} \propto \alpha\,\delta_t, \qquad
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t), \qquad
\text{incentive}_{\text{cue}} \text{ amplified by sensitization}.
\tag{15.6.1}
\]

where \(\delta_t\) is the TD/reward-prediction error carried by dopamine, \(\alpha\) is a learning
rate, \(V\) is the predicted value, and the amplified incentive is what the cue acquires through
repeated exposure.

**Reasoning chain.** Drugs of abuse can produce strong dopaminergic/RPE-like signals that stamp in
cue–drug associations, as in (15.6.1). With repeated exposure, incentive salience ("wanting") to
cues can become **sensitized** while subjective liking may tolerate or decline. Cues then trigger
craving and drug-seeking even without expected pleasure.

Because the motivational signal is amplified and pathologically conditioned, behavior can persist
as if prediction errors keep reinforcing pursuit, resisting normal devaluation. The usual braking
mechanism — a prediction that catches up with the outcome and drives \(\delta_t\to 0\) — is
undermined when the cue's incentive keeps growing instead of extinguishing.

**Why it makes sense.** In ordinary learning, surprise is self-limiting: the better the prediction,
the smaller the error, and the less there is to learn. Sensitization breaks that negative feedback.
The direction is telling: an amplified signal stamps the association in more strongly, which makes
the cue more motivating, which drives more seeking — a positive loop rather than the healthy
convergence to zero error.

**Check your understanding.** Prediction: reduce the dopaminergic response to the cue. Cue-triggered
craving falls even if the drug's pleasantness is unchanged. Conversely, if the drug is devalued by
satiation, model-based evaluation would reduce seeking, but sensitized wanting may override that
and keep the behavior going.

### Derived 7 — Collective reinforcement learning [derived]

- **Book location:** Section 15.10, PDF page 353.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Show how a population of noisy/correlated neurons can encode signals more
  reliably than a single neuron and relate to collective coding.

**Hook.** A single neuron is noisy; a population can be much more reliable, but only if its members
do not all make the same mistake. The algebra says exactly how much correlation costs.

**Formal bridge.** If neuron \(i\) reports \(x_i=s+\eta_i\) with noise \(\eta_i\), the population
average has variance

\[
\bar x = s + \frac{1}{N}\sum_{i=1}^{N}\eta_i, \qquad
\mathrm{Var}(\bar x)=\frac{\sigma^2}{N}\ \text{(independent noise)}, \qquad
\mathrm{Var}(\bar x)=\frac{\sigma^2}{N}\bigl(1+(N-1)\rho\bigr)\ \text{(equal pairwise correlation \(\rho\))}.
\tag{15.7.1}
\]

where \(s\) is the encoded signal, \(N\) the number of neurons, \(\eta_i\) zero-mean noise with
variance \(\sigma^2\), \(\bar x\) the average report, and \(\rho\) the equal pairwise noise
correlation.

**Reasoning chain.** In (15.7.1), let each neuron report \(x_i=s+\eta_i\). Averaging \(N\) of them gives
\(\bar x=s+(1/N)\sum_i\eta_i\), so the signal is preserved while the noise is divided by \(N\) in
the sum. If the noises are independent with variance \(\sigma^2\), then
\(\mathrm{Var}(\bar x)=\sigma^2/N\), so signal-to-noise improves with population size.

With correlations the variance becomes \((\sigma^2/N)(1+(N-1)\rho)\) for equal pairwise
correlation \(\rho\). Diversity and redundancy still help unless \(\rho=1\): as \(N\) grows the
variance approaches \(\rho\sigma^2\) rather than zero, so the benefit saturates. Distributed
dopamine/value signals can therefore support robust collective reinforcement coding, but only to
the extent that the neurons' errors are not perfectly shared.

**Why it makes sense.** Independent errors cancel in an average: positive and negative noise terms
tend to offset, so variance falls as \(1/N\). Perfectly correlated errors cannot cancel — every
neuron makes the same mistake, and the population is just one noisy neuron repeated. Positive
correlation therefore increases variance relative to the independent case, which is why it costs
accuracy.

**Check your understanding.** Prediction: double \(N\) with independent noise and the variance
halves. With \(\rho=0.1\), however, variance tends toward \(0.1\sigma^2\) as \(N\) grows and averaging
saturates — adding more neurons eventually buys almost nothing.

### Derived 8 — Actor vs critic: credit assignment depth [derived]

- **Book location:** Section 15.7–15.8, PDF pages 346–348.
- **Type:** Math.
- **Problem (paraphrased):** Derive how the actor update depends on \(\delta _t\) and eligibility traces, and
  why actor learning is bootstrapped to the critic.

**Hook.** A reward arrives long after the action that caused it. Eligibility traces keep that action
"eligible" for credit, and the critic supplies the error that credit is based on.

**Formal bridge.** With a critic baseline, the actor's policy gradient is estimated from the TD
error, and a decayed trace spreads credit over earlier actions:

\[
\nabla_\theta J(\theta) \approx \mathbb{E}\!\left[\delta_t\,\nabla_\theta \ln \pi_\theta(A_t\mid S_t)\right], \qquad
\delta_t = R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t).
\tag{15.8.1}
\]

\[
z_t = \gamma\lambda\, z_{t-1} + \nabla_\theta \ln \pi_\theta(A_t\mid S_t), \qquad
\theta \leftarrow \theta + \alpha_a\,\delta_t\, z_t.
\tag{15.8.2}
\]

where \(\theta\) are actor parameters, \(\pi_\theta\) the policy, \(J\) the expected return, \(w\)
the critic weights, \(V_w\) the critic's value estimate, \(\delta_t\) the TD error, \(z_t\) the
actor's eligibility trace with decay \(\gamma\lambda\), and \(\alpha_a\) the actor step size.

**Reasoning chain.** The policy-gradient-with-baseline estimate in (15.8.1) replaces the full
return with the TD error \(\delta_t=R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t)\), which lowers variance
while keeping the gradient direction.

With eligibility traces, the actor maintains
\(z_t=\gamma\lambda z_{t-1}+\nabla_\theta\ln\pi_\theta(A_t|S_t)\) and updates
\(\theta \leftarrow \theta+\alpha_a\delta_t z_t\). An action taken \(k\) steps before the error
still appears in \(z_t\) with weight \((\gamma\lambda)^k\), so credit is assigned to earlier
actions in proportion to their decayed trace rather than only to the last action.

Because \(\delta_t\) is computed from the critic's value estimates, the actor's improvement signal
is bootstrapped to — and limited by — the critic's predictions. A biased critic gives the actor a
biased reinforcement signal.

**Why it makes sense.** Without traces, the actor could only reinforce the action immediately
preceding the error, which is useless when rewards are delayed; the trace solves the temporal gap.
Bootstrapping makes sense for the same reason as in the critic: the useful quantity is "better than
expected," not "reward occurred." But it creates a dependency — if the critic's prediction is wrong,
the actor is pushed in the wrong direction, so actor learning cannot be more accurate than its
critic.

**Check your understanding.** Prediction: raise \(\lambda\) toward 1. Credit spreads further back
to older actions, so the most recent action receives a smaller share and updates become noisier.
Improving the critic, by contrast, makes \(\delta_t\) more informative and gives the actor better
directed updates even with the same trace.

---

## Coding Exercises Mapped to Repository Code

| Derived prompt | Book section | Repo code | Repo tests | Pytest command |
|---|---|---|---|---|
| Derived 1–2 | §15.3–15.6 | `chapter_15/solution.py` (`PhasicDopamineModel`, `simulate_conditioning`, `simulate_conditioning_trace`, `cs_delta`, `us_delta`) | `TestPhasicDopamineModel`, `TestPhasicDopamineTask`, `TestConditioningTrace` | `python -m pytest chapter_15/tests/ -v` |
| Derived 3, 8 | §15.7–15.8 | *Theory only in this repo chapter (no actor/policy-gradient implementation; repo provides action-free TD/RPE correspondence only)* | — | — |
| Derived 4–7 | §15.9–15.12 | *Theory only in this repo chapter* | — | — |

- Full chapter command: `python -m pytest chapter_15/tests/ -v`
- Lint: `ruff check chapter_15/`

- Full chapter command: `python -m pytest chapter_15/tests/ -v`
- Lint: `ruff check chapter_15/`

---

## What to take away

- The reward-prediction-error hypothesis is a claim about surprise, not reward: \(\delta_t\) bursts
  for an unexpected US, transfers to the cue as \(V\) grows, and dips below baseline when an
  expected US is omitted. That omission dip is the decisive evidence, because a pure reward
  detector can only go quiet — it cannot signal "worse than expected."
- One scalar \(\delta_t\) does two jobs. The critic uses it to correct its value estimate, the actor
  to reinforce the action just taken; replacing it with raw reward would keep strengthening
  actions that produced fully expected outcomes and would destroy the baseline property derived
  in Chapter 13.
- Wanting is not liking. Dopamine tracks incentive salience, so sensitized cue-driven seeking can
  grow while pleasure fades: ordinary prediction errors shrink as outcomes become expected, but
  addiction turns that self-limiting feedback into a positive loop.
- The actor's eligibility trace spreads credit back by \((\gamma\lambda)^k\), yet the error it
  spreads is only as good as the critic that computed it. Population coding has the same theme of
  hard limits: averaging \(N\) neurons cancels independent noise as \(\sigma^2/N\), but correlated
  noise leaves a floor near \(\rho\sigma^2\), so redundancy eventually stops paying.

**Where the theory meets the code.** The dopamine/RPE and conditioning-trace correspondences
follow from the TD(0) update and have a small simulation here; the actor–critic, hedonistic,
model-based, and population-coding prompts are theoretical. The neural identifications are
interpretive links to the algorithms of Chapters 12–13, not additional learning rules.

---

*Chapter 15 (Neuroscience) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
