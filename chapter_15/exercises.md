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
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction`, plus public neuroscience reviews.
  Disagreements resolved by independent reasoning.
- All worked analyses are **rewritten from scratch**.

---

## Derived Exercises

### Derived 1 — The reward prediction error hypothesis [derived]

- **Book location:** Section 15.3, PDF page 334.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive the TD reward prediction error \(\delta _t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\)
  and explain, under the reward-prediction-error hypothesis, why phasic dopamine is better modeled as
  encoding prediction error than reward itself, and why responses shift from outcomes to predictive cues.
- **Solution.** From the Bellman prediction target, the one-step TD error is
  \(\delta _t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\) (the difference between received+predicted future value and
  current prediction). **Before learning** \(V\approx 0\), so an unexpected reward gives \(\delta _t \approx R_{t+1} > 0\)
  (phasic burst at US). **After learning**, the cue predicts reward (`V(CS)` rises, `V` before US
  approaches expected discounted reward), so \(\delta _t\) becomes positive at the predictive **cue** and ~0 at a
  fully predicted US (\(R_{t+1} + \gamma V(S_{t+1}) \approx V(S_t)\)). Thus phasic dopamine tracks *changes in
  prediction* (RPE), not reward magnitude per se.
- **Explanation (student-friendly).** bootstrapping makes δ respond to surprise, explaining the cue shift.

### Derived 2 — Schultz et al.: cue→outcome shift and omission dip [derived]

- **Book location:** Section 15.5, PDF page 339; Section 15.6, PDF page 341.
- **Type:** Computation / Math.
- **Problem (paraphrased):** Show how \(\delta _t\) explains (a) burst to unexpected US, (b) burst to CS with
  ~0 response to predicted US, and (c) dip at omitted US.
- **Solution.** Let the predicted value just before US be \(V^- \approx E[R]\) after conditioning.
  (a) **Unpredicted US:** \(V\approx 0 ⇒ \delta _US \approx R - 0 > 0\).
  (b) **Predicted US:** cue onset yields \(\delta _CS = 0 + \gamma V^+ - V_CS^- > 0\) (prediction increases), while
  at US \(\delta _US \approx R + 0 - V^- \approx 0\).
  (c) **Omission:** at expected US time `R=0` but `V^- > 0`, so \(\delta _US \approx 0 + 0 - V^- < 0\) (a phasic dip).
  These three signatures follow directly from the same TD error signal.
- **Explanation (student-friendly).** one RPE variable unifies burst, transfer, and omission responses.

### Derived 3 — Neural actor–critic [derived]

- **Book location:** Sections 15.7–15.8, PDF pages 346–348.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive actor and critic learning rules in the TD actor–critic framework and
  map actor/critic roles to brain structures; explain the role of \(\delta\) for both.
- **Solution.** **Critic (value/prediction):** learns `V_w(s)` with TD error
  \(\delta _t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)\) and update \(w \leftarrow w + \alpha _c \delta _t \nabla _w V_w(S_t)\) (policy
  evaluation). **Actor (policy):** parameterizes \(\pi _\theta (a|s)\) and updates
  \(\theta \leftarrow \theta + \alpha _a \delta _t \nabla _\theta ln \pi _\theta (A_t|S_t)\) (policy improvement), using the same scalar \(\delta _t\) as a
  reinforcement signal. Neurally, the critic is often associated with VTA/SNc dopamine projections
  signaling \(\delta _t\), while the actor is associated with dorsal/ventral striatum and corticostriatal
  plasticity selecting actions; both are modulated by dopamine/RPE.
- **Explanation (student-friendly).** a shared TD error can train both action selection and value prediction.

### Derived 4 — Hedonistic neurons [derived]

- **Book location:** Section 15.9, PDF page 352.
- **Type:** Theory.
- **Problem (paraphrased):** Contrast "liking" and "wanting"; explain dissociation between hedonic value
  and incentive salience and its relevance to addiction.
- **Solution.** **Liking** refers to hedonic pleasure/impact; **wanting** refers to incentive salience or
  motivational drive to obtain a reward (Berridge & Robinson). Dopamine is more closely linked to
  wanting/RPE signaling than to liking: manipulations can increase wanting (cue-triggered pursuit,
  craving) without increasing (or while decreasing) liking. In addiction, sensitization of wanting can
  make cues compulsively motivating even when the drug is no longer pleasurable.
- **Explanation (student-friendly).** motivation ("wanting") and pleasure ("liking") are separable processes.

### Derived 5 — Model-based methods in the brain [derived]

- **Book location:** Section 15.11, PDF page 355.
- **Type:** Theory.
- **Problem (paraphrased):** Argue for hippocampal involvement in model-based RL, contrast with
  model-free habits, and describe empirical signatures.
- **Solution.** **Model-based** control uses an internal model of transitions/outcomes to evaluate actions
  by planning (sensitive to devaluation and contingency changes). **Model-free** control relies on cached
  values/policies. The hippocampus supports relational/spatial representations and prospective
  evaluation consistent with cognitive maps and planning; dorsolateral striatum is more associated with
  habits. Empirically, behavior can shift from goal-directed (devaluation-sensitive) to habitual with
  overtraining (e.g. Daw/Niv/Dayan tasks), indicating parallel systems.
- **Explanation (student-friendly).** the brain likely implements both planning and cached control.

### Derived 6 — Addiction as a persistent prediction error [derived]

- **Book location:** Section 15.12, PDF page 357.
- **Type:** Theory.
- **Problem (paraphrased):** Explain how drug cues hijack RPE/wanting, why motivation can persist after
  pleasure fades, and why "sensitization" maintains aberrant learning/signaling.
- **Solution.** Drugs of abuse can produce strong dopaminergic/RPE-like signals that stamp in
  cue–drug associations. With repeated exposure, incentive salience ("wanting") to cues can become
  **sensitized** while subjective liking may tolerate or decline; cues then trigger craving and
  drug-seeking even without expected pleasure. Because the motivational signal is amplified/pathologically
  conditioned, behavior can persist as if prediction errors keep reinforcing pursuit, resisting normal
  devaluation.
- **Explanation (student-friendly).** maladaptive amplification of wanting sustains compulsive seeking.

### Derived 7 — Collective reinforcement learning [derived]

- **Book location:** Section 15.10, PDF page 353.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Show how a population of noisy/correlated neurons can encode signals more
  reliably than a single neuron and relate to collective coding.
- **Solution.** Let each neuron `i` report `x_i = s + η_i` with noise `η_i`. Averaging `N` neurons gives
  `\bar x = s + (1/N)Σ η_i`; if noises are independent with variance \(\sigma ^2\), `Var(\bar x)=σ²/N`,
  improving signal-to-noise with population size. With correlations, variance becomes
  \((\sigma ^2 /N)(1 + (N-1)\rho )\) (for equal pairwise correlation \(\rho\)), so diversity/redundancy still helps unless
  correlations are perfect. Distributed dopamine/value signals can thus support robust collective
  reinforcement coding.
- **Explanation (student-friendly).** population codes average noise and increase reliability.

### Derived 8 — Actor vs critic: credit assignment depth [derived]

- **Book location:** Section 15.7–15.8, PDF pages 346–348.
- **Type:** Math.
- **Problem (paraphrased):** Derive how the actor update depends on \(\delta _t\) and eligibility traces, and
  why actor learning is bootstrapped to the critic.
- **Solution.** For policy gradient with a critic baseline, \(\nabla _\theta J(\theta ) \approx E[\delta _t \nabla _\theta ln \pi _\theta (A_t|S_t)]\)
  with \(\delta _t = R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t)\). With eligibility traces, the actor maintains
  \(z_t = \gamma \lambda z_{t-1} + \nabla _\theta ln \pi _\theta (A_t|S_t)\) and updates \(\theta \leftarrow \theta + \alpha _a \delta _t z_t\), assigning credit to
  earlier actions in proportion to their decayed trace. Because \(\delta _t\) is computed from the critic’s
  value estimates, the actor’s improvement signal is bootstrapped to (and limited by) the critic’s
  predictions.
- **Explanation (student-friendly).** traces distribute credit over time, but the TD error source is the critic.

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

## Verification and Quality Notes

- Derived analyses were re-derived independently; public neuroscience literature is used only as a
  cross-check. Dopamine-RPE equivalence follows the TD(0) update.

---

*Chapter 15 (Neuroscience) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
