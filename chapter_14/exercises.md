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
- **Cross-check sources (sanity check only, never copied):** public repositories
  `Habanoz/reinforcement-learning-an-introduction`, `AntonioSerrano/Implementation-of-RL-algorithms-from-Sutton-and-Barto-2018`, and
  `vojtamolda/reinforcement-learning-an-introduction`, plus public neuroscience/psychology reviews.
  Disagreements resolved by independent derivation.
- All worked derivations are **rewritten from scratch**.

---

## Derived Exercises

### Derived 1 — Rescorla–Wagner rule and blocking [derived]

- **Book location:** Section 14.2, PDF page 303 (classical conditioning).
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive the Rescorla–Wagner update \(ΔV_i = \alpha _i \beta (\lambda - \sum _j V_j)\) for
  stimulus `i`; use it to predict Kamin's blocking experiment: (a) Phase 1: CS_A alone paired with the
  US until fully conditioned; (b) Phase 2: compound CS_A + CS_B paired with the same US. Show
  `V(CS_B)` remains near 0.
- **Solution.** For a compound trial with present cues `x_i ∈ {0,1}`, the associative strength update is
  \(V_i \leftarrow V_i + \alpha _i \beta (\lambda - V_AX)\) where \(V_AX = \sum _j x_j V_j\) and \(\lambda\) is the US asymptote (e.g. 1).
  **Phase 1 (A→US):** only A present, so \(V_A \leftarrow V_A + \alpha _A \beta (\lambda - V_A)\) and \(V_A \to \lambda\); `V_B = 0`.
  **Phase 2 (AX→US):** \(V_AX = V_A + V_B \approx \lambda + 0\), hence the prediction error
  \(\delta = \lambda - V_AX \approx 0\) and \(ΔV_B = \alpha _B \beta \delta \approx 0\). Because A already fully predicts the US, no error
  remains to drive learning for B — the **blocking** effect.
- **Explanation (student-friendly).** prediction errors drive learning; a fully predictive cue blocks
  association to redundant cues.

### Derived 2 — TD model of classical conditioning [derived]

- **Book location:** Section 14.2, PDF page 303.
- **Type:** Math.
- **Problem (paraphrased):** Write the TD(0) update for a conditioning trial and show that (a) a
  presence trial (CS followed by US) increases `V(CS)`, (b) an extinction trial decreases `V(CS)`
  toward 0, and (c) conditioning reaches a nonzero fixed point when the US is reliably predicted.
- **Solution.** The TD prediction error is \(\delta _t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\) with update
  \(V(S_t) \leftarrow V(S_t) + \alpha \delta _t\). (a) **Presence trial:** at the CS state preceding the US, `R_{t+1}>0`
  (or `V(S_{t+1})` reflects the upcoming US) so \(\delta _t > 0\) initially and `V(CS)` increases.
  (b) **Extinction:** with no US, `R_{t+1}=0` and the bootstrapped target falls, so \(\delta _t < 0\) when
  `V(CS)` overpredicts, driving \(V(CS) \to 0\). (c) **Fixed point:** \(E[\delta _t | S_t]=0\) implies
  \(V(S_t)=E[R_{t+1}+\gamma V(S_{t+1})]\); for a reliably predicted US this yields `V(CS)>0` (the CS
  acquires predictive value equal to the discounted/expected outcome).
- **Explanation (student-friendly).** TD’s bootstrapped error explains acquisition, extinction, and
  second-order conditioning as consequences of one update rule.

### Derived 3 — Instrumental vs classical conditioning [derived]

- **Book location:** Section 14.3, PDF page 315.
- **Type:** Theory.
- **Problem (paraphrased):** Contrast classical conditioning with instrumental conditioning and explain
  why Q-learning/Sarsa-style TD control is instrumental rather than classical.
- **Solution.** **Classical conditioning** is stimulus–outcome prediction: a CS comes to predict a US
  regardless of the animal’s behavior (Pavlovian; modeled by value prediction / Rescorla–Wagner or TD).
  **Instrumental (operant) conditioning** is action–outcome learning: the organism’s behavior changes
  the probability of reinforcement (Thorndike’s law of effect). TD **control** methods such as
  Sarsa/Q-learning learn action values `Q(s,a)` and select actions to maximize return; because the
  policy/choice affects outcomes, they model instrumental learning. Prediction-only TD (estimating `V`)
  corresponds to classical conditioning.
- **Explanation (student-friendly).** prediction is Pavlovian/classical; control (action selection that
  changes outcomes) is instrumental.

### Derived 4 — Delayed reinforcement [derived]

- **Book location:** Section 14.4, PDF page 319.
- **Type:** Math / Theory.
- **Problem (paraphrased):** Derive how eligibility traces bridge the delay between an action/state and
  a later outcome; explain the role of \(\lambda\).
- **Solution.** With linear value approximation `\hat v(s,w)=w^T x(s)`, accumulating eligibility traces
  evolve as \(e_t = \gamma \lambda e_{t-1} + x(S_t)\) (or `∇_w \hat v(S_t,w)` in general), and the TD(λ) weight
  update is \(w \leftarrow w + \alpha \delta _t e_t\), where `δ_t = R_{t+1} + γ \hat v(S_{t+1}) − \hat v(S_t)`. When a
  delayed reward arrives at time `t+k`, `e_{t+k}` still contains a decayed trace
  \((\gamma \lambda )^k x(S_t)\) of earlier states, so credit is assigned backward across the delay. Larger \(\lambda\)
  (closer to 1) keeps traces alive longer, bridging longer delays but with higher variance; smaller
  \(\lambda\) emphasizes recent, bootstrapped credit.
- **Explanation (student-friendly).** decaying eligibility traces propagate TD errors to temporally distant
  causes of reward.

### Derived 5 — Cognitive maps as learned models [derived]

- **Book location:** Section 14.5, PDF page 320.
- **Type:** Theory.
- **Problem (paraphrased):** Explain the difference between a latent world model and a cache of
  conditioned responses, relate cognitive maps to model-based RL, and mention place-cell evidence.
- **Solution.** A **cache of conditioned responses** stores direct stimulus→response/value associations
  learned from experience (model-free). A **latent world model** (cognitive map) represents the
  environment’s structure/transitions, allowing planning and detour/shortcut behavior without direct
  retraining. Tolman’s maze results and hippocampal **place cells** (spatial firing fields) support
  map-like representations. In RL terms, cognitive maps correspond to **model-based** methods (e.g.
  learning a transition model and planning, as in Dyna), whereas cached responses correspond to
  model-free value/policy learning.
- **Explanation (student-friendly).** internal models enable planning and flexible re-routing beyond cached
  habits.

### Derived 6 — Habitual vs goal-directed behavior [derived]

- **Book location:** Section 14.6, PDF page 321.
- **Type:** Theory / Math.
- **Problem (paraphrased):** Contrast habits (model-free) with goal-directed behavior (model-based) and
  explain the reinforcer-devaluation test.
- **Solution.** **Goal-directed** behavior is sensitive to the current value of outcomes and action
  contingencies (model-based evaluation/planning); **habitual** behavior is triggered by stimuli via
  cached action tendencies (model-free). In a **reinforcer-devaluation** test, the outcome is devalued
  (e.g. satiation/aversion) without further training: goal-directed agents reduce the action because
  they evaluate the now-devalued outcome, while habitual control persists. Overtraining tends to shift
  control from model-based to model-free systems, consistent with dual-process accounts (and with
  Dyna-like planning vs direct TD control in RL).
- **Explanation (student-friendly).** devaluation sensitivity distinguishes outcome-evaluating (model-based)
  from cached (model-free) control.

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

## Verification and Quality Notes

- Derived derivations were re-derived independently; public conditioning/Neuroscience literature used
  only as a cross-check. Concentration-style bounds not used in this chapter.

---

*Chapter 14 (Psychology) derived exercises document — part of the exercises-and-solutions series
(`chapter_XX/exercises.md`).*
