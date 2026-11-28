"""Solution implementation for Chapter 15 - Neuroscience.

Reference: Sutton & Barto (2nd Ed), Chapter 15 (Neuroscience).

Implements:
- Alg 15.1 — Phasic dopamine / TD-error model (Section 15.6, the idealized
  conditioning task of Figure 15.5; reward prediction error hypothesis of
  dopamine neuron activity, Section 15.3; Montague, Dayan & Sejnowski 1996)
- Alg 15.2 — Cue-reward conditioning trace (Section 15.6; reproduces the
  event sequence of Figures 15.3–15.5: early trials, large phasic response
  at the US; later trials, response shifts to the CS; omission after
  learning, dip below baseline; re-insertion, response reappears)

These are *models of neural signaling*, not just algorithms: the TD error
delta stands in for the phasic activity of dopamine neurons (Section 15.2).
Delta here is an action-free prediction error (Section 15.6), matching how
the book compares TD errors to dopamine neuron activity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ============================================================================
# Algorithm 15.1 / 15.2 core: TD(0) prediction over a trial, CSC layout
# Book Section 15.6, Figure 15.5
# ============================================================================


class PhasicDopamineModel:
    """TD(lambda) predictor of trial-timed reward with a CSC state layout.

    Book Sections 14.2.3 and 15.6. Each time step in a trial is a distinct
    state (complete serial compound, CSC, layout), and the inter-trial
    interval is a single frozen state with value 0 — the episodic assumption
    of Section 15.6 ("predictions are conﬁned to individual trials"). Hence
    the ITI state is never updated, and the earliest reward-predicting
    state of a trial makes a positive TD error relative to it.

    Phasic dopamine stand-in: the TD error delta produced per transition is
    read as a phasic dopamine response (Section 15.3). A negative delta is
    a drop below the neuron's background rate (Montague et al. 1996).

    The update is the backward-view semi-gradient TD(lambda) of Chapter 12
    with tabular (one-hot) features, equivalently Eqs. (14.4)-(14.6):

        delta_t = R_{t+1} + gamma * v(S_{t+1}) - v(S_t)   (14.5, tabular)
        z_{t+1} = gamma * lambda * z_t + 1[S_t]           (14.6, tabular)
        v <- v + alpha * delta_t * z_t                    (14.4, tabular)

    Layout: state 0 = ITI (frozen), states 1..n_states = trial time steps,
    CS onset at state 1, reward (US) enters state us_index.

    Args:
        n_states: Number of trial time steps (states 1..n_states).
        alpha: TD step size.
        gamma: Discount factor (book assumes ~1; default 1.0).
        lam: Eligibility trace decay (book default 0.95; Section 14.2.4).
        With lam = 0 this degenerates to TD(0).
    """

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> None:
        """init (reference implementation).


        Args:
            n_states: See the relevant book section and function body.
            alpha: See the relevant book section and function body.
            gamma: See the relevant book section and function body.
            lam: See the relevant book section and function body.

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15)."""
        if n_states < 1:
            raise ValueError("n_states must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        self.n_states = int(n_states)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lam = float(lam)
        # Index 0 is the frozen ITI state (value and trace always 0).
        self.values = np.zeros(self.n_states + 1, dtype=np.float64)
        self.traces = np.zeros(self.n_states + 1, dtype=np.float64)

    @property
    def iti_value(self) -> float:
        """Frozen ITI state value (0 by construction, episodic assumption).


        Returns:
            iti value as defined in Section 15.6.

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15)."""
        # carry out the iti value step (book Chapter 15).
        return 0.0

    def begin_trial(self) -> None:
        """Reset eligibility traces at the start of a trial.

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15)."""
        # reset trial-scoped state before a new conditioning trial (book Chapter 15).
        self.traces = np.zeros(self.n_states + 1, dtype=np.float64)

    def step(
        self,
        state: int,
        next_state: int | None,
        reward: float,
    ) -> float:
        """One TD(lambda) update from `state`; return the TD error delta.

        Args:
            state: Current state index (0 = ITI, never updated).
            next_state: Next state index, or None for the terminal
                transition after the last trial state (bootstraps v = 0).
            reward: R_{t+1}, the target observed on the transition.

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15).

        Returns:
            step as defined in Section 15.6."""
        # next value: value chosen conditionally (book Chapter 15).
        next_value = 0.0 if next_state is None else self.values[next_state]
        # delta: TD error or update difference for this step (book Chapter 15).
        delta = reward + self.gamma * next_value - self.values[state]
        self.traces = self.gamma * self.lam * self.traces
        if state != 0:  # ITI state never earns eligibility (Section 15.6 assumption)
            self.traces[state] += 1.0
            # self.values: add value computed from its operands into the running value (book ...
            self.values += self.alpha * delta * self.traces
        return float(delta)

    def run_trial(self, us_index: int, reward: float = 1.0) -> np.ndarray:
        """Run one trial; return per-transition TD errors.

        Trial sequence: ITI(0) -> 1 -> 2 -> ... -> n_states -> terminal.
        R = reward enters state us_index (1 <= us_index <= n_states).

        Returns:
            np.ndarray of length n_states + 1: deltas for transitions
            0->1, 1->2, ..., n_states->terminal.

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15).

        Args:
            us_index: See Section 15.6 and the implementation for its role.
            reward: See Section 15.6 and the implementation for its role."""
        # run one experimental trial and return its measurements (book Chapter 15).
        if not 1 <= us_index <= self.n_states:
            raise ValueError("us_index must be in [1, n_states]")
        self.begin_trial()
        # deltas: per-step TD errors or differences (book Chapter 15).
        deltas = np.empty(self.n_states + 1, dtype=np.float64)
        # Iterate over k: one pass per state (book algorithm).
        for k in range(self.n_states):
            # entering: value computed from its operands (book Chapter 15).
            entering = k + 1
            # r: value chosen conditionally (book Chapter 15).
            r = float(reward) if entering == us_index else 0.0
            deltas[k] = self.step(k, k + 1, r)
        deltas[self.n_states] = self.step(self.n_states, None, 0.0)
        return deltas

    def cs_delta(self, deltas: np.ndarray) -> float:
        """Delta on the ITI->CS transition (the 'CS response').

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15).

        Args:
            deltas: See Section 15.6 and the implementation for its role.

        Returns:
            cs delta as defined in Section 15.6."""
        # carry out the cs delta step (book Chapter 15).
        return float(deltas[0])

    def us_delta(self, deltas: np.ndarray, us_index: int) -> float:
        """Delta on the transition into the US state (the 'US response').

        Transition into state us_index is transition index us_index - 1.

        Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15).

        Args:
            deltas: See Section 15.6 and the implementation for its role.
            us_index: See Section 15.6 and the implementation for its role.

        Returns:
            us delta as defined in Section 15.6."""
        # carry out the us delta step (book Chapter 15).
        return float(deltas[us_index - 1])


def _run_and_summarize_trial(
    model: PhasicDopamineModel, us_index: int, reward: float,
) -> tuple[np.ndarray, float, float, int]:
    """Run one trial and collect the four summaries recorded in the history.

    The peak is taken over |delta| because the early US response and the later
    dip at an omitted US are both large, but opposite in sign.

    Book: Sutton & Barto (2nd ed.), Algorithm 15.1 (Chapter 15).

    Args:
        model: See Algorithm 15.1 and the implementation for its role.
        us_index: See Algorithm 15.1 and the implementation for its role.
        reward: See Algorithm 15.1 and the implementation for its role.

    Returns:
        run and summarize trial as defined in Algorithm 15.1."""
    # run the experiment and collect the figure data (book Chapter 15).
    # deltas: per-step TD errors or differences (book Chapter 15).
    deltas = model.run_trial(us_index, reward)
    return (
        deltas,
        model.cs_delta(deltas),
        model.us_delta(deltas, us_index),
        int(np.argmax(np.abs(deltas))),
    )


def simulate_conditioning(
    n_trials: int = 60,
    n_states: int = 20,
    us_index: int = 15,
    reward: float = 1.0,
    alpha: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.95,
    omit_trials: tuple[int, ...] = (),
) -> dict:
    """Algorithm 15.1 — simulate the Section 15.6 idealized task.

    Book Section 15.6, Figure 15.5. An agent learns to predict a reward R*
    that arrives `us_index` steps into each trial (e.g., an instruction cue
    -> trigger cue -> juice sequence). The TD error, standing in for phasic
    dopamine, is tracked across trials. Reward omission trials can be
    interleaved with `omit_trials` to probe the 'reward omitted' condition
    of Figure 15.5 ('R omitted' panel).

    Args:
        n_trials: Number of training trials.
        n_states: Trial length in time steps.
        us_index: State into which the US reward enters.
        reward: US magnitude (R*).
        alpha: TD step size.
        gamma: Discount factor.
        lam: Eligibility trace decay (0.95 book default; 0 = TD(0)).
        omit_trials: Indices of trials in which the US is omitted
            (reward = 0), to simulate the 'reward omitted' condition of
            Figure 15.5 ('R omitted' panel).

    Returns:
        dict with keys: 'deltas' (n_trials x n_transitions array),
        'cs_delta' and 'us_delta' (per-trial arrays), 'peak_transition'
        (per-trial argmax |delta|), 'values' (final state values),
        'us_index', 'n_states', 'reward'.
    """
    # carry out the simulate conditioning step (book Chapter 15).
    # model: result of PhasicDopamineModel(...) (book Chapter 15).
    model = PhasicDopamineModel(n_states, alpha=alpha, gamma=gamma, lam=lam)
    # deltas history: result of np.empty(...) (book Chapter 15).
    deltas_history = np.empty((n_trials, n_states + 1), dtype=np.float64)
    # cs deltas: result of np.empty(...) (book Chapter 15).
    cs_deltas = np.empty(n_trials, dtype=np.float64)
    # us deltas: result of np.empty(...) (book Chapter 15).
    us_deltas = np.empty(n_trials, dtype=np.float64)
    # peaks: result of np.empty(...) (book Chapter 15).
    peaks = np.empty(n_trials, dtype=np.int64)

    # Iterate over trial: one pass per index (book algorithm).
    for trial in range(n_trials):
        # r: value chosen conditionally (book Chapter 15).
        r = 0.0 if trial in omit_trials else float(reward)
        # (deltas, cs, us, peak): result of _run_and_summarize_trial(...) (book Chapter 15).
        deltas, cs, us, peak = _run_and_summarize_trial(model, us_index, r)
        deltas_history[trial] = deltas
        cs_deltas[trial] = cs
        us_deltas[trial] = us
        peaks[trial] = peak

    return {
        "deltas": deltas_history,
        "cs_delta": cs_deltas,
        "us_delta": us_deltas,
        "peak_transition": peaks,
        "values": model.values.copy(),
        "us_index": us_index,
        "n_states": n_states,
        "reward": float(reward),
    }


def simulate_conditioning_trace(
    n_acquisition: int = 40,
    n_extinction: int = 30,
    n_reacquisition: int = 20,
    n_states: int = 20,
    us_index: int = 15,
    reward: float = 1.0,
    alpha: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> dict:
    """Algorithm 15.2 — full conditioning trace with extinction and reinsertion.

    Book Sections 15.5–15.6 (Figures 15.3–15.5). Runs three phases:

    1. acquisition   — CS -> US trials; the phasic response to the US
       collapses and a response to the CS grows (cue shift; Schultz et al.,
       1993),
    2. extinction    — CS alone (US omitted); the response at the expected
       US time dips below baseline (negative delta),
    3. reacquisition — CS -> US restored; the response reappears.

    Returns:
        dict with keys: 'cs_delta' and 'us_delta' (per-trial arrays over
        the whole run), 'phase' (per-trial phase label), 'boundaries'
        ((n_acquisition, n_acquisition + n_extinction)), 'values' (final
        state values), 'us_index', 'n_states', 'reward'.

    Book: Sutton & Barto (2nd ed.), Algorithm 15.2 (Chapter 15).

    Args:
        n_acquisition: See Algorithm 15.2 and the implementation for its role.
        n_extinction: See Algorithm 15.2 and the implementation for its role.
        n_reacquisition: See Algorithm 15.2 and the implementation for its role.
        n_states: See Algorithm 15.2 and the implementation for its role.
        us_index: See Algorithm 15.2 and the implementation for its role.
        reward: See Algorithm 15.2 and the implementation for its role.
        alpha: See Algorithm 15.2 and the implementation for its role.
        gamma: See Algorithm 15.2 and the implementation for its role.
        lam: See Algorithm 15.2 and the implementation for its role."""
    # carry out the simulate conditioning trace step (book Chapter 15).
    if n_acquisition < 1 or n_extinction < 1 or n_reacquisition < 1:
        raise ValueError("phase lengths must be positive")
    # model: result of PhasicDopamineModel(...) (book Chapter 15).
    model = PhasicDopamineModel(n_states, alpha=alpha, gamma=gamma, lam=lam)
    # total: running total accumulator (book Chapter 15).
    total = n_acquisition + n_extinction + n_reacquisition
    # cs deltas: result of np.empty(...) (book Chapter 15).
    cs_deltas = np.empty(total, dtype=np.float64)
    # us deltas: result of np.empty(...) (book Chapter 15).
    us_deltas = np.empty(total, dtype=np.float64)
    # phases: list[str] attribute/state used by the algorithm (book Chapter 15).
    phases: list[str] = []

    # schedule: value computed from its operands (book Chapter 15).
    schedule = (
        [("acquisition", reward)] * n_acquisition
        + [("extinction", 0.0)] * n_extinction
        + [("reacquisition", reward)] * n_reacquisition
    )
    # Iterate over (trial, (phase, r)): indexed entries (book algorithm).
    for trial, (phase, r) in enumerate(schedule):
        # deltas: per-step TD errors or differences (book Chapter 15).
        deltas = model.run_trial(us_index, r)
        cs_deltas[trial] = model.cs_delta(deltas)
        us_deltas[trial] = model.us_delta(deltas, us_index)
        phases.append(phase)

    return {
        "cs_delta": cs_deltas,
        "us_delta": us_deltas,
        "phase": phases,
        "boundaries": (n_acquisition, n_acquisition + n_extinction),
        "values": model.values.copy(),
        "us_index": us_index,
        "n_states": n_states,
        "reward": float(reward),
    }


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 15)
# ---------------------------------------------------------------------------

# Fixed demo settings for the Fig. 15.5-like panels. The simulation API's own
# defaults stay untouched; these name the values the demo has always used.
_DEMO_ALPHA = 0.1
_DEMO_GAMMA = 1.0
_DEMO_LAM = 0.95
_DEMO_ACQUISITION_TRIALS = 50
_DEMO_OMITTED_TRIAL = 50
_DEMO_TOTAL_TRIALS = 51


def _run_demo_conditioning(
    n_trials: int,
    n_states: int,
    us_index: int,
    omit_trials: tuple[int, ...] = (),
) -> dict:
    """Run the demo's fixed-parameter conditioning simulation.

    The omission panel needs the same learned values as acquisition plus one
    extra trial, hence a fresh run with the last trial's reward removed.

    Book: Sutton & Barto (2nd ed.), Algorithm 15.1 (Chapter 15).

    Args:
        n_trials: See Algorithm 15.1 and the implementation for its role.
        n_states: See Algorithm 15.1 and the implementation for its role.
        us_index: See Algorithm 15.1 and the implementation for its role.
        omit_trials: See Algorithm 15.1 and the implementation for its role.

    Returns:
        run demo conditioning as defined in Algorithm 15.1."""
    # run the experiment and collect the figure data (book Chapter 15).
    return simulate_conditioning(
        n_trials=n_trials,
        n_states=n_states,
        us_index=us_index,
        reward=1.0,
        alpha=_DEMO_ALPHA,
        gamma=_DEMO_GAMMA,
        lam=_DEMO_LAM,
        omit_trials=omit_trials,
    )


def _draw_delta_panel(
    ax: Any, deltas: np.ndarray, t: np.ndarray, us_index: int, title: str,
) -> None:
    """Draw one within-trial delta trace with the CS and expected-US markers.

    The two vertical lines are reference times, not data, so every panel uses
    the same styling and the US line is clamped to the plotted range.

    Book: Sutton & Barto (2nd ed.), Algorithm 15.1 (Chapter 15).

    Args:
        ax: See Algorithm 15.1 and the implementation for its role.
        deltas: See Algorithm 15.1 and the implementation for its role.
        t: See Algorithm 15.1 and the implementation for its role.
        us_index: See Algorithm 15.1 and the implementation for its role.
        title: See Algorithm 15.1 and the implementation for its role."""
    # internal helper for the draw delta panel step (book Chapter 15).
    ax.plot(t, deltas, color="#2c7fb8")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#55a868", linestyle="--", linewidth=1, alpha=0.7, label="CS onset")
    ax.axvline(min(us_index, len(deltas) - 1), color="#c44e52", linestyle="--",
               linewidth=1, alpha=0.7, label="US time")
    ax.set_ylabel("TD error (delta)")
    ax.set_title(title)


def plot_phasic_dopamine_conditioning(
    save_path: str | None = None,
    show: bool = False,
    n_states: int = 20,
    us_index: int = 15,
    seed: int = 0,
):
    """Phasic dopamine / TD-error conditioning traces (Sec. 15.6, Fig. 15.5-like).

    Sutton & Barto (2nd ed.), Sec. 15.3 & 15.6, Fig. 15.5 (reward prediction
    error hypothesis of dopamine): TD error delta_t models phasic dopamine during
    classical conditioning with a CSC representation.
      - Early acquisition: positive delta at reward delivery (US), ~0 at cue.
      - Late acquisition: delta shifts to cue onset (CS); ~0 at predicted US.
      - Omission after learning: positive delta at cue, negative dip at omitted US.
    Uses `simulate_conditioning` to obtain within-trial delta traces.

    Book: Sutton & Barto (2nd ed.), Section 15.6 (Chapter 15).

    Args:
        save_path: See Section 15.6 and the implementation for its role.
        show: See Section 15.6 and the implementation for its role.
        n_states: See Section 15.6 and the implementation for its role.
        us_index: See Section 15.6 and the implementation for its role.
        seed: See Section 15.6 and the implementation for its role."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Acquisition run: early vs late trial deltas
    acq = _run_demo_conditioning(_DEMO_ACQUISITION_TRIALS, n_states, us_index)
    # early: result of np.asarray(...) (book Chapter 15).
    early = np.asarray(acq["deltas"][0], dtype=np.float64)
    # late: result of np.asarray(...) (book Chapter 15).
    late = np.asarray(acq["deltas"][-1], dtype=np.float64)

    # Train then omit reward to show dip at expected US time
    omit = _run_demo_conditioning(
        _DEMO_TOTAL_TRIALS, n_states, us_index, omit_trials=(_DEMO_OMITTED_TRIAL,),
    )
    # omission: result of np.asarray(...) (book Chapter 15).
    omission = np.asarray(omit["deltas"][-1], dtype=np.float64)

    # t: array of evenly spaced values (book Chapter 15).
    t = np.arange(len(early))
    # (fig, axes): result of plt.subplots(...) (book Chapter 15).
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    # panels: collection of the listed elements (book Chapter 15).
    panels = [
        (axes[0], early, "Early acquisition: delta at reward (US)"),
        (axes[1], late, "Late acquisition: delta shifts to cue (CS)"),
        (axes[2], omission, "Omission after learning: dip at omitted US"),
    ]
    # Iterate over (ax, d, title): each item in panels (book algorithm).
    for ax, d, title in panels:
        _draw_delta_panel(ax, d, t, us_index, title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time step within trial")
    fig.suptitle("S&B (2nd ed.), Sec. 15.6 (Fig. 15.5-like): Phasic dopamine / TD error")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes
