"""Solution implementation for Chapter 14 - Psychology.

Reference: Sutton & Barto (2nd Ed), Chapter 14 (Psychology).

Implements:
- Alg 14.1 — Rescorla-Wagner model (Eqs. 14.1-14.3, Section 14.2.2)
- Alg 14.2 — TD model of classical conditioning (Eqs. 14.4-14.6, Section 14.2.3)
- Alg 14.3 — Kamin blocking experiment runner (Sections 14.2.1-14.2.2)

These are *models of animal learning*, not just algorithms. Where the book
describes a model rather than an executable algorithm, the docstrings say
so explicitly. The Rescorla-Wagner model is a trial-level model: Eqs.
(14.1)-(14.3) update associative strengths once per conditioning trial. The
TD model is a real-time model: Eqs. (14.4)-(14.6) update associative
strengths at each time step within/between trials.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ============================================================================
# Algorithm 14.1: Rescorla-Wagner model
# Book Section 14.2.2, Eqs. 14.1-14.3
# ============================================================================


class RescorlaWagnerModel:
    """Rescorla-Wagner model of classical conditioning.

    Book Section 14.2.2. Trial-level model: one update per conditioning
    trial. The aggregate associative strength of a compound CS AX is
    v(AX) = w_A + w_X (Eq. 14.1 with binary presence features), and each
    component's strength changes by the same prediction error, scaled by
    its own salience (Eqs. 14.2-14.3):

        v(s, w) = w^T x(s)                      (14.1)
        w_{t+1} = w_t + alpha * delta_t * x(S_t)  (14.2)
        delta_t = R_t - v(S_t, w_t)             (14.3)

    This is essentially the LMS/Widrow-Hoff rule applied to US prediction.

    Args:
        n_stimuli: Number of distinct conditionable stimuli.
        alpha: Step size (shared by all stimuli, unless saliences given).
        saliences: Optional per-stimulus step-size multipliers (the alpha_i
            of the Rescorla-Wagner formalism). If provided, it overrides
            alpha per stimulus; its length must match n_stimuli.
        lambda_us: Asymptotic associative strength supportable by the US
            (the R_Y / lambda of the book; default 1.0).
    """

    def __init__(
        self,
        n_stimuli: int,
        alpha: float = 0.1,
        saliences: Sequence[float] | None = None,
        lambda_us: float = 1.0,
    ) -> None:
        if n_stimuli < 1:
            raise ValueError("n_stimuli must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if saliences is not None:
            saliences_arr = np.asarray(saliences, dtype=np.float64)
            if saliences_arr.shape != (n_stimuli,):
                raise ValueError("saliences must have one entry per stimulus")
            if np.any(saliences_arr <= 0.0):
                raise ValueError("saliences must be positive")
            self.alphas = alpha * saliences_arr
        else:
            self.alphas = np.full(n_stimuli, alpha, dtype=np.float64)
        self.lambda_us = float(lambda_us)
        self.weights = np.zeros(n_stimuli, dtype=np.float64)

    def predict(self, x: Sequence[float]) -> float:
        """Aggregate associative strength v(s, w) = w^T x(s). Eq. (14.1)."""
        return float(self.weights @ np.asarray(x, dtype=np.float64))

    def trial(self, x: Sequence[float], target: float | None = None) -> float:
        """Run one conditioning trial; return the prediction error delta.

        Eqs. (14.2)-(14.3). Only the strengths of stimuli present on the
        trial (non-zero components of x) are adjusted.

        Args:
            x: Binary (or real-valued) feature vector for the trial's
                compound CS.
            target: US magnitude for this trial; defaults to lambda_us.
                Pass 0.0 to model extinction (US omitted).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != self.weights.shape:
            raise ValueError("x must have one component per stimulus")
        target_value = self.lambda_us if target is None else float(target)
        delta = target_value - self.predict(x_arr)
        self.weights = self.weights + self.alphas * delta * x_arr
        return float(delta)

    def strength(self, index: int) -> float:
        """Associative strength of one stimulus (its component of w)."""
        return float(self.weights[index])

    def run_trials(
        self,
        trials: Sequence[tuple[Sequence[float], float]],
    ) -> list[float]:
        """Run many trials; return the list of per-trial prediction errors.

        Args:
            trials: Sequence of (x, target) pairs.
        """
        return [self.trial(x, target) for x, target in trials]


def excitation_curve(
    model: RescorlaWagnerModel,
    x: Sequence[float],
    n_trials: int,
    target: float | None = None,
) -> np.ndarray:
    """Acquisition curve: v(compound) after each of n_trials pairings.

    Returns the aggregate associative strength of the compound after every
    trial (length n_trials). Book Section 14.2.2: strengths grow toward
    the asymptote R_Y and stop changing when v = R_Y.
    """
    history = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        model.trial(x, target)
        history[i] = model.predict(x)
    return history


# ============================================================================
# Algorithm 14.2: TD model of classical conditioning
# Book Section 14.2.3, Eqs. 14.4-14.6
# ============================================================================


class TDConditioningModel:
    """TD model of classical conditioning (real-time model).

    Book Section 14.2.3. Unlike the trial-level Rescorla-Wagner model,
    t labels time steps within a trial rather than whole trials, and the
    update uses eligibility traces and a TD error:

        w_{t+1} = w_t + alpha * delta_t * z_t          (14.4)
        delta_t = R_{t+1} + gamma * v_hat(S_{t+1}, w_t) - v_hat(S_t, w_t)  (14.5)
        z_{t+1} = gamma * lambda * z_t + x(S_t)        (14.6)

    With gamma = 0 the model reduces to the Rescorla-Wagner model (modulo
    the one-time-step lead in the prediction target). The update below
    applies the trace accumulation trace <- gamma*lambda*trace + x(s_t)
    before the weight update, the standard backward-view reading of
    Eqs. (14.4)-(14.6), matching the eligibility-trace update of Chapter 6.

    The model is representation-agnostic ("the TD model does not commit to
    any particular state representation"). Helpers build the presence and
    complete-serial-compound (CSC) representations of Figure 14.2.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 0.05,
        gamma: float = 0.97,
        lam: float = 0.95,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.traces = np.zeros(n_features, dtype=np.float64)

    def predict(self, x: Sequence[float]) -> float:
        """Aggregate associative strength v_hat(s, w) = w^T x(s). Eq. (14.1)."""
        return float(self.weights @ np.asarray(x, dtype=np.float64))

    def strength(self, index: int) -> float:
        """Associative strength (weight) of one feature/stimulus."""
        return float(self.weights[index])

    def begin_trial(self) -> None:
        """Reset eligibility traces at the start of a trial."""
        self.traces = np.zeros(self.weights.shape, dtype=np.float64)

    def step(
        self,
        x_t: Sequence[float],
        x_next: Sequence[float],
        reward_next: float,
        terminal: bool = False,
    ) -> float:
        """Advance one time step; return the TD error delta.

        Args:
            x_t: Feature vector of the current state S_t.
            x_next: Feature vector of S_{t+1}.
            reward_next: R_{t+1}, the target observed on the transition.
            terminal: If True, S_{t+1} is terminal and v_hat(S_{t+1}) = 0.
        """
        x_current = np.asarray(x_t, dtype=np.float64)
        x_following = np.asarray(x_next, dtype=np.float64)
        if x_current.shape != self.weights.shape:
            raise ValueError("x_t must have one component per feature")
        if x_following.shape != self.weights.shape:
            raise ValueError("x_next must have one component per feature")
        next_value = 0.0 if terminal else self.predict(x_following)
        delta = reward_next + self.gamma * next_value - self.predict(x_current)
        self.traces = self.gamma * self.lam * self.traces + x_current
        self.weights = self.weights + self.alpha * delta * self.traces
        return float(delta)


def presence_trial(
    n_steps: int,
    cs_onset: int,
    reward_time: int,
    reward: float = 1.0,
    cs_index: int = 0,
    n_stimuli: int = 1,
) -> tuple[list[np.ndarray], list[float], list[bool]]:
    """Build a presence-representation trial.

    Presence representation (Figure 14.2, right column): one feature per
    CS component, valued 1 whenever the component is present.

    Args:
        n_steps: Number of time steps in the trial (states 0..n_steps).
        cs_onset: First time step at which the CS is present.
        reward_time: State index into which the US reward enters
            (R = reward on the transition into this state). Must be in
            [1, n_steps].
        reward: US magnitude (R_Y of the Rescorla-Wagner model).
        cs_index: Which stimulus feature the CS drives.
        n_stimuli: Total number of stimuli (features).

    Returns:
        (features, rewards, terminals): features has n_steps + 1 entries
        (states 0..n_steps); rewards[k] is R_{k+1} for the transition from
        state k to k + 1, except the final entry bootstraps the terminal
        state; terminals[k] flags the terminal transition.
    """
    if not 1 <= reward_time <= n_steps:
        raise ValueError("reward_time must be in [1, n_steps]")
    if not 0 <= cs_onset < reward_time:
        raise ValueError("cs_onset must be in [0, reward_time)")
    if not 0 <= cs_index < n_stimuli:
        raise ValueError("cs_index must be in [0, n_stimuli)")
    features: list[np.ndarray] = []
    for t in range(n_steps + 1):
        x = np.zeros(n_stimuli, dtype=np.float64)
        if t >= cs_onset:
            x[cs_index] = 1.0
        features.append(x)
    rewards: list[float] = []
    terminals: list[bool] = []
    for k in range(n_steps):
        entering = k + 1
        rewards.append(float(reward) if entering == reward_time else 0.0)
        terminals.append(False)
    terminals[-1] = True
    return features, rewards, terminals


def csc_trial(
    n_steps: int,
    cs_onset: int,
    reward_time: int,
    reward: float = 1.0,
) -> tuple[list[np.ndarray], list[float], list[bool]]:
    """Build a complete-serial-compound (CSC) representation trial.

    CSC representation (Figure 14.2, left column): when a stimulus begins,
    a sequence of precisely-timed short-duration internal signals is
    initiated, one per time step until the stimulus ends ("tapped delay
    line"). Each (stimulus, time-since-onset) pair is a distinct feature.

    Returns:
        (features, rewards, terminals) in the same format as
        presence_trial, with n_steps features (one per time step from
        the CS onset onward). The layout here uses time-since-CS-onset
        features only, so every state from cs_onset on is representable.
    """
    if not 1 <= reward_time <= n_steps:
        raise ValueError("reward_time must be in [1, n_steps]")
    if not 0 <= cs_onset < reward_time:
        raise ValueError("cs_onset must be in [0, reward_time)")
    n_features = n_steps + 1
    features: list[np.ndarray] = []
    for t in range(n_steps + 1):
        x = np.zeros(n_features, dtype=np.float64)
        if t >= cs_onset:
            x[t] = 1.0
        features.append(x)
    rewards: list[float] = []
    terminals: list[bool] = []
    for k in range(n_steps):
        entering = k + 1
        rewards.append(float(reward) if entering == reward_time else 0.0)
        terminals.append(False)
    terminals[-1] = True
    return features, rewards, terminals


def run_trial(
    model: TDConditioningModel,
    features: Sequence[np.ndarray],
    rewards: Sequence[float],
    terminals: Sequence[bool],
) -> np.ndarray:
    """Run one trial; return the per-transition TD errors (delta list).

    features has len(rewards) + 1 entries. The final transition bootstraps
    the terminal state (v = 0).
    """
    if len(features) != len(rewards) + 1:
        raise ValueError("features must have one more entry than rewards")
    if len(rewards) != len(terminals):
        raise ValueError("rewards and terminals must have equal length")
    model.begin_trial()
    deltas = np.empty(len(rewards), dtype=np.float64)
    for k, reward in enumerate(rewards):
        deltas[k] = model.step(
            features[k],
            features[k + 1],
            reward,
            terminal=terminals[k],
        )
    return deltas


def train_presence(
    model: TDConditioningModel,
    n_trials: int,
    n_steps: int,
    cs_onset: int,
    reward_time: int,
    reward: float = 1.0,
    cs_index: int = 0,
    n_stimuli: int = 1,
    compound: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Train a model on presence-representation trials; return delta history.

    Args:
        compound: Optional indices of additional stimuli to present as a
            compound with the CS on every trial.
    """
    history: list[np.ndarray] = []
    for _ in range(n_trials):
        features, rewards, terminals = presence_trial(
            n_steps, cs_onset, reward_time, reward, cs_index, n_stimuli,
        )
        if compound:
            for t, x in enumerate(features):
                if t >= cs_onset:
                    for extra_index in compound:
                        x[extra_index] = 1.0
        history.append(run_trial(model, features, rewards, terminals))
    return history


# ============================================================================
# Algorithm 14.3: Blocking effect simulation (compound-cue conditioning)
# Book Sections 14.2.1-14.2.2 (Kamin blocking paradigm)
# ============================================================================


def run_blocking_experiment(
    phase1_trials: int = 40,
    phase2_trials: int = 40,
    alpha: float = 0.1,
    lambda_us: float = 1.0,
    replicates: int = 16,
    us_noise: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Kamin blocking experiment (Book Sections 14.2.1-14.2.2).

    Phase 1: the "blocking" group is conditioned with stimulus A alone
    (A -> US); the "control" group is exposed to A without the US (A -> .).
    Phase 2: both groups receive compound trials A+B -> US.
    Phase 3 (test): B alone; its associative strength after phase 2 is the
    measured conditioning to the blocked cue.

    This is a *simulation of an experimental paradigm* built on the
    Rescorla-Wagner model — the book describes blocking as the phenomenon
    the model was created to explain, and gives no separate executable
    algorithm for it.

    Args:
        phase1_trials: Number of phase-1 trials per group.
        phase2_trials: Number of phase-2 (compound) trials per group.
        alpha: Rescorla-Wagner step size.
        lambda_us: Asymptotic US magnitude (R_Y).
        replicates: Independent noisy replicates per group. Noise is
            applied to the US magnitude of every trial (0 disables it).
        us_noise: Standard deviation of Gaussian US-magnitude noise.
        seed: Seed for the replicate noise stream.

    Returns:
        dict with per-replicate records and summary statistics:
        'blocking_delta_b' / 'control_delta_b' (phase-2 change in B's
        strength, one value per replicate), 'blocking_final_b' /
        'control_final_b' (B's strength after phase 2), and the
        'n_replicates' / 'phase1_trials' / 'phase2_trials' settings.
    """
    if phase1_trials < 0 or phase2_trials < 0:
        raise ValueError("trial counts must be non-negative")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    record: dict[str, list[float]] = {
        "blocking_delta_b": [],
        "blocking_final_b": [],
        "control_delta_b": [],
        "control_final_b": [],
    }

    for _ in range(replicates):
        blocking = RescorlaWagnerModel(2, alpha=alpha, lambda_us=lambda_us)
        control = RescorlaWagnerModel(2, alpha=alpha, lambda_us=lambda_us)

        # Phase 1: blocking group gets A -> US; control group gets A alone.
        for _ in range(phase1_trials):
            noisy_us = lambda_us if us_noise == 0.0 else float(
                lambda_us + rng.normal(0.0, us_noise)
            )
            blocking.trial([1.0, 0.0], noisy_us)
            control.trial([1.0, 0.0], 0.0)

        # Phase 2: both groups get A + B -> US.
        blocking_b_before = blocking.strength(1)
        control_b_before = control.strength(1)
        for _ in range(phase2_trials):
            noisy_us = lambda_us if us_noise == 0.0 else float(
                lambda_us + rng.normal(0.0, us_noise)
            )
            blocking.trial([1.0, 1.0], noisy_us)
            control.trial([1.0, 1.0], noisy_us)

        record["blocking_delta_b"].append(blocking.strength(1) - blocking_b_before)
        record["control_delta_b"].append(control.strength(1) - control_b_before)
        record["blocking_final_b"].append(blocking.strength(1))
        record["control_final_b"].append(control.strength(1))

    return {
        "blocking_delta_b": record["blocking_delta_b"],
        "blocking_final_b": record["blocking_final_b"],
        "control_delta_b": record["control_delta_b"],
        "control_final_b": record["control_final_b"],
        "n_replicates": replicates,
        "phase1_trials": phase1_trials,
        "phase2_trials": phase2_trials,
        "mean_blocking_delta": float(np.mean(record["blocking_delta_b"])),
        "mean_control_delta": float(np.mean(record["control_delta_b"])),
    }


# ---------------------------------------------------------------------------
# Figure reproduction (Sutton & Barto, 2nd ed., Ch. 14)
# ---------------------------------------------------------------------------

def plot_conditioning_and_blocking(
    save_path: str | None = None,
    show: bool = False,
    n_trials: int = 60,
    seed: int = 0,
):
    """Classical conditioning acquisition and blocking demos (Sec. 14.2).

    Sutton & Barto (2nd ed.), Ch. 14 (Psychology), Sec. 14.2.2 (Rescorla-Wagner
    model) and Sec. 14.2.1-14.2.3 (blocking, TD model of conditioning):
      - Left: acquisition curve for a single CS under Rescorla-Wagner
        (associative strength approaches lambda_US over trials).
      - Right: Kamin blocking experiment via `run_blocking_experiment` comparing
        control (B alone in phase 2) vs blocked (B paired with pretrained A);
        blocked B acquires much less association than control, illustrating the
        prediction-error account of blocking.
    This is a demonstration of implemented models/experiments, not a pixel-
    perfect reproduction of a specific book figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    # Acquisition (RW), Sec. 14.2.2
    model = RescorlaWagnerModel(n_stimuli=1, alpha=0.1, lambda_us=1.0)
    strengths = []
    for _ in range(n_trials):
        model.trial(np.array([1.0]), target=1.0)
        strengths.append(model.weights[0])

    # Blocking (RW experiment), Sec. 14.2.1-14.2.2
    blk = run_blocking_experiment(
        phase1_trials=40, phase2_trials=40, alpha=0.1, lambda_us=1.0,
        replicates=30, us_noise=0.0, seed=seed,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot(np.arange(1, n_trials + 1), strengths, color="#1f77b4")
    ax1.set_xlabel("Trials")
    ax1.set_ylabel("Associative strength V(CS)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Acquisition (Rescorla-Wagner)")

    labels = ["Control B", "Blocked B"]
    vals = [float(blk.get("mean_control_delta", np.mean(blk["control_delta_b"]))),
            float(blk.get("mean_blocking_delta", np.mean(blk["blocking_delta_b"])))]
    ctrl_arr = np.asarray(blk["control_delta_b"], dtype=float)
    blk_arr = np.asarray(blk["blocking_delta_b"], dtype=float)
    errs = [float(np.std(ctrl_arr)), float(np.std(blk_arr))]
    ax2.bar(labels, vals, yerr=errs, capsize=5, color=["#4c78a8", "#f58518"], alpha=0.9)
    ax2.set_ylabel("Phase-2 change in V(B)")
    ax2.set_ylim(0, max(1.05, max(vals) + max(errs) + 0.1))
    ax2.set_title("Kamin blocking")

    fig.suptitle("S&B (2nd ed.), Ch. 14 (Sec. 14.2): Classical conditioning demos")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax1, ax2)
