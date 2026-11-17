"""Blackjack (21) environment (Sutton & Barto, 2nd ed., Example 5.1, Sec. 5.1)."""

from __future__ import annotations

import numpy as np

ACTIONS = ("hit", "stick")
N_ACTIONS = len(ACTIONS)


def _card_value(rank: int) -> int:
    """Map a drawn rank (1..13) to its blackjack value (face cards = 10)."""
    return 10 if rank > 10 else rank


class BlackjackEnv:
    """Blackjack (21) environment (Example 5.1). Book Section 5.1.

    Observation: (player_sum, dealer_showing, usable_ace).
    Actions: 'hit' or 'stick'. Reward +1 win, -1 loss, 0 draw; gamma = 1.
    Cards are drawn from an infinite deck. The dealer hits on < 17 and sticks
    on >= 17. Naturals (21 on the deal) end the episode immediately: a player
    natural wins unless the dealer also has a natural (draw).
    """

    ACTIONS = ACTIONS

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._player_cards: list[int] = []
        self._dealer_cards: list[int] = []
        self._forced_reward: float | None = None

    @staticmethod
    def _hand_value(cards: list[int]) -> tuple[int, bool]:
        """Return (best_sum, usable_ace) for a hand. Aces count as 11 or 1."""
        total = sum(_card_value(c) for c in cards)
        aces = sum(1 for c in cards if c == 1)
        usable = aces > 0 and total + 10 <= 21
        total += 10 if usable else 0
        return int(min(total, 99)), bool(usable)

    def _draw(self) -> int:
        return int(self.rng.integers(1, 14))

    def reset(self) -> tuple:
        """Reset; resolve naturals immediately if they occur on the deal."""
        self._player_cards = [self._draw(), self._draw()]
        self._dealer_cards = [self._draw(), self._draw()]
        self._forced_reward = None

        player_sum, _ = self._hand_value(self._player_cards)
        dealer_sum, _ = self._hand_value(self._dealer_cards)
        if player_sum == 21 or dealer_sum == 21:
            player_wins = player_sum == 21 and dealer_sum != 21
            draw = player_sum == 21 and dealer_sum == 21
            self._forced_reward = 0.0 if draw else (1.0 if player_wins else -1.0)
        return self._observation()

    def _observation(self) -> tuple:
        player_sum, usable = self._hand_value(self._player_cards)
        dealer_showing = _card_value(self._dealer_cards[0])
        return (int(player_sum), int(dealer_showing), bool(usable))

    def step(self, action: str) -> tuple:
        """Take one action; returns (reward, observation, done)."""
        if action not in ("hit", "stick"):
            raise ValueError("action must be 'hit' or 'stick'")

        if self._forced_reward is not None:
            return self._forced_reward, self._observation(), True

        if action == "hit":
            self._player_cards.append(self._draw())
            player_sum, _ = self._hand_value(self._player_cards)
            if player_sum > 21:
                return -1.0, self._observation(), True
            return 0.0, self._observation(), False

        # 'stick' – dealer plays: hit until sum >= 17.
        dealer_sum, _ = self._hand_value(self._dealer_cards)
        while dealer_sum < 17:
            self._dealer_cards.append(self._draw())
            dealer_sum, _ = self._hand_value(self._dealer_cards)
        player_sum, _ = self._hand_value(self._player_cards)

        if dealer_sum > 21:
            return 1.0, self._observation(), True
        if player_sum > dealer_sum:
            return 1.0, self._observation(), True
        if player_sum < dealer_sum:
            return -1.0, self._observation(), True
        return 0.0, self._observation(), True
