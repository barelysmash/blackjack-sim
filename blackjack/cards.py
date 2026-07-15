"""Card and shoe primitives, optimized for Monte Carlo throughput.

Cards are plain ints (2-11, where 11 = Ace and all T/J/Q/K collapse to 10).
Suits and faces are irrelevant to EV, so we don't model them: this makes
dealing a list.pop() on ints and hand evaluation pure integer math,
roughly an order of magnitude faster than object-per-card designs.
"""

from __future__ import annotations

import random
from typing import List

# One 52-card deck by value: 4 each of 2-9 and Ace(11), 16 tens.
_DECK: List[int] = (
    [v for v in range(2, 10) for _ in range(4)]  # 2-9
    + [10] * 16                                   # T, J, Q, K
    + [11] * 4                                    # Aces
)


def build_shoe(num_decks: int, rng: random.Random) -> List[int]:
    """Return a shuffled shoe of ``num_decks`` decks."""
    shoe = _DECK * num_decks
    rng.shuffle(shoe)
    return shoe


def hand_total(cards: List[int]) -> tuple[int, bool]:
    """Return (best_total, is_soft) for a hand.

    Aces enter as 11 and are demoted to 1 while the hand would bust.
    ``is_soft`` is True when at least one ace still counts as 11.
    """
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


def is_blackjack(cards: List[int]) -> bool:
    """Natural 21: exactly two cards totalling 21."""
    return len(cards) == 2 and sum(cards) == 21
