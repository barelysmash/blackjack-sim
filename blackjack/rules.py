"""Table rules configuration.

Defaults model the classic MIT-team target game: 6-deck shoe, dealer
stands on soft 17 (S17), blackjack pays 3:2, double after split allowed,
split aces receive one card, ~75% penetration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    num_decks: int = 6
    dealer_stands_soft_17: bool = True   # S17
    blackjack_payout: float = 1.5        # 3:2
    double_after_split: bool = True      # DAS
    max_hands: int = 4                   # split to at most 4 hands
    resplit_aces: bool = False
    hit_split_aces: bool = False         # split aces get exactly one card
    penetration: float = 0.75            # fraction of shoe dealt before shuffle
    dealer_peeks: bool = True            # US peek for blackjack
    allow_insurance: bool = True

    @property
    def reshuffle_at(self) -> int:
        """Cards remaining that triggers a reshuffle."""
        return int(self.num_decks * 52 * (1.0 - self.penetration))
