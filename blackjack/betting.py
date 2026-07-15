"""Bet sizing strategies keyed to the true count.

The MIT approach in one line: play perfect basic strategy, size the bet
in proportion to the current advantage. Advantage in a 6-deck S17 game is
roughly ``-0.5% + 0.5% per point of true count``, so the count is a direct
edge meter. Three sizers are provided:

* ``FlatBet``        — baseline / control group
* ``SpreadBet``      — discrete unit ramp by true count (table-friendly)
* ``KellyBet``       — bankroll-proportional, edge/variance (the theory
                       the MIT ramp approximated; default half-Kelly)

All sizers may return 0 units, which the engine treats as "wong out":
sit out the round while the table burns cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Per-round variance of a basic-strategy blackjack hand, in squared units.
ROUND_VARIANCE = 1.32
BASE_EDGE = -0.005          # house edge at TC 0 (6D, S17, DAS)
EDGE_PER_TC = 0.005         # advantage gained per true count point


def estimated_edge(true_count: float) -> float:
    return BASE_EDGE + EDGE_PER_TC * true_count


@dataclass
class FlatBet:
    units: float = 1.0

    def bet(self, true_count: float, bankroll: float, min_bet: float) -> float:
        return min(min_bet * self.units, bankroll)


@dataclass
class SpreadBet:
    """Discrete ramp: units bet at each floored true count.

    Default is a 1-12 spread appropriate for 6 decks. Counts below
    ``wong_out_below`` sit out entirely (0 units) if set.
    """

    ramp: Dict[int, float] = field(
        default_factory=lambda: {0: 1, 1: 2, 2: 4, 3: 8, 4: 10, 5: 12}
    )
    wong_out_below: float | None = None   # e.g. -1.0 to leave bad shoes

    def bet(self, true_count: float, bankroll: float, min_bet: float) -> float:
        if self.wong_out_below is not None and true_count < self.wong_out_below:
            return 0.0
        tc = int(true_count) if true_count >= 0 else 0
        tc = min(tc, max(self.ramp))
        units = self.ramp.get(tc, 1)
        return min(min_bet * units, bankroll)


@dataclass
class KellyBet:
    """Fractional Kelly: bet = bankroll * fraction * edge / variance.

    ``fraction=0.5`` (half-Kelly) trades a little growth for a large
    reduction in risk of ruin — the standard practical choice.
    """

    fraction: float = 0.5
    max_units: float = 20.0
    wong_out_below: float | None = None

    def bet(self, true_count: float, bankroll: float, min_bet: float) -> float:
        if self.wong_out_below is not None and true_count < self.wong_out_below:
            return 0.0
        edge = estimated_edge(true_count)
        if edge <= 0:
            return min(min_bet, bankroll)
        raw = bankroll * self.fraction * edge / ROUND_VARIANCE
        units = max(1.0, min(raw / min_bet, self.max_units))
        # Round to whole units to look like a human bettor.
        return min(min_bet * round(units), bankroll)
