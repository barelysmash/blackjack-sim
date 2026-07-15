"""Hi-Lo card counting.

Fixes the core defect in the legacy simulator: it accumulated a *running*
count but labeled and used it as a *true* count. Bet sizing and play
deviations are calibrated to the true count — running count divided by
decks remaining — so the legacy bet ramp fired on the wrong signal
(a +6 running count deep in a shoe is very different from +6 off the top).
"""

from __future__ import annotations

# Hi-Lo tag per card value (index by card value 2..11).
_HILO = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1}


class HiLoCounter:
    """Tracks running count; converts to true count on demand."""

    __slots__ = ("running",)

    def __init__(self) -> None:
        self.running = 0

    def reset(self) -> None:
        self.running = 0

    def see(self, card: int) -> None:
        """Register a card as it becomes visible."""
        self.running += _HILO[card]

    def true_count(self, cards_remaining: int) -> float:
        """Running count normalized per remaining deck.

        Decks remaining is floored at half a deck so end-of-shoe division
        doesn't explode.
        """
        decks = max(cards_remaining / 52.0, 0.5)
        return self.running / decks
