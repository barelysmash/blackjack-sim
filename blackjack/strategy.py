"""Playing strategy: book basic strategy plus Hi-Lo count deviations.

Tables encode 6-deck, S17, DAS, no-surrender basic strategy — the same
game the legacy matrices targeted, with the known legacy errors corrected
(e.g. split 21 is no longer treated as a natural; split aces take one card).

Deviations implement the "Illustrious 18" index plays plus insurance at
true count >= +3. These index plays plus a bet spread are where virtually
all of a counter's edge comes from; the legacy simulator bet with the
count but never varied its play, leaving roughly a third of the available
EV on the table.

Actions
-------
H  hit
S  stand
D  double if allowed, else hit
Ds double if allowed, else stand
P  split
"""

from __future__ import annotations

from typing import List, Optional

from .cards import hand_total

H, S, D, DS, P = "H", "S", "D", "Ds", "P"

# Dealer upcard columns: 2 3 4 5 6 7 8 9 10 A  -> index = upcard - 2 (A=11 -> 9)
_COL = {u: u - 2 for u in range(2, 12)}

# Hard totals 5..21 (row index = total - 5)
HARD: List[List[str]] = [
    # 2  3  4  5  6  7  8  9  T  A
    [H, H, H, H, H, H, H, H, H, H],  # 5
    [H, H, H, H, H, H, H, H, H, H],  # 6
    [H, H, H, H, H, H, H, H, H, H],  # 7
    [H, H, H, H, H, H, H, H, H, H],  # 8
    [H, D, D, D, D, H, H, H, H, H],  # 9
    [D, D, D, D, D, D, D, D, H, H],  # 10
    [D, D, D, D, D, D, D, D, D, H],  # 11 (S17: hit vs A)
    [H, H, S, S, S, H, H, H, H, H],  # 12
    [S, S, S, S, S, H, H, H, H, H],  # 13
    [S, S, S, S, S, H, H, H, H, H],  # 14
    [S, S, S, S, S, H, H, H, H, H],  # 15
    [S, S, S, S, S, H, H, H, H, H],  # 16
    [S, S, S, S, S, S, S, S, S, S],  # 17
    [S, S, S, S, S, S, S, S, S, S],  # 18
    [S, S, S, S, S, S, S, S, S, S],  # 19
    [S, S, S, S, S, S, S, S, S, S],  # 20
    [S, S, S, S, S, S, S, S, S, S],  # 21
]

# Soft totals A2..A9 (soft 13..20; row index = non-ace value - 2)
SOFT: List[List[str]] = [
    # 2  3   4   5   6   7  8  9  T  A
    [H, H, H, D, D, H, H, H, H, H],   # A,2 (13)
    [H, H, H, D, D, H, H, H, H, H],   # A,3 (14)
    [H, H, D, D, D, H, H, H, H, H],   # A,4 (15)
    [H, H, D, D, D, H, H, H, H, H],   # A,5 (16)
    [H, D, D, D, D, H, H, H, H, H],   # A,6 (17)
    [DS, DS, DS, DS, DS, S, S, H, H, H],  # A,7 (18)
    [S, S, S, S, S, S, S, S, S, S],   # A,8 (19)
    [S, S, S, S, S, S, S, S, S, S],   # A,9 (20)
]

# Pairs by paired card value 2..11 (row index = value - 2). DAS assumed.
PAIRS: List[List[str]] = [
    # 2  3  4  5  6  7  8  9  T  A
    [P, P, P, P, P, P, H, H, H, H],  # 2,2
    [P, P, P, P, P, P, H, H, H, H],  # 3,3
    [H, H, H, P, P, H, H, H, H, H],  # 4,4
    [D, D, D, D, D, D, D, D, H, H],  # 5,5 (play as hard 10)
    [P, P, P, P, P, H, H, H, H, H],  # 6,6
    [P, P, P, P, P, P, H, H, H, H],  # 7,7
    [P, P, P, P, P, P, P, P, P, P],  # 8,8
    [P, P, P, P, P, S, P, P, S, S],  # 9,9
    [S, S, S, S, S, S, S, S, S, S],  # T,T
    [P, P, P, P, P, P, P, P, P, P],  # A,A
]

# Illustrious 18 index plays (multi-deck S17, no surrender), plus the
# two negative "stand unless the count drops" indices. Each entry:
# (hard_total, dealer_up): (index, action_at_or_above, action_below)
# Doubling entries fall back to hit below the index via basic tables.
DEVIATIONS = {
    (16, 10): (0, S, H),
    (16, 9):  (5, S, H),
    (15, 10): (4, S, H),
    (13, 2):  (-1, S, H),   # stand unless TC < -1
    (13, 3):  (-2, S, H),
    (12, 2):  (3, S, H),
    (12, 3):  (2, S, H),
    (12, 4):  (0, S, H),    # stand unless TC < 0
    (12, 5):  (-2, S, H),
    (12, 6):  (-1, S, H),
    (11, 11): (1, D, H),    # 11 vs A
    (10, 10): (4, D, H),
    (10, 11): (4, D, H),
    (9, 2):   (1, D, H),
    (9, 7):   (3, D, H),
    (8, 6):   (2, D, H),
}
SPLIT_TENS = {5: 5, 6: 4}       # T,T vs 5 at TC>=5; vs 6 at TC>=4 (off by default)
INSURANCE_INDEX = 3.0           # take insurance at TC >= +3


class BasicStrategy:
    """Book basic strategy; optionally applies count deviations."""

    def __init__(self, use_deviations: bool = False, split_tens: bool = False):
        self.use_deviations = use_deviations
        self.split_tens = split_tens

    def take_insurance(self, true_count: float) -> bool:
        return self.use_deviations and true_count >= INSURANCE_INDEX

    def decide(
        self,
        cards: List[int],
        dealer_up: int,
        can_double: bool,
        can_split: bool,
        true_count: float = 0.0,
    ) -> str:
        """Return one of H/S/D(as hit fallback)/P resolved for legality."""
        col = _COL[dealer_up]
        total, soft = hand_total(cards)
        is_pair = len(cards) == 2 and cards[0] == cards[1]

        action: Optional[str] = None

        # Count deviations take precedence over the book play (hard hands
        # and T,T splits only, per Illustrious 18).
        if self.use_deviations and not soft:
            if is_pair and cards[0] == 10 and self.split_tens and can_split:
                idx = SPLIT_TENS.get(dealer_up)
                if idx is not None and true_count >= idx:
                    return P
            dev = DEVIATIONS.get((total, dealer_up))
            if dev is not None and not (is_pair and cards[0] != 5):
                idx, hi, lo = dev
                action = hi if true_count >= idx else lo

        if action is None:
            if is_pair and can_split and cards[0] != 5:
                action = PAIRS[cards[0] - 2][col]
            elif soft and total < 21:
                # Row keyed by "the rest of the hand" beside one ace-as-11.
                non_ace = total - 11
                if 2 <= non_ace <= 9:
                    action = SOFT[non_ace - 2][col]
                else:
                    action = S
            else:
                action = HARD[total - 5][col]

        # Resolve legality fallbacks.
        if action == P and not can_split:
            is_pair2 = False  # treat as its hard/soft total
            total2, soft2 = hand_total(cards)
            if soft2 and 2 <= total2 - 11 <= 9:
                action = SOFT[total2 - 11 - 2][col]
            else:
                action = HARD[total2 - 5][col]
        if action == D and not can_double:
            action = H
        if action == DS:
            action = D if can_double else S
        if action == D and not can_double:
            action = H
        return action
