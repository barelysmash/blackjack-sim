"""Sanity tests. Run: python -m pytest tests/ (or python tests/test_core.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blackjack.betting import FlatBet, SpreadBet
from blackjack.cards import hand_total, is_blackjack
from blackjack.counting import HiLoCounter
from blackjack.engine import Simulator
from blackjack.rules import Rules
from blackjack.strategy import BasicStrategy
from blackjack.stats import summarize


def test_hand_values():
    assert hand_total([11, 10]) == (21, True)
    assert hand_total([11, 11]) == (12, True)
    assert hand_total([11, 5, 10]) == (16, False)   # ace demoted
    assert hand_total([10, 10, 5]) == (25, False)
    assert is_blackjack([11, 10]) and not is_blackjack([7, 7, 7])


def test_true_count_normalizes_by_decks():
    c = HiLoCounter()
    for _ in range(6):
        c.see(5)                       # running +6
    assert c.true_count(6 * 52) == 1.0  # 6 decks left -> TC +1
    assert c.true_count(52) == 6.0      # 1 deck left  -> TC +6


def test_book_plays():
    s = BasicStrategy()
    assert s.decide([10, 6], 10, True, False) == "H"    # 16 v T: hit
    assert s.decide([8, 8], 10, True, True) == "P"      # always split 8s
    assert s.decide([5, 6], 5, True, False) == "D"      # 11 v 5: double
    assert s.decide([11, 7], 9, True, False) == "H"     # soft 18 v 9: hit
    assert s.decide([11, 7], 3, True, False) == "D"     # soft 18 v 3 (Ds)
    assert s.decide([11, 7], 3, False, False) == "S"    # Ds falls back to stand


def test_deviations():
    s = BasicStrategy(use_deviations=True)
    assert s.decide([10, 6], 10, True, False, true_count=1.0) == "S"   # I18
    assert s.decide([10, 6], 10, True, False, true_count=-1.0) == "H"
    assert s.take_insurance(3.2) and not s.take_insurance(2.0)


def test_simulation_edge_is_sane():
    """Flat-bet basic strategy should land near the known ~-0.5% edge."""
    sim = Simulator(Rules(), BasicStrategy(), FlatBet(), seed=7)
    res = sim.run(600, bankroll=1e9, min_bet=1.0, stop_on_ruin=False)
    s = summarize(res.records, 1.0)
    assert s["rounds"] > 15_000
    assert -1.5 < s["edge_pct_of_action"] < 0.5, s["edge_pct_of_action"]


def test_wong_out_burns_rounds():
    sim = Simulator(Rules(), BasicStrategy(),
                    SpreadBet(wong_out_below=99.0), seed=1)  # always out
    res = sim.run(3, bankroll=1e9, min_bet=1.0, stop_on_ruin=False)
    assert res.rounds == 0 and res.wonged_out > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
